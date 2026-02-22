import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from transformers import AutoImageProcessor, CLIPModel
from models.base_model import BaseModel
from utils.metrics import compute_open_set_metrics


class ContrastiveCLIP(BaseModel):
    """
    Contrastive image-image learning using a frozen pretrained CLIP vision encoder
    and a trainable projection head.
    """

    def __init__(self, config, device):
        super().__init__(config, device)

        self.config = config
        self.device = device

        model_name = config.get(
            "clip_model_name",
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        )

        # Image processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Load full CLIP model (we will only use vision_model)
        self.encoder = CLIPModel.from_pretrained(model_name).to(device)
        self.encoder.eval()

        # Freeze CLIP completely
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Vision encoder output dimension
        hidden_dim = self.encoder.vision_model.config.hidden_size
        emb_dim = int(config.get("embedding_dim", 256))

        # Trainable projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

        self.temperature = float(config.get("temperature", 0.07))
        self.to(device)

    # --------------------------------------------------------
    # Encoding
    # --------------------------------------------------------

    def _encode_images(self, images):
        """
        Extract frozen CLIP vision features.
        images: list of PIL images, or a torch Tensor of shape [B, C, H, W] or [C, H, W]
        """
        with torch.no_grad():
            proc = self.image_processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = proc["pixel_values"].to(self.device)

            vision_out = self.encoder.vision_model(pixel_values=pixel_values)
            feats = vision_out.pooler_output  # [B, hidden_dim]

        return feats.float()

    def forward(self, images):
        feats = self._encode_images(images)
        emb = self.projector(feats)
        return emb

    # --------------------------------------------------------
    # Contrastive loss
    # --------------------------------------------------------

    def compute_contrastive_loss(self, z1, z2):
        """
        Symmetric InfoNCE loss.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)

        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_i2j + loss_j2i)

    # --------------------------------------------------------
    # Training / validation
    # --------------------------------------------------------

    def train_step(self, batch):
        """
        batch = (img1_list, img2_list)
        """
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Expected batch = (img1_list, img2_list)")

        img1, img2 = batch[0], batch[1]

        z1 = self(img1)
        z2 = self(img2)

        loss = self.compute_contrastive_loss(z1, z2)
        return loss

    def val_step(self, batch):
        """
        Returns:
            labels: [1, 1, ..., 0, 0]
            preds:  binary predictions
            scores: cosine similarities
        """
        img1, img2 = batch[0], batch[1]

        with torch.no_grad():
            z1 = F.normalize(self(img1), dim=1)
            z2 = F.normalize(self(img2), dim=1)

            sim_pos = F.cosine_similarity(z1, z2, dim=1)
            sim_neg = F.cosine_similarity(
                z1, torch.roll(z2, shifts=1, dims=0), dim=1
            )

        scores = torch.cat([sim_pos, sim_neg]).cpu().tolist()
        labels = [1] * len(sim_pos) + [0] * len(sim_neg)

        thr = float(self.config.get("threshold", 0.5))
        preds = [s > thr for s in scores]

        return labels, preds, scores

    def validate(self, val_loader, split="val", auth_loader=None, ground_truth_map=None, orphan_gt_map=None):
        """
        If split == 'test' and auth_loader is provided, evaluate query-to-ALL-authentic
        classification and retrieval metrics using the gallery from auth_loader.
        Otherwise, falls back to pairwise validation and in-batch retrieval metrics.
        
        Args:
            val_loader: Validation/test data loader.
            split: 'val' or 'test'.
            auth_loader: Gallery loader for retrieval evaluation.
            ground_truth_map: Mapping for connected queries (manipulated -> authentic).
            orphan_gt_map: Set of orphan query paths (no authentic match).
        """
        self.eval()

        if split != "test" or auth_loader is None:
            # Original pairwise validation and retrieval (no gallery)
            y_true, y_pred, y_scores = [], [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating ({split})"):
                    labels, preds, scores = self.val_step(batch)
                    y_true.extend(labels)
                    y_pred.extend(preds)
                    y_scores.extend(scores)

            f1 = f1_score(y_true, y_pred, zero_division=0)

            if split == "test":
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                try:
                    roc_auc = (
                        roc_auc_score(y_true, y_scores)
                        if len(set(y_true)) > 1
                        else float("nan")
                    )
                except Exception:
                    roc_auc = float("nan")

                print("\nClassification Metrics (Test Set):")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                print(f"F1 Score:  {f1:.4f}")
                print(
                    f"ROC-AUC:   {roc_auc:.4f}"
                    if not np.isnan(roc_auc)
                    else "ROC-AUC:   N/A"
                )

                self.compute_retrieval_metrics(val_loader)

            return f1

        # === TEST WITH GALLERY ===
        f1 = self.retrieval_against_gallery(val_loader, auth_loader, ground_truth_map, orphan_gt_map)
        return f1

    # --------------------------------------------------------
    # Gallery-based retrieval + classification
    # --------------------------------------------------------

    @torch.no_grad()
    def _embed_loader(self, loader):
        """
        Embed all images from a loader that yields either:
          - (images, paths), or
          - images only

        Returns:
          embs: [N, D] normalized embeddings
          paths: list[str] or empty list if not provided
        """
        embs, paths = [], []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, pths = batch
            else:
                imgs, pths = batch, None

            z = F.normalize(self(imgs), dim=1)
            embs.append(z)
            if pths is not None:
                paths.extend(pths)

        embs = torch.cat(embs, dim=0) if len(embs) > 0 else torch.empty(0, int(self.config.get("embedding_dim", 256)), device=self.device)
        return embs, paths

    @torch.no_grad()
    def retrieval_against_gallery(self, query_loader, gallery_loader, ground_truth_map=None, orphan_gt_map=None):
        """
        Evaluate queries vs an ALL-authentic gallery using MATCH-A metrics.

        query_loader should yield:
          - (img1, img2, m_path, a_path) when return_paths=True
          - or (img1, img2) without paths (then ground_truth_map is needed)

        gallery_loader should yield:
          - (auth_img, auth_path) covering ALL unique authentic images desired
          
        Args:
            query_loader: Query data loader.
            gallery_loader: Gallery data loader.
            ground_truth_map: Mapping for connected queries (manipulated -> authentic).
            orphan_gt_map: Set of orphan query paths (no authentic match).
        """
        # Build gallery
        G, g_paths = self._embed_loader(gallery_loader)
        if G.numel() == 0:
            print("Gallery is empty; retrieval cannot proceed.")
            return 0.0
        path_to_idx = {p: i for i, p in enumerate(g_paths)} if g_paths else {}

        # Track connected vs orphan queries separately for MATCH-A metrics
        connected_ranks = []  # Ranks for connected queries
        connected_max_scores = []  # Max similarity scores for connected queries
        orphan_max_scores = []  # Max similarity scores for orphan queries
        
        thr = float(self.config.get("threshold", 0.5))

        for batch in tqdm(query_loader, desc="Evaluating queries vs gallery"):
            # Expected (img1, img2, m_paths, a_paths) when return_paths=True
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                img1, img2, m_paths, a_paths = batch[:4]
            else:
                img1, img2 = batch[:2]
                m_paths, a_paths = None, None

            Zq = F.normalize(self(img1), dim=1)     # [B, D]
            sims = torch.matmul(Zq, G.T)            # [B, N]
            
            # Get max similarity scores for each query
            max_scores = sims.max(dim=1)[0]  # [B]

            # Top-1 predictions per query
            top1_idx = torch.argmax(sims, dim=1)    # [B]
            top1_scores = sims.gather(1, top1_idx.view(-1, 1)).squeeze(1).tolist()

            for i in range(Zq.size(0)):
                pred_path = g_paths[int(top1_idx[i].item())] if g_paths else None
                top1_score = float(top1_scores[i])

                # Ground-truth authentic path
                gt_path = None
                if ground_truth_map and m_paths is not None:
                    gt_path = ground_truth_map.get(m_paths[i], None)
                elif a_paths is not None:
                    gt_path = a_paths[i]

                # Check if this is an orphan query (no ground truth)
                is_orphan = False
                if orphan_gt_map and m_paths is not None:
                    is_orphan = m_paths[i] in orphan_gt_map
                elif gt_path is None:
                    is_orphan = True

                if is_orphan:
                    # Orphan query - track max score for abstention metrics
                    orphan_max_scores.append(float(max_scores[i].item()))
                else:
                    # Connected query - track rank for retrieval metrics
                    connected_max_scores.append(float(max_scores[i].item()))
                    
                    # Retrieval metrics (rank of the true authentic in the gallery)
                    if gt_path is not None and gt_path in path_to_idx:
                        tgt = path_to_idx[gt_path]
                        ranking = torch.argsort(sims[i], descending=True)
                        pos = (ranking == tgt).nonzero(as_tuple=True)[0]
                        rank = pos.item() + 1 if len(pos) > 0 else None
                        if rank is not None:
                            connected_ranks.append(rank)

        # ===== Compute MATCH-A Metrics =====
        print("\n" + "="*60)
        print("MATCH-A Evaluation Results")
        print("="*60)
        
        matcha_metrics = compute_open_set_metrics(
            ranks_conn=connected_ranks,
            max_scores_conn=connected_max_scores,
            max_scores_orph=orphan_max_scores,
            thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
            ks=(1, 5, 10, 50)
        )
        
        print("\n" + "="*60)
        
        # Return the full metrics dict for eval.py to process
        return matcha_metrics

    # --------------------------------------------------------
    # Retrieval metrics (fallback: within-loader, pairwise)
    # --------------------------------------------------------

    def compute_retrieval_metrics(self, loader):
        """
        img1 = query, img2 = true match within the same loader.
        """
        self.eval()

        emb_a, emb_b = [], []

        with torch.no_grad():
            for img1, img2, *rest in tqdm(loader, desc="Extracting embeddings"):
                z1 = F.normalize(self(img1), dim=1)
                z2 = F.normalize(self(img2), dim=1)
                emb_a.append(z1)
                emb_b.append(z2)

        if not emb_a or not emb_b:
            print("\nRetrieval Metrics: N/A (empty embeddings)")
            return

        A = torch.cat(emb_a, dim=0)
        B = torch.cat(emb_b, dim=0)

        sim = torch.matmul(A, B.T)  # [N, N]
        N = sim.size(0)

        mAP = 0.0
        recall_at_k = {k: 0 for k in [1, 3, 5, 10]}

        for i in range(N):
            ranking = torch.argsort(sim[i], descending=True)
            rank = (ranking == i).nonzero(as_tuple=True)

            if len(rank[0]) == 0:
                continue

            rank = rank[0].item() + 1
            mAP += 1.0 / rank

            for k in recall_at_k:
                if rank <= k:
                    recall_at_k[k] += 1

        mAP /= max(N, 1)
        for k in recall_at_k:
            recall_at_k[k] /= max(N, 1)

        print("\nRetrieval Metrics (within-loader):")
        print(f"mAP: {mAP:.4f}")
        for k in [1, 3, 5, 10]:
            print(f"Recall@{k}: {recall_at_k[k]:.4f}")
