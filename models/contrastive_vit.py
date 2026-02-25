# models/contrastive_vit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from models.base_model import BaseModel
from utils.metrics import compute_open_set_metrics


class ContrastiveViT(BaseModel):
    """
    Contrastive encoder + projector using DINOv2 via torch.hub.

    - Loads backbone: facebookresearch/dinov2 (e.g., dinov2_vitb14).
    - Backbone is frozen; projector is trained.
    - Validation computes: F1 on 'val', classification + retrieval on 'test'.

      Classification labels are derived from whether the predicted top-1 path matches the ground-truth authentic path.
    """
    def __init__(self, config, device):
        super().__init__(config, device)
        self.device = device
        self.config = config

        # Backbone
        self.backbone, feat_dim = self._build_backbone(self.config.get("vit_name", "dinov2_vitb14"))
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval().to(self.device)

        emb_dim = int(self.config.get("embedding_dim", feat_dim))

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
        self.to(self.device)

    def _build_backbone(self, vit_name: str):
        name = str(vit_name).lower().strip()
        if name in ("dinov2_vitb14", "dino_v2_b14", "dinov2-b-14", "dinov2_b14"):
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif name in ("dinov2_vitl14", "dino_v2_l14", "dinov2-l-14", "dinov2_l14"):
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        elif name in ("dinov2_vits14", "dino_v2_s14", "dinov2-s-14", "dinov2_s14"):
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif name in ("dinov2_vitg14", "dino_v2_g14", "dinov2-g-14", "dinov2_g14"):
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        else:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        feat_dim = (
            getattr(model, "embed_dim", None)
            or getattr(model, "num_features", None)
            or getattr(model, "hidden_dim", None)
            or getattr(model, "width", None)
            or 768
        )
        return model, int(feat_dim)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features (CLS/pooled representation).
        """
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
            if isinstance(out, dict):
                for key in ("x_norm_clstoken", "cls_token", "pooled", "feat"):
                    if key in out and isinstance(out[key], torch.Tensor):
                        return out[key].float()
                for v in out.values():
                    if isinstance(v, torch.Tensor):
                        return v.float()
            elif isinstance(out, torch.Tensor):
                return out.float()
        y = self.backbone(x)
        if isinstance(y, torch.Tensor):
            return y.float()
        raise RuntimeError("Unable to extract features from DINOv2 backbone output.")

    def forward(self, x):
        feats = self._extract_features(x)  # [B, feat_dim]
        emb = self.projector(feats)        # [B, emb_dim]
        return emb

    def match_score(self, A, B):
        """
        Cosine similarity matrix between two embedding sets:
        A: [N, D], B: [M, D] => returns [N, M]
        """
        A = F.normalize(A, dim=1)
        B = F.normalize(B, dim=1)
        return torch.matmul(A, B.t())

    def compute_contrastive_loss(self, z1, z2, temperature=None):
        temp = float(self.config.get("temperature", 0.1)) if temperature is None else float(temperature)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        n = z1.size(0)

        reps = torch.cat([z1, z2], dim=0)                     # [2N, D]
        logits = torch.matmul(reps, reps.t()) / temp          # [2N, 2N]

        mask = torch.eye(2 * n, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(mask, -1e9)

        targets = torch.arange(2 * n, device=logits.device)
        targets = (targets + n) % (2 * n)

        loss = F.cross_entropy(logits, targets)
        return loss

    def train_step(self, batch):
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError("ContrastiveViT.train_step expects (img1, img2, ...)")
        img1, img2 = batch[0], batch[1]
        if isinstance(img1, torch.Tensor) and img1.numel() == 0:
            return None
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)
        if self.loss_fn is None:
            raise ValueError("ContrastiveViT requires a loss function (infonce).")
        z1 = self(img1)
        z2 = self(img2)
        return self.loss_fn(z1, z2)

    def val_step(self, batch):
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError("ContrastiveViT.val_step expects (img1, img2, ...)")
        # Handle both 2-tuple (img1, img2) and 4-tuple (anchors, positives, paths1, paths2) formats
        img1 = batch[0]
        img2 = batch[1]
        if isinstance(img1, torch.Tensor) and img1.numel() == 0:
            return None
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)

        with torch.no_grad():
            z1 = self(img1)
            z2 = self(img2)
            sim_pos = F.cosine_similarity(z1, z2, dim=1)
            sim_neg = F.cosine_similarity(z1, torch.roll(z2, shifts=1, dims=0), dim=1)

        if self.loss_fn is None:
            raise ValueError("ContrastiveViT requires a loss function (infonce).")
        loss = self.loss_fn(z1, z2)

        pair_count = min(len(sim_pos), len(sim_neg))
        hit_correct = (sim_pos[:pair_count] > sim_neg[:pair_count]).sum().item()
        return loss, hit_correct, pair_count

    def validate(self, val_loader, split="val", auth_loader=None, ground_truth_map=None, orphan_gt_map=None):
        """
        - 'val': F1 over batched positives + in-batch negatives.
        - 'test': classification + retrieval against authentic DB.

          Classification label = 1 iff predicted top-1 path equals the ground-truth authentic path.
          
        Args:
            val_loader: Validation/test data loader.
            split: 'val' or 'test'.
            auth_loader: Gallery loader for retrieval evaluation.
            ground_truth_map: Mapping for connected queries (manipulated -> authentic).
            orphan_gt_map: Set of orphan query paths (no authentic match).
        """
        self.eval()

        if split != "test":
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating ({split})"):
                    labels, preds = self.val_step(batch)
                    y_true.extend([int(l) for l in labels])
                    y_pred.extend([bool(p) for p in preds])
            return f1_score(y_true, y_pred, zero_division=0)

        # ===== TEST: Build authentic DB embeddings =====
        if auth_loader is None:
            raise ValueError("Provide auth_loader for test split to compare queries against all authentic samples.")

        auth_embeddings, auth_paths = [], []
        with torch.no_grad():
            for imgs, paths in tqdm(auth_loader, desc="Encoding authentic DB"):
                imgs = imgs.to(self.device, non_blocking=True)
                emb = self(imgs)
                auth_embeddings.append(emb)
                auth_paths.extend(paths)
        auth_embeddings = torch.cat(auth_embeddings, dim=0)
        path_to_idx = {p: i for i, p in enumerate(auth_paths)}

        # ===== TEST: Evaluate queries vs DB =====
        # Track connected vs orphan queries separately for MATCH-A metrics
        connected_ranks = []  # Ranks for connected queries
        connected_max_scores = []  # Max similarity scores for connected queries
        orphan_max_scores = []  # Max similarity scores for orphan queries
        
        ks = [1, 5, 10, 50]
        thr = float(self.config.get("threshold", 0.5))

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating test queries"):
                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    raise ValueError("Expected (img1, img2, [q_path, gt_path]) in test loader")

                q_imgs = batch[0].to(self.device, non_blocking=True)
                gt_paths = batch[3] if len(batch) >= 4 else None
                q_paths = batch[2] if len(batch) >= 4 else None

                q_embs = self(q_imgs)                                # [B, D]
                sims = self.match_score(q_embs, auth_embeddings)     # [B, N]
                
                # Get max similarity scores for each query
                max_scores = sims.max(dim=1)[0]  # [B]

                for i in range(q_embs.size(0)):
                    row = sims[i]
                    sorted_idx = torch.argsort(row, descending=True)
                    top1_idx = int(sorted_idx[0].item())
                    top1_score = float(row[top1_idx].item())
                    pred_path = auth_paths[top1_idx]

                    # Ground-truth authentic path for this query
                    gt_path = None
                    if ground_truth_map and q_paths is not None:
                        gt_path = ground_truth_map.get(q_paths[i], None)
                    elif gt_paths is not None:
                        gt_path = gt_paths[i]
                    
                    # Check if this is an orphan query (no ground truth)
                    is_orphan = False
                    if orphan_gt_map and q_paths is not None:
                        is_orphan = q_paths[i] in orphan_gt_map
                    elif gt_path is None:
                        is_orphan = True

                    if is_orphan:
                        # Orphan query - track max score for abstention metrics
                        orphan_max_scores.append(float(max_scores[i].item()))
                    else:
                        # Connected query - track rank for retrieval metrics
                        connected_max_scores.append(float(max_scores[i].item()))
                        
                        if gt_path is not None and gt_path in path_to_idx:
                            target_idx = path_to_idx[gt_path]
                            rank_pos = (sorted_idx == target_idx).nonzero(as_tuple=True)[0]
                            rank = rank_pos.item() + 1 if len(rank_pos) > 0 else None
                            if rank is not None:
                                connected_ranks.append(rank)

        # ===== Compute MATCH-A Metrics =====
        matcha_metrics = compute_open_set_metrics(
            ranks_conn=connected_ranks,
            max_scores_conn=connected_max_scores,
            max_scores_orph=orphan_max_scores,
            thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
            ks=(1, 5, 10, 50)
        )
        # Return the full metrics dict for eval.py to process
        return matcha_metrics
