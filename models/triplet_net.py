# models/triplet_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import f1_score
from tqdm import tqdm

from models.base_model import BaseModel
from utils.metrics import compute_open_set_metrics


class TripletNet(BaseModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.device = device
        self.config = config

        # Backbone + projector
        backbone_name = str(self.config.get("backbone", "resnet50")).lower().strip()
        self.backbone, feat_dim = self._build_backbone(backbone_name)
        emb_dim = int(self.config.get("embedding_dim", feat_dim))
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.to(self.device)

    def _build_backbone(self, backbone_name: str):
        pretrained = bool(self.config.get("pretrained", True))

        if backbone_name.startswith("resnet"):
            if not hasattr(models, backbone_name):
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            weights = "DEFAULT" if pretrained else None
            model = getattr(models, backbone_name)(weights=weights)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, int(feat_dim)

        if backbone_name.startswith("vit"):
            if not hasattr(models, backbone_name):
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            weights = "DEFAULT" if pretrained else None
            model = getattr(models, backbone_name)(weights=weights)

            feat_dim = getattr(model, "hidden_dim", None)
            if feat_dim is None and hasattr(model, "heads") and hasattr(model.heads, "head"):
                feat_dim = model.heads.head.in_features
            if feat_dim is None:
                raise RuntimeError("Unable to infer ViT feature dimension.")

            if hasattr(model, "heads"):
                model.heads = nn.Identity()
            return model, int(feat_dim)

        raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features)

    def compute_triplet_loss(self, emb_a, emb_p, emb_n):
        d_pos = F.pairwise_distance(emb_a, emb_p)
        d_neg = F.pairwise_distance(emb_a, emb_n)
        margin = float(self.config.get('margin', 0.2))
        return torch.mean(torch.clamp(d_pos - d_neg + margin, min=0.0))

    # --------------------------------------------------------
    # Similarity computation
    # --------------------------------------------------------

    def match_score(self, emb1, emb2):
        """
        Compute cosine similarity between two embedding tensors.
        
        Args:
            emb1: First embedding tensor [N, D]
            emb2: Second embedding tensor [N, D]
            
        Returns:
            Cosine similarity scores [N]
        """
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        return torch.sum(emb1 * emb2, dim=1)

    # --------------------------------------------------------
    # Training / validation
    # --------------------------------------------------------

    def train_step(self, batch):
        anchor, positive, negative = batch[:3]
        if positive is None or negative is None:
            return None
        if isinstance(anchor, torch.Tensor) and anchor.numel() == 0:
            return None
        if self.loss_fn is None:
            raise ValueError("TripletNet requires a loss function (triplet).")
        anchor = anchor.to(self.device, non_blocking=True)
        positive = positive.to(self.device, non_blocking=True)
        negative = negative.to(self.device, non_blocking=True)
        emb_a = self(anchor)
        emb_p = self(positive)
        emb_n = self(negative)
        return self.loss_fn(emb_a, emb_p, emb_n)

    def val_step(self, batch):
        anchor, positive, negative = batch[:3]
        if positive is None or negative is None:
            return None
        if isinstance(anchor, torch.Tensor) and anchor.numel() == 0:
            return None
        if self.loss_fn is None:
            raise ValueError("TripletNet requires a loss function (triplet).")
        anchor = anchor.to(self.device, non_blocking=True)
        positive = positive.to(self.device, non_blocking=True)
        negative = negative.to(self.device, non_blocking=True)
        emb_a = self(anchor)
        emb_p = self(positive)
        emb_n = self(negative)
        loss = self.loss_fn(emb_a, emb_p, emb_n)
        sim_pos = F.cosine_similarity(emb_a, emb_p)
        sim_neg = F.cosine_similarity(emb_a, emb_n)
        pair_count = min(len(sim_pos), len(sim_neg))
        hit_correct = (sim_pos[:pair_count] > sim_neg[:pair_count]).sum().item()
        return loss, hit_correct, pair_count

    def validate(self, val_loader, split="val", auth_loader=None, ground_truth_map=None, orphan_gt_map=None):
        """
        Evaluate the model on validation/test set using MATCH-A metrics.
        
        Args:
            val_loader: Validation/test data loader.
            split: 'val' or 'test'.
            auth_loader: Gallery loader for retrieval evaluation.
            ground_truth_map: Mapping for connected queries (manipulated -> authentic).
            orphan_gt_map: Set of orphan query paths (no authentic match).
        """
        self.eval()
        y_true, y_pred, y_scores = [], [], []

        if split != "test":
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating ({split})"):
                    labels, preds = self.val_step(batch)
                    y_true.extend([int(l) for l in labels])
                    y_pred.extend([bool(p) for p in preds])
            return f1_score(y_true, y_pred, zero_division=0)

        # === TEST SPLIT: Full classification + retrieval evaluation ===
        if auth_loader is None:
            raise ValueError("TripletNet.validate requires auth_loader for test split.")

        # Encode authentic DB
        auth_embeddings, auth_paths = [], []
        with torch.no_grad():
            for imgs, paths in tqdm(auth_loader, desc="Encoding authentic images"):
                imgs = imgs.to(self.device)
                emb = self(imgs)
                auth_embeddings.append(emb)
                auth_paths.extend(paths)
            auth_embeddings = torch.cat(auth_embeddings, dim=0)
        path_to_idx = {p: i for i, p in enumerate(auth_paths)}
        G = F.normalize(auth_embeddings, dim=1)

        # Track connected vs orphan queries separately for MATCH-A metrics
        connected_ranks = []  # Ranks for connected queries
        connected_max_scores = []  # Max similarity scores for connected queries
        orphan_max_scores = []  # Max similarity scores for orphan queries
        
        thr = float(self.config.get('threshold', 0.5))

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating test queries"):
                # batch may be (anchor, positive, negative, m_paths, a_paths) on test
                if len(batch) >= 5:
                    anchor, _, _, m_paths, a_paths = batch
                else:
                    anchor, _, _ = batch
                    m_paths, a_paths = None, None

                anchor = anchor.to(self.device)
                B = anchor.size(0)

                # Embed the entire batch once
                emb_q = self(anchor)
                Zq = F.normalize(emb_q, dim=1)
                sims = torch.matmul(Zq, G.T)  # [B, N]
                
                # Get max similarity scores for each query
                max_scores = sims.max(dim=1)[0]  # [B]

                # Top-1 predictions
                top1_idx = torch.argmax(sims, dim=1)  # [B]
                top1_scores = sims.gather(1, top1_idx.view(-1, 1)).squeeze(1).tolist()

                for i in range(B):
                    pred_path = auth_paths[int(top1_idx[i].item())]
                    top1_score = float(top1_scores[i])

                    # Ground-truth authentic path for this query
                    gt_path = None
                    if ground_truth_map and m_paths is not None:
                        gt_path = ground_truth_map.get(m_paths[i], None)
                    elif a_paths is not None:
                        gt_path = a_paths[i]

                    # Check if this is an orphan query (no ground truth)
                    # Orphan queries have a_paths[i] = None
                    is_orphan = (a_paths is not None and a_paths[i] is None) or \
                                (gt_path is None and orphan_gt_map and m_paths is not None and m_paths[i] in orphan_gt_map)

                    if is_orphan:
                        # Orphan query - track max score for abstention metrics
                        orphan_max_scores.append(float(max_scores[i].item()))
                    else:
                        # Connected query - track rank for retrieval metrics
                        connected_max_scores.append(float(max_scores[i].item()))
                        
                        # Retrieval metrics
                        if gt_path is not None:
                            if gt_path in path_to_idx:
                                target_idx = path_to_idx[gt_path]
                                ranking = torch.argsort(sims[i], descending=True)
                                pos = (ranking == target_idx).nonzero(as_tuple=True)[0]
                                rank = pos.item() + 1 if len(pos) > 0 else None
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
