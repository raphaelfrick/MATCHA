# models/triplet_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from models.base_model import BaseModel
from utils.metrics import compute_open_set_metrics


class TripletNet(BaseModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.device = device
        self.config = config

        # Backbone + projector
        self.backbone = models.resnet50(weights="DEFAULT")
        self.backbone.fc = nn.Identity()
        emb_dim = int(self.config.get('embedding_dim', 512))
        self.projector = nn.Sequential(
            nn.Linear(2048, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.to(self.device)

    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features)

    def compute_triplet_loss(self, emb_a, emb_p, emb_n):
        d_pos = F.pairwise_distance(emb_a, emb_p)
        d_neg = F.pairwise_distance(emb_a, emb_n)
        margin = float(self.config.get('margin', 0.2))
        return torch.mean(torch.clamp(d_pos - d_neg + margin, min=0.0))

    def train_step(self, batch):
        anchor, positive, negative = batch[:3]
        anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
        emb_a = self(anchor)
        emb_p = self(positive)
        emb_n = self(negative)
        return self.compute_triplet_loss(emb_a, emb_p, emb_n)

    def val_step(self, batch):
        anchor, positive, negative = batch[:3]
        anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
        emb_a = self(anchor)
        emb_p = self(positive)
        emb_n = self(negative)
        sim_pos = F.cosine_similarity(emb_a, emb_p)
        sim_neg = F.cosine_similarity(emb_a, emb_n)
        thr = float(self.config.get('threshold', 0.5))
        labels = [1] * len(sim_pos) + [0] * len(sim_neg)
        preds = [(float(s.item()) > thr) for s in sim_pos] + [(float(s.item()) > thr) for s in sim_neg]
        return labels, preds

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
                        
                        # Debug: print info about first few connected queries
                        if i == 0 and len(connected_max_scores) <= 3:
                            print(f"Debug: m_paths[i]={m_paths[i] if m_paths else 'None'}")
                            print(f"Debug: gt_path={gt_path}")
                            print(f"Debug: gt_path in path_to_idx: {gt_path in path_to_idx if gt_path else 'N/A'}")
                            print(f"Debug: Sample gallery paths: {list(path_to_idx.keys())[:3]}")
                        
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
