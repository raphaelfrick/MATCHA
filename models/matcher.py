"""Unified matcher model: encoder + projector + loss."""

from typing import Any, Dict, Optional, Sequence, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.base_model import BaseModel
from encoders.registry import build_encoder
from utils.metrics import compute_open_set_metrics


def build_projector(in_dim: int, out_dim: int, config: Dict[str, Any]) -> nn.Module:
    proj = str(config.get("projector", "mlp")).lower().strip()
    if proj in {"linear"}:
        return nn.Linear(in_dim, out_dim)
    if proj in {"mlp_bn", "mlpbn"}:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
    # default mlp
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True),
        nn.Linear(out_dim, out_dim),
    )


class MatcherModel(BaseModel):
    """Unified model for encoder + projector + loss."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        self.encoder, enc_dim = build_encoder(config, device)
        emb_dim = int(config.get("embedding_dim", enc_dim))
        self.projector = build_projector(enc_dim, emb_dim, config)
        self.temperature = float(config.get("temperature", 0.07))

        freeze_encoder = bool(config.get("freeze_encoder", True))
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.to(device)

    def forward(self, images):
        feats = self.encoder(images)
        return self.projector(feats)

    def preprocess(self, images):
        return self.encoder.preprocess(images)

    def _infonce_pair_metrics(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_pos = F.cosine_similarity(z1, z2, dim=1)
        sim_neg = F.cosine_similarity(z1, torch.roll(z2, shifts=1, dims=0), dim=1)
        pair_count = min(len(sim_pos), len(sim_neg))
        hit_correct = (sim_pos[:pair_count] > sim_neg[:pair_count]).sum().item()
        if self.loss_fn is None:
            raise ValueError("MatcherModel requires a loss function.")
        loss = self.loss_fn(z1, z2)
        return loss, hit_correct, pair_count

    def _filter_connected(
        self,
        anchors: List[Any],
        positives: List[Any],
        negatives: Optional[List[Any]] = None,
        require_negative: bool = False,
    ) -> Tuple[List[Any], List[Any], Optional[List[Any]]]:
        filtered = []
        if negatives is None:
            for a, p in zip(anchors, positives):
                if p is not None:
                    filtered.append((a, p))
            if not filtered:
                return [], [], None
            a_out, p_out = zip(*filtered)
            return list(a_out), list(p_out), None

        for a, p, n in zip(anchors, positives, negatives):
            if p is None:
                continue
            if require_negative and n is None:
                continue
            filtered.append((a, p, n))

        if not filtered:
            return [], [], []
        a_out, p_out, n_out = zip(*filtered)
        return list(a_out), list(p_out), list(n_out)

    def train_step(self, batch):
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Expected batch with at least (anchor, positive).")
        anchor, positive = batch[0], batch[1]
        negative = batch[2] if len(batch) > 2 else None

        if positive is None or (isinstance(anchor, list) and len(anchor) == 0) or (isinstance(anchor, torch.Tensor) and anchor.numel() == 0):
            return None

        loss_name = str(self.config.get("loss", "infonce")).lower().strip()
        if loss_name == "triplet":
            if negative is None or (isinstance(negative, list) and len(negative) == 0):
                return None
            if self.loss_fn is None:
                raise ValueError("MatcherModel requires a loss function.")
            if isinstance(anchor, list) and isinstance(positive, list) and isinstance(negative, list):
                anchor, positive, negative = self._filter_connected(anchor, positive, negative, require_negative=True)
                if not anchor:
                    return None
            emb_a = self(anchor)
            emb_p = self(positive)
            emb_n = self(negative)
            return self.loss_fn(emb_a, emb_p, emb_n)

        # InfoNCE
        if isinstance(anchor, list) and isinstance(positive, list):
            anchor, positive, _ = self._filter_connected(anchor, positive, None, require_negative=False)
            if not anchor:
                return None
        z1 = self(anchor)
        z2 = self(positive)
        if self.loss_fn is None:
            raise ValueError("MatcherModel requires a loss function.")
        return self.loss_fn(z1, z2)

    def val_step(self, batch):
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Expected batch with at least (anchor, positive).")
        anchor, positive = batch[0], batch[1]
        negative = batch[2] if len(batch) > 2 else None

        if positive is None or (isinstance(anchor, list) and len(anchor) == 0) or (isinstance(anchor, torch.Tensor) and anchor.numel() == 0):
            return None

        loss_name = str(self.config.get("loss", "infonce")).lower().strip()
        if loss_name == "triplet":
            if negative is None or (isinstance(negative, list) and len(negative) == 0):
                return None
            if self.loss_fn is None:
                raise ValueError("MatcherModel requires a loss function.")
            if isinstance(anchor, list) and isinstance(positive, list) and isinstance(negative, list):
                anchor, positive, negative = self._filter_connected(anchor, positive, negative, require_negative=True)
                if not anchor:
                    return None
            emb_a = self(anchor)
            emb_p = self(positive)
            emb_n = self(negative)
            loss = self.loss_fn(emb_a, emb_p, emb_n)
            sim_pos = F.cosine_similarity(emb_a, emb_p)
            sim_neg = F.cosine_similarity(emb_a, emb_n)
            pair_count = min(len(sim_pos), len(sim_neg))
            hit_correct = (sim_pos[:pair_count] > sim_neg[:pair_count]).sum().item()
            return loss, hit_correct, pair_count

        # InfoNCE
        if isinstance(anchor, list) and isinstance(positive, list):
            anchor, positive, _ = self._filter_connected(anchor, positive, None, require_negative=False)
            if not anchor:
                return None
        z1 = self(anchor)
        z2 = self(positive)
        return self._infonce_pair_metrics(z1, z2)

    @torch.no_grad()
    def _embed_loader(self, loader, desc: Optional[str] = None):
        embs, paths = [], []
        iterable = tqdm(loader, desc=desc) if desc else loader
        for batch in iterable:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                imgs, pths = batch[0], batch[1]
            else:
                imgs, pths = batch, None
            z = F.normalize(self(imgs), dim=1)
            embs.append(z)
            if pths is not None:
                paths.extend(pths)
        if embs:
            embs = torch.cat(embs, dim=0)
        else:
            emb_dim = int(self.config.get("embedding_dim", 256))
            embs = torch.empty(0, emb_dim, device=self.device)
        return embs, paths

    @torch.no_grad()
    def retrieval_against_gallery(self, query_loader, gallery_loader, ground_truth_map=None, orphan_gt_map=None):
        G, g_paths = self._embed_loader(gallery_loader, desc="Encoding gallery")
        if G.numel() == 0:
            return {"Hit@k": {}, "MRR": None}
        path_to_idx = {p: i for i, p in enumerate(g_paths)} if g_paths else {}

        connected_ranks = []
        connected_max_scores = []
        orphan_max_scores = []

        for batch in tqdm(query_loader, desc="Evaluating queries vs gallery"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 5:
                img1, _, _, m_paths, a_paths = batch[:5]
            else:
                img1 = batch[0] if isinstance(batch, (list, tuple)) else batch
                m_paths, a_paths = None, None

            if isinstance(img1, list) and len(img1) == 0:
                continue

            Zq = F.normalize(self(img1), dim=1)
            sims = torch.matmul(Zq, G.T)
            max_scores = sims.max(dim=1)[0]
            top1_idx = torch.argmax(sims, dim=1)

            for i in range(Zq.size(0)):
                gt_path = None
                if ground_truth_map and m_paths is not None:
                    gt_path = ground_truth_map.get(m_paths[i], None)
                elif a_paths is not None:
                    gt_path = a_paths[i]

                is_orphan = False
                if orphan_gt_map and m_paths is not None:
                    is_orphan = m_paths[i] in orphan_gt_map
                elif gt_path is None:
                    is_orphan = True

                if is_orphan:
                    orphan_max_scores.append(float(max_scores[i].item()))
                else:
                    connected_max_scores.append(float(max_scores[i].item()))
                    if gt_path is not None and gt_path in path_to_idx:
                        tgt = path_to_idx[gt_path]
                        ranking = torch.argsort(sims[i], descending=True)
                        pos = (ranking == tgt).nonzero(as_tuple=True)[0]
                        rank = pos.item() + 1 if len(pos) > 0 else None
                        if rank is not None:
                            connected_ranks.append(rank)

        matcha_metrics = compute_open_set_metrics(
            ranks_conn=connected_ranks,
            max_scores_conn=connected_max_scores,
            max_scores_orph=orphan_max_scores,
            thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
            ks=(1, 5, 10, 50),
        )
        return matcha_metrics
