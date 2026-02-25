"""Unified dataset for query/positive(/negative) pairs."""

import random
from typing import Any, List, Optional, Tuple
from torch.utils.data import Dataset

from data.base_dataset import BaseImageDataset, _open_rgb


class QueryPairsDataset(BaseImageDataset):
    """
    Dataset that yields (anchor, positive, negative, anchor_path, positive_path).

    For orphans, positive/negative are None.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        transform: Optional[Any] = None,
        negative_sampling: str = "random_authentic",
        return_paths: Optional[bool] = None,
    ) -> None:
        super().__init__(csv_path, split, transform)

        self.negative_sampling = str(negative_sampling).lower().strip()
        self.return_paths = bool(return_paths) if return_paths is not None else (split in {"test", "val", "train"})

        self.authentic_pool = self.df["authentic_filepath"].dropna().unique().tolist()
        self.pairs: List[Tuple[str, Optional[str]]] = []
        self.orphan_indices: List[int] = []

        for _, row in self.df.iterrows():
            manip = row.get("manipulated")
            auth = row.get("authentic_filepath")
            if isinstance(manip, str) and manip:
                if isinstance(auth, str) and auth:
                    self.pairs.append((manip, auth))
                else:
                    self.orphan_indices.append(len(self.pairs))
                    self.pairs.append((manip, None))

    def __len__(self) -> int:
        return len(self.pairs)

    def _sample_negative(self, positive_path: str) -> Optional[str]:
        if self.negative_sampling in {"none", "off", "false"}:
            return None
        candidates = [p for p in self.authentic_pool if p != positive_path]
        if not candidates:
            return None
        return random.choice(candidates)

    def __getitem__(self, idx: int) -> Tuple:
        manip_path, auth_path = self.pairs[idx]
        anchor = _open_rgb(manip_path)

        if auth_path is None:
            if self.transform:
                anchor = self.transform(anchor)
            if self.return_paths:
                return anchor, None, None, manip_path, None
            return anchor, None, None

        positive = _open_rgb(auth_path)
        negative_path = self._sample_negative(auth_path)
        negative = _open_rgb(negative_path) if negative_path else None

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            if negative is not None:
                negative = self.transform(negative)

        if self.return_paths:
            return anchor, positive, negative, manip_path, auth_path
        return anchor, positive, negative
