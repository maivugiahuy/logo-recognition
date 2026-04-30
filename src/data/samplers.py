"""
Batch samplers:
  MPerClassSampler       — k classes × m images (standard ProxyNCA++ batch)
  HardNegativeBatchSampler — k classes × m images + 1 HN class per positive class
                             Sec 3.2.3: batch size ∈ [km, 2km]
"""
import json
import random
from collections import defaultdict

from torch.utils.data import Sampler


class MPerClassSampler(Sampler):
    """Yields batches of k classes × m images each."""

    def __init__(self, labels: list[int], k: int, m: int, seed: int = 0):
        self.labels = labels
        self.k = k
        self.m = m
        self.rng = random.Random(seed)

        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.class_to_indices[lbl].append(idx)
        self.classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= m]

    def __iter__(self):
        classes = self.classes.copy()
        self.rng.shuffle(classes)
        batch: list[int] = []
        for cls in classes:
            indices = self.class_to_indices[cls].copy()
            self.rng.shuffle(indices)
            chosen = (indices * ((self.m // len(indices)) + 1))[:self.m]
            batch.extend(chosen)
            if len(batch) >= self.k * self.m:
                yield batch
                batch = []

    def __len__(self) -> int:
        return len(self.classes) // self.k


class HardNegativeBatchSampler(Sampler):
    """
    For each of k positive classes, optionally appends 1 HN class (m images).
    hn_map: {class_idx: [hn_class_idx, ...]} built from mine_hn.py output.
    batch size ∈ [km, 2km].
    """

    def __init__(
        self,
        labels: list[int],
        hn_map: dict[int, list[int]],
        k: int,
        m: int,
        seed: int = 0,
    ):
        self.labels = labels
        self.hn_map = hn_map
        self.k = k
        self.m = m
        self.rng = random.Random(seed)

        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.class_to_indices[lbl].append(idx)
        self.classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= m]

    def _sample_class(self, cls: int) -> list[int]:
        indices = self.class_to_indices[cls].copy()
        self.rng.shuffle(indices)
        return (indices * ((self.m // len(indices)) + 1))[:self.m]

    def __iter__(self):
        classes = self.classes.copy()
        self.rng.shuffle(classes)
        batch: list[int] = []
        hn_in_batch: set[int] = set()

        for i in range(0, len(classes) - self.k + 1, self.k):
            pos_classes = classes[i:i + self.k]
            batch = []
            hn_in_batch = set(pos_classes)

            for cls in pos_classes:
                batch.extend(self._sample_class(cls))

            # append one HN class per positive class if available
            for cls in pos_classes:
                candidates = [c for c in self.hn_map.get(cls, []) if c not in hn_in_batch]
                if candidates:
                    hn_cls = self.rng.choice(candidates)
                    hn_in_batch.add(hn_cls)
                    batch.extend(self._sample_class(hn_cls))

            yield batch

    def __len__(self) -> int:
        return len(self.classes) // self.k


def load_hn_map(hn_json: str, class_to_idx: dict[str, int]) -> dict[int, list[int]]:
    """Load hn_map.json (class_name → [hn_class_names]) and convert to idx→[idx]."""
    with open(hn_json) as f:
        raw: dict[str, list[str]] = json.load(f)
    result: dict[int, list[int]] = {}
    for cls, hn_list in raw.items():
        if cls not in class_to_idx:
            continue
        idx = class_to_idx[cls]
        result[idx] = [class_to_idx[c] for c in hn_list if c in class_to_idx]
    return result
