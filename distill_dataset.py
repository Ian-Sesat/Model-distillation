import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from realign_extract_features import load_pairs


class DistillDataset(Dataset):
    def __init__(self, pairs_dir, preprocess, valid_mask=None):
        """
        Args:
            pairs_dir  : CC12M folder (captions.tsv + images) — SAME dir used
                         to build the teacher cache, so pair order matches.
            preprocess : the STUDENT's preprocess transform.
            valid_mask : optional bool array from the teacher extraction; if
                         given, only pairs marked valid are used (skips any the
                         teacher couldn't embed).
        """
        self.pairs = load_pairs(pairs_dir)          # [(img_path, caption), ...]
        self.preprocess = preprocess

        if valid_mask is not None:
            if len(valid_mask) != len(self.pairs):
                raise ValueError(
                    f"valid_mask length {len(valid_mask)} != pairs "
                    f"{len(self.pairs)} — cache and pairs_dir are out of sync")
            self.index_map = np.where(valid_mask)[0].tolist()
        else:
            self.index_map = list(range(len(self.pairs)))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        orig_idx = self.index_map[i]           # ORIGINAL pair index (teacher cache)
        path, caption = self.pairs[orig_idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.preprocess(img)
        except Exception:

            return None
        return img, caption, orig_idx


def collate_drop_none(batch):
    """Drop failed items; return (images, captions, indices) or None if empty."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, caps, idxs


if __name__ == "__main__":
    n = 10
    valid = np.ones(n, dtype=bool)
    valid[[2, 5, 7]] = False

    # mimic index_map construction
    index_map = np.where(valid)[0].tolist()
    print("valid pairs:", index_map)
    print("expected   : [0, 1, 3, 4, 6, 8, 9]")
    assert index_map == [0, 1, 3, 4, 6, 8, 9], "index_map wrong!"

    # dataset position 0 -> orig 0, pos 2 -> orig 3 (skips invalid 2), etc.
    checks = {0: 0, 1: 1, 2: 3, 3: 4, 4: 6, 5: 8, 6: 9}
    for pos, expected_orig in checks.items():
        got = index_map[pos]
        status = "OK" if got == expected_orig else "WRONG"
        print(f"  dataset pos {pos} -> orig index {got} (expect {expected_orig}) {status}")
        assert got == expected_orig

    print("\nindex alignment verified: dataset returns ORIGINAL pair indices,")
    print("so teacher_cache[idx] always matches the student's live pair.")