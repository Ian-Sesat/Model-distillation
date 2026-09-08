"""
train_mci0_minilm.py
===============================================================================
Train the new MCi0 + MiniLM model by distilling from MCIP-SigLIP2.

Reuses:
  - the dual-teacher loss (MCIP-SigLIP2 as BOTH teachers: cross-modal T2I + image-image I2I)
  - the MCIP-SigLIP2 teacher caches you already extracted (distill_teacher_mcip_siglip/)
  - the mmap teacher-loading pattern (avoids the 31GB RAM overflow)

Everything trainable: MCi0 + MiniLM + both projectors (Option B).
More epochs than fine-tuning (default 20) since projectors are random and the two
encoders were never trained together. LR warmup + cosine decay.
"""
import os, json, math, argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from mci0_minilm_model import MCi0MiniLM
from distill_loss_dual import dual_teacher_loss   # your existing dual-teacher loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_img",   required=True, help="MCIP-SigLIP2 image feats .npy")
    ap.add_argument("--t_txt",   required=True, help="MCIP-SigLIP2 text feats .npy")
    ap.add_argument("--t_valid", required=True, help="MCIP-SigLIP2 valid mask .npy")
    ap.add_argument("--pairs_dir", required=True, help="CC12M dir (captions.tsv + images)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shared_dim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="higher than fine-tuning: random projectors need to move. "
                         "warmup+cosine. encoders move more gently via the shared lr.")
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--tau_teacher", type=float, default=0.07)
    ap.add_argument("--tau_student", type=float, default=0.07)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--w_t2i", type=float, default=1.0)
    ap.add_argument("--w_i2i", type=float, default=1.0)
    ap.add_argument("--logit_scale", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=50)
    return ap.parse_args()


def load_pairs(pairs_dir):
    """captions.tsv -> list of (abs_image_path, caption), aligned to teacher-cache order.
    IMPORTANT: this must match the SAME order the teacher extraction used, so row i in
    the teacher cache corresponds to pair i here. The teacher was extracted from this
    same captions.tsv, so the natural row order aligns."""
    tsv = os.path.join(pairs_dir, "captions.tsv")
    img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    cands = [pairs_dir, os.path.join(pairs_dir, "images")]
    def resolve(ref):
        for b in cands:
            p = os.path.join(b, ref)
            if os.path.exists(p): return p
        return ref if (os.path.isabs(ref) and os.path.exists(ref)) else None
    pairs, col = [], None
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            if col is None:
                c0 = parts[0].lower().endswith(img_exts)
                c1 = parts[1].lower().endswith(img_exts)
                if c0 and not c1:
                    col = 0
                elif c1 and not c0:
                    col = 1
                else:
                    # header row (neither col is an image path) -> skip it, matching
                    # the teacher extraction, so index i aligns with teacher row i
                    if parts[0].lower() in ("image", "filename", "img", "path"):
                        continue
                    col = 0
            img_ref, cap = parts[col], parts[1-col]
            pairs.append((resolve(img_ref), cap))
    return pairs


class PairDataset(Dataset):
    """Yields (image_tensor, caption, original_index). Index maps into the teacher cache.
    Only pairs valid in the teacher AND with a resolvable image are used."""
    def __init__(self, pairs, valid, transform):
        self.transform = transform
        self.items = [(i, p, c) for i, (p, c) in enumerate(pairs)
                      if valid[i] and p is not None]
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        idx, path, cap = self.items[k]
        try:
            img = self.transform(Image.open(path).convert("RGB"))
            return img, cap, idx
        except Exception:
            return None

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    imgs = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, caps, idxs


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    # ---- model (encoders + projectors, all trainable) ----
    model = MCi0MiniLM(shared_dim=args.shared_dim, device=device).to(device).train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"new model: MCi0 + MiniLM + projectors | trainable params: {n_train:,}")

    # ---- teacher caches (MCIP-SigLIP2) via mmap (avoid RAM overflow) ----
    t_img   = np.load(args.t_img,   mmap_mode="r")   # (N, 768) MCIP-SigLIP2 image
    t_txt   = np.load(args.t_txt,   mmap_mode="r")   # (N, 768) MCIP-SigLIP2 text
    t_valid = np.load(args.t_valid).astype(bool)
    print(f"teacher (MCIP-SigLIP2): img {t_img.shape} txt {t_txt.shape} | valid {t_valid.sum():,}")

    # ---- data ----
    pairs = load_pairs(args.pairs_dir)
    print(f"pairs from captions.tsv: {len(pairs):,}")
    ds = PairDataset(pairs, t_valid, model.image_transform(256))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate, drop_last=True)
    print(f"trainable pairs: {len(ds):,} | batch {args.batch} | {len(loader):,} steps/epoch")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(loader)
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    step = 0
    total_steps = args.epochs * len(loader)
    pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)
    for epoch in range(args.epochs):
        for batch in loader:
            if batch is None:
                continue
            imgs, caps, idxs = batch
            imgs = imgs.to(device, non_blocking=True)

            cur_lr = lr_at(step)
            for pg in opt.param_groups: pg["lr"] = cur_lr

            s_img = model.encode_image(imgs)          # (B, 512) L2-normed
            s_txt = model.encode_text(caps)           # (B, 512) L2-normed

            # gather MCIP-SigLIP2 teacher rows for this batch (mmap -> copy just these)
            with torch.no_grad():
                idxs_np = idxs.numpy()
                teach_img = F.normalize(torch.from_numpy(t_img[idxs_np].copy()).float()
                                        .to(device, non_blocking=True), dim=-1)
                teach_txt = F.normalize(torch.from_numpy(t_txt[idxs_np].copy()).float()
                                        .to(device, non_blocking=True), dim=-1)

            # dual-teacher loss: MCIP-SigLIP2 as BOTH teachers
            #   cross-modal (T2I) from teacher img+txt; image-image (I2I) from teacher img
            loss, parts = dual_teacher_loss(
                s_img, s_txt,
                teach_img, teach_txt,     # T2I cross-modal reference (MCIP-SigLIP2)
                teach_img,                # I2I image-image reference (MCIP-SigLIP2)
                tau_teacher=args.tau_teacher, tau_student=args.tau_student,
                lam=args.lam, w_t2i=args.w_t2i, w_i2i=args.w_i2i,
                logit_scale=args.logit_scale)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.update(1)
            pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.3f}")

            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} "
                      f"it={parts['L_it']:.4f} ti={parts['L_ti']:.4f} ii={parts['L_ii']:.4f} "
                      f"clip={parts['L_clip']:.4f} lr={cur_lr:.2e}", flush=True)
            step += 1

        # rolling latest (crash safety) + a milestone checkpoint every 5 epochs + final
        ep = epoch + 1
        torch.save({"model": model.state_dict(), "epoch": ep},
                   os.path.join(args.out_dir, "mci0_minilm_latest.pth"))
        if ep % 5 == 0:
            torch.save({"model": model.state_dict(), "epoch": ep},
                       os.path.join(args.out_dir, f"mci0_minilm_epoch{ep}.pth"))
            print(f"  [epoch {ep}] saved milestone checkpoint", flush=True)
        if ep == args.epochs:
            torch.save({"model": model.state_dict(), "epoch": ep},
                       os.path.join(args.out_dir, "mci0_minilm_final.pth"))
            print(f"  [epoch {ep}] saved final", flush=True)

    print("DONE.")


if __name__ == "__main__":
    main()