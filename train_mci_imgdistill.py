import os, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from mci_imgdistill_model import MCiImgDistill, image_image_kl


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["mci0","mci2"], required=True)
    ap.add_argument("--t_img",   required=True, help="MCIP-SigLIP2 image feats .npy")
    ap.add_argument("--t_valid", required=True, help="MCIP-SigLIP2 valid mask .npy")
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shared_dim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=50)
    return ap.parse_args()


def load_pairs(pairs_dir):
    """captions.tsv -> list of image paths, in teacher-cache order (header skipped)."""
    tsv = os.path.join(pairs_dir, "captions.tsv")
    img_exts = (".jpg",".jpeg",".png",".webp",".bmp")
    cands = [pairs_dir, os.path.join(pairs_dir,"images")]
    def resolve(ref):
        for b in cands:
            p = os.path.join(b, ref)
            if os.path.exists(p): return p
        return ref if (os.path.isabs(ref) and os.path.exists(ref)) else None
    paths, col = [], None
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            if col is None:
                c0 = parts[0].lower().endswith(img_exts)
                c1 = parts[1].lower().endswith(img_exts)
                if c0 and not c1: col = 0
                elif c1 and not c0: col = 1
                else:
                    if parts[0].lower() in ("image","filename","img","path"):
                        continue   # skip header
                    col = 0
            paths.append(resolve(parts[col]))
    return paths


class ImgDataset(Dataset):
    def __init__(self, paths, valid, transform):
        self.transform = transform
        self.items = [(i, p) for i, p in enumerate(paths) if valid[i] and p is not None]
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        idx, path = self.items[k]
        try:
            return self.transform(Image.open(path).convert("RGB")), idx
        except Exception:
            return None

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    imgs = torch.stack([b[0] for b in batch])
    idxs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs, idxs


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    model = MCiImgDistill(which=args.which, shared_dim=args.shared_dim, device=device).to(device).train()
    print(f"{args.which} + projector | trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # teacher image embeddings (MCIP-SigLIP2) via mmap
    t_img   = np.load(args.t_img, mmap_mode="r")
    t_valid = np.load(args.t_valid).astype(bool)
    print(f"teacher img {t_img.shape} | valid {t_valid.sum():,}")

    paths = load_pairs(args.pairs_dir)
    print(f"pairs (header-skipped): {len(paths):,}  (should match teacher rows)")
    ds = ImgDataset(paths, t_valid, model.image_transform(256))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate, drop_last=True)
    print(f"trainable images: {len(ds):,} | batch {args.batch} | {len(loader):,} steps/epoch")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(loader)
    def lr_at(s):
        if s < args.warmup_steps: return args.lr * (s+1)/args.warmup_steps
        prog = (s-args.warmup_steps)/max(1, total_steps-args.warmup_steps)
        return args.lr * 0.5 * (1+math.cos(math.pi*prog))

    step = 0
    pbar = tqdm(total=total_steps, desc="imgdistill", dynamic_ncols=True)
    for epoch in range(args.epochs):
        for batch in loader:
            if batch is None: continue
            imgs, idxs = batch
            imgs = imgs.to(device, non_blocking=True)
            for pg in opt.param_groups: pg["lr"] = lr_at(step)

            s_img = model.encode_image(imgs)          # (B,512) projected, L2-normed
            with torch.no_grad():
                idxs_np = idxs.numpy()
                teach = F.normalize(torch.from_numpy(t_img[idxs_np].copy()).float()
                                    .to(device, non_blocking=True), dim=-1)
            loss = image_image_kl(s_img, teach, tau_t=args.tau, tau_s=args.tau)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.update(1); pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}")
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} "
                      f"lr={lr_at(step):.2e}", flush=True)
            step += 1

        ep = epoch + 1
        torch.save({"model": model.state_dict(), "epoch": ep},
                   os.path.join(args.out_dir, f"imgdistill_{args.which}_latest.pth"))
        if ep % 5 == 0 or ep == args.epochs:
            torch.save({"model": model.state_dict(), "epoch": ep},
                       os.path.join(args.out_dir, f"imgdistill_{args.which}_epoch{ep}.pth"))
            print(f"  [epoch {ep}] saved", flush=True)
    print("DONE.")


if __name__ == "__main__":
    main()