import os, argparse, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import mcip_train as M
from distill_dataset import DistillDataset, collate_drop_none


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True,
                    help="open_clip student key in mcip_train.MODELS (e.g. mobileclip2_s2)")
    ap.add_argument("--i2i_img", required=True,
                    help="npy of teacher image features (e.g. MCIP-SigLIP2 over the pairs)")
    ap.add_argument("--i2i_valid", required=True, help="npy valid mask for the teacher features")
    ap.add_argument("--pairs_dir", required=True, help="image-caption pairs dir (captions.tsv)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="GENTLE lr — the student is already strong; we refine its "
                         "image geometry, not overwrite it. Too high wrecks S2.")
    ap.add_argument("--tau_teacher", type=float, default=0.07)
    ap.add_argument("--tau_student", type=float, default=0.07)
    ap.add_argument("--w_anchor", type=float, default=0.3,
                    help="weight on the light CLIP anchor (keeps img-text compatible "
                         "with the frozen text tower). Set 0 to disable.")
    ap.add_argument("--logit_scale", type=float, default=100.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=50)
    return ap.parse_args()


def image_image_kl(student_img, teacher_img, tau_t, tau_s):
    """Diagonal-masked image-image relational KL. Matches how each image relates
    to the OTHERS (self-similarity=1 masked out so it doesn't swamp softmax)."""
    B = student_img.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=student_img.device)
    NEG = torch.finfo(student_img.dtype).min

    def masked_softmax(emb, tau, log=False):
        S = (emb @ emb.t()) / tau
        S = S.masked_fill(~mask, NEG)
        return F.log_softmax(S, dim=1) if log else F.softmax(S, dim=1)

    teacher_p    = masked_softmax(teacher_img, tau_t, log=False)
    student_logp = masked_softmax(student_img, tau_s, log=True)
    kl = teacher_p * (torch.log(teacher_p + 1e-12) - student_logp)
    kl = kl.masked_fill(~mask, 0.0).sum(dim=1)
    return kl.mean()


def clip_anchor(s_img, s_txt, logit_scale):
    logits = logit_scale * s_img @ s_txt.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    # --- load student (open_clip); BOTH towers exist but only IMAGE trains ---
    model, preprocess, processor, tokenizer, emb_dim = M.load_model(args.student)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(state, strict=False)
        print(f"resumed from {args.resume}")
    model = model.to(device).train()

    # freeze EVERYTHING, then unfreeze only the image (visual) tower
    n_img, n_txt = 0, 0
    for name, p in model.named_parameters():
        if name.startswith("visual"):
            p.requires_grad = True;  n_img += p.numel()
        else:
            p.requires_grad = False; n_txt += p.numel()
    print(f"IMAGE tower trainable: {n_img:,} params | TEXT tower FROZEN: {n_txt:,} params")

    # --- teacher image features via mmap (RAM-safe) ---
    t_img = np.load(args.i2i_img, mmap_mode='r')
    t_valid = np.load(args.i2i_valid).astype(bool)
    print(f"teacher img feats {t_img.shape} (mmap) | valid {t_valid.sum():,}")

    ds = DistillDataset(args.pairs_dir, preprocess, valid_mask=t_valid)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=collate_drop_none, drop_last=True)
    print(f"dataset: {len(ds):,} valid pairs | batch {args.batch} | {len(loader):,} steps/epoch")

    # only image-tower params go to the optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    step = 0
    pbar = tqdm(total=total_steps, desc="i2i-distill", dynamic_ncols=True, mininterval=10.0)
    for epoch in range(args.epochs):
        for batch in loader:
            if batch is None:
                continue
            imgs, caps, idxs = batch
            imgs = imgs.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)

            s_img = F.normalize(model.encode_image(imgs).float(), dim=-1)

            # gather teacher image rows for this batch (mmap + copy)
            with torch.no_grad():
                idxs_np = idxs.cpu().numpy()
                t_i = F.normalize(torch.from_numpy(t_img[idxs_np].copy()).float()
                                  .to(device, non_blocking=True), dim=-1)

            L_i2i = image_image_kl(s_img, t_i, args.tau_teacher, args.tau_student)

            # light anchor: keep image compatible with the (frozen) text tower
            if args.w_anchor > 0:
                tokens = tokenizer(caps).to(device)
                with torch.no_grad():
                    s_txt = F.normalize(model.encode_text(tokens).float(), dim=-1)  # frozen text
                L_anchor = clip_anchor(s_img, s_txt, args.logit_scale)
            else:
                L_anchor = torch.zeros((), device=device)

            loss = L_i2i + args.w_anchor * L_anchor

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()

            pbar.update(1)
            pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}", i2i=f"{L_i2i.item():.4f}")
            if step % args.log_every == 0:
                tqdm.write(f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} "
                           f"i2i={L_i2i.item():.4f} anchor={L_anchor.item():.4f} "
                           f"lr={sched.get_last_lr()[0]:.2e}")
            step += 1

        if epoch + 1 == args.epochs:
            out = os.path.join(args.out_dir, f"i2idistill_{args.student}_final.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, out)
            print(f"  saved {out}", flush=True)

    pbar.close()
    print("DONE.")


if __name__ == "__main__":
    main()
