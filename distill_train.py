import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import mcip_train as M
from distill_loss_apple import apple_distillation_loss
from distill_dataset import DistillDataset, collate_drop_none


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True,
                    help="student model key in mcip_train.MODELS (e.g. rn50x4)")
    ap.add_argument("--clip_lib", action="store_true",
                    help="load the student via the ORIGINAL CLIP LIBRARY "
                         "(clip.load) instead of open_clip. Use for a "
                         "deployment-faithful OpenAI RN50 (QuickGELU, fp32).")
    ap.add_argument("--clip_lib_name", default="RN50",
                    help="clip-library model name when --clip_lib is set")
    ap.add_argument("--teacher_dir", required=True,
                    help="dir with img_feats_*, text_feats_*, valid_* from Section 1")
    ap.add_argument("--teacher_model", required=True,
                    help="teacher model tag used in the cache filenames (e.g. mobileclip2_s2)")
    ap.add_argument("--pairs_dir", required=True, help="CC12M dir (captions.tsv + images)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau_teacher", type=float, default=0.07)
    ap.add_argument("--tau_student", type=float, default=0.07)
    ap.add_argument("--lam", type=float, default=0.7,
                    help="lambda: weight on distillation vs CLIP anchor "
                         "(L = (1-lam)*L_clip + lam*L_distill)")
    ap.add_argument("--logit_scale", type=float, default=100.0,
                    help="scale for the CLIP anchor loss")
    ap.add_argument("--w_ii", type=float, default=0.0,
                    help="weight on the EXTRA image-image relational term "
                         "(0 = pure Apple cross-modal loss; >0 directly targets "
                         "I2I retrieval geometry, like MCIP). Try 1.0-5.0.")
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint (.pth) to load into the student "
                         "before training — continue from an already-distilled "
                         "model instead of the pretrained base.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad_checkpoint", action="store_true",
                    help="enable gradient checkpointing on the student (saves "
                         "memory, allows larger batch)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    return ap.parse_args()



def load_clip_library_student(clip_name, device):
    """
    Load an OpenAI CLIP-library model as a TRAINABLE student.

    Differences from open_clip that must be handled:
      * clip.load() puts the model in fp16 when device is CUDA. We call
        .float() to train in fp32 (fp16 training is numerically fragile;
        fp32 is the safe default and avoids overflow-to-NaN).
      * tokenizer is clip.tokenize (not the open_clip tokenizer).
      * preprocess transform comes from clip.load().
    Returns: model, preprocess, tokenize_fn, emb_dim
    """
    import clip
    model, preprocess = clip.load(clip_name, device=device, jit=False)
    model = model.float()                 # fp16 -> fp32 (avoid overflow NaNs)
    # emb_dim from a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        emb_dim = model.encode_image(dummy).shape[-1]
    return model, preprocess, clip.tokenize, emb_dim


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    # load STUDENT: both towers trainable (do NOT apply the MCIP freeze) 
    if args.clip_lib:
        # deployment-faithful: OpenAI clip-library RN50 (QuickGELU, fp32)
        model, preprocess, tokenizer, emb_dim = load_clip_library_student(
            args.clip_lib_name, device)
        print(f"student loaded via CLIP LIBRARY ({args.clip_lib_name}), fp32")
    else:
        model, preprocess, processor, tokenizer, emb_dim = M.load_model(args.student)
    for p in model.parameters():
        p.requires_grad = True                       # BOTH towers train

    # optionally resume from an already-distilled checkpoint
    if args.resume is not None:
        ck = torch.load(args.resume, map_location="cpu")
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"resumed from {args.resume} | missing={len(missing)} "
              f"unexpected={len(unexpected)}")
    if args.grad_checkpoint and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        print("gradient checkpointing: ON (student)")
    model = model.to(device).train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"student '{args.student}' | trainable params: {n_train:,} | emb_dim={emb_dim}")

    # load TEACHER cache (embeddings only, no model) 
    t_img = np.load(os.path.join(args.teacher_dir, f"img_feats_{args.teacher_model}.npy"))
    t_txt = np.load(os.path.join(args.teacher_dir, f"text_feats_{args.teacher_model}.npy"))
    valid = np.load(os.path.join(args.teacher_dir, f"valid_{args.teacher_model}.npy"))
    print(f"teacher cache: img {t_img.shape} txt {t_txt.shape} | valid {int(valid.sum()):,}")

    t_img_t = torch.from_numpy(t_img).pin_memory()
    t_txt_t = torch.from_numpy(t_txt).pin_memory()

    # dataset / loader: live images + caption + ORIGINAL pair index 
    ds = DistillDataset(args.pairs_dir, preprocess, valid_mask=valid)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=collate_drop_none, drop_last=True)
    print(f"dataset: {len(ds):,} valid pairs | batch {args.batch} | "
          f"{len(loader):,} steps/epoch")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    step = 0
    pbar = tqdm(total=total_steps, desc="distill", dynamic_ncols=True,
                mininterval=10.0)
    for epoch in range(args.epochs):
        for batch in loader:
            if batch is None:
                continue
            imgs, caps, idxs = batch
            imgs = imgs.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)
            s_img = F.normalize(model.encode_image(imgs).float(), dim=-1)
            if args.clip_lib:
                tokens = tokenizer(caps, truncate=True).to(device)
            else:
                tokens = tokenizer(caps).to(device)
            s_txt = F.normalize(model.encode_text(tokens).float(), dim=-1)

            with torch.no_grad():
                idxs_cpu = idxs.cpu()
                teach_img = t_img_t[idxs_cpu].to(device, non_blocking=True)
                teach_txt = t_txt_t[idxs_cpu].to(device, non_blocking=True)

            loss, parts = apple_distillation_loss(
                s_img, s_txt, teach_img, teach_txt,
                tau_teacher=args.tau_teacher, tau_student=args.tau_student,
                lam=args.lam, logit_scale=args.logit_scale, w_ii=args.w_ii)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            pbar.update(1)
            # live loss in the bar's postfix (updates with the bar, ~every 10s)
            pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}",
                             distill=f"{parts['L_distill']:.3f}",
                             clip=f"{parts['L_clip']:.3f}")

            if step % args.log_every == 0:
                # also write a full line to the log for a permanent record
                tqdm.write(
                    f"epoch={epoch} step={step}/{total_steps} "
                    f"loss={loss.item():.4f} "
                    f"it={parts['L_it']:.4f} ti={parts['L_ti']:.4f} "
                    f"ii={parts['L_ii']:.4f} "
                    f"distill={parts['L_distill']:.4f} clip={parts['L_clip']:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e}")

            step += 1

        # save ONE checkpoint per completed epoch 
        ep_ckpt = os.path.join(args.out_dir,
                               f"distill_{args.student}_epoch{epoch+1}.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "step": step},
                   ep_ckpt)
        print(f"  [epoch {epoch+1}] saved {ep_ckpt}", flush=True)

    pbar.close()
    print(f"DONE. {args.epochs} epochs complete.")


if __name__ == "__main__":
    main()