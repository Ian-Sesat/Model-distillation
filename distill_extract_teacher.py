"""
Usage (baseline teacher, no checkpoint):
    python distill_extract_teacher.py --model mobileclip2_s2 \
        --pairs_dir /media/.../cc12m \
        --out_dir   /media/.../distill_teacher_s2 \
        --batch 64 --workers 8

"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mcip_train as M                      # MODELS, load_model, get_text_embeddings
from realign_extract_features import (       # reuse the proven CC12M readers
    load_pairs, ImageOnlyDataset,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="teacher model key in mcip_train.MODELS (e.g. mobileclip2_s2)")
    ap.add_argument("--pairs_dir", required=True,
                    help="folder containing captions.tsv + images")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--teacher_ckpt", default=None,
                    help="OPTIONAL fine-tuned teacher checkpoint. Omit to use the "
                         "baseline open_clip weights as the teacher.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--text_batch", type=int, default=64)
    ap.add_argument("--skip_images", action="store_true",
                    help="skip PASS 1 and load existing img_feats .npy "
                         "(resume after an image pass already completed)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    # load teacher (baseline weights already applied by load_model) 
    model, preprocess, processor, tokenizer, emb_dim = M.load_model(args.model)

    if args.teacher_ckpt is not None:
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded teacher ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("no checkpoint given -> using BASELINE open_clip weights as teacher")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(False)
        print("gradient checkpointing: OFF (inference)")
    print(f"teacher '{args.model}' ready | emb_dim={emb_dim}")

    # load CC12M pairs 
    pairs = load_pairs(args.pairs_dir)
    n = len(pairs)
    print(f"pairs: {n:,}", flush=True)

    img_feats_path = os.path.join(args.out_dir, f"img_feats_{args.model}.npy")
    txt_feats_path = os.path.join(args.out_dir, f"text_feats_{args.model}.npy")
    valid_path     = os.path.join(args.out_dir, f"valid_{args.model}.npy")
    caps_path      = os.path.join(args.out_dir, f"captions_{args.model}.json")

    all_img   = np.zeros((n, emb_dim), dtype=np.float32)
    all_txt   = np.zeros((n, emb_dim), dtype=np.float32)
    all_valid = np.zeros((n,), dtype=bool)

    # PASS 1 — IMAGE embeddings (fp32, finite-guarded)  [structure from
    #          realign_extract_features.py]

    if args.skip_images:
        print("PASS 1/2: SKIPPED — loading existing img_feats", flush=True)
        loaded = np.load(img_feats_path)
        if loaded.shape != all_img.shape:
            raise RuntimeError(
                f"existing img_feats shape {loaded.shape} != expected "
                f"{all_img.shape}; cannot resume")
        all_img = loaded
        # provisionally mark all rows valid; text pass re-validates below.
        all_valid[:] = True
    else:
        _run_image_pass(model, pairs, preprocess, args, device,
                        all_img, all_valid, img_feats_path)

    _run_text_pass(model, tokenizer, pairs, args, device, all_img, all_txt, all_valid,
                   txt_feats_path, valid_path, caps_path, emb_dim, n)


def _run_image_pass(model, pairs, preprocess, args, device,
                    all_img, all_valid, img_feats_path):
    import torch.nn.functional as F
    ds = ImageOnlyDataset(pairs, preprocess)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    nan_batches = 0
    done = 0
    print("PASS 1/2: image embeddings ...", flush=True)
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = model.encode_image(imgs)          # fp32, NO autocast
            feats = F.normalize(feats, dim=-1).float()

            finite = torch.isfinite(feats).all(dim=-1)
            if not finite.all():
                nan_batches += 1
                feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

            feats  = feats.cpu().numpy()
            finite = finite.cpu().numpy()
            for j, gi in enumerate(idxs.tolist()):
                if gi >= 0 and finite[j]:
                    all_img[gi] = feats[j]
                    all_valid[gi] = True     # provisional; text pass may invalidate
            done += len(idxs)

    if nan_batches:
        print(f"  WARNING: {nan_batches} image batch(es) had non-finite rows")
    np.save(img_feats_path, all_img)
    print(f"  saved {img_feats_path}")


def _run_text_pass(model, tokenizer, pairs, args, device, all_img, all_txt, all_valid,
                   txt_feats_path, valid_path, caps_path, emb_dim, n):
    import torch.nn.functional as F
    captions = [c for _, c in pairs]
    print("PASS 2/2: text embeddings ...", flush=True)

    txt_nan = 0
    with torch.no_grad():
        for start in range(0, n, args.text_batch):
            batch_caps = captions[start:start + args.text_batch]
            # get_text_embeddings expects a NESTED list (one caption-list per
            # image) and returns a TUPLE (embeddings, caption_indices). For
            # distillation each pair has exactly one caption, so wrap each in a
            # singleton list; the returned order then matches batch order 1:1.
            nested = [[c] for c in batch_caps]
            txt_embs, _ = M.get_text_embeddings(model, nested, args.model, tokenizer)
            # already (T, D) L2-normalised on GPU; move to CPU float
            t = txt_embs.detach().float().cpu()
            # re-normalise defensively (no-op if already unit norm)
            t = F.normalize(t, dim=-1)

            finite = torch.isfinite(t).all(dim=-1)
            if not finite.all():
                txt_nan += 1
                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

            t = t.numpy()
            finite = finite.numpy()
            for j in range(len(batch_caps)):
                gi = start + j
                all_txt[gi] = t[j]
                if not finite[j]:
                    all_valid[gi] = False     # invalid text -> pair unusable

    if txt_nan:
        print(f"  WARNING: {txt_nan} text batch(es) had non-finite rows")
    np.save(txt_feats_path, all_txt)
    np.save(valid_path, all_valid)
    with open(caps_path, "w") as f:
        json.dump(captions, f)

    # report 
    v = int(all_valid.sum())
    print(f"\nDONE. valid pairs: {v:,}/{n:,}")
    print(f"  img_feats : (already saved)  shape ({n}, {emb_dim})")
    print(f"  text_feats: {txt_feats_path}  shape ({n}, {emb_dim})")
    print(f"  valid mask: {valid_path}")
    # sanity: are cached vectors unit-norm on the valid rows?
    if v > 0:
        vi = np.where(all_valid)[0][:1000]
        in_ = np.linalg.norm(all_img[vi], axis=1).mean()
        tn_ = np.linalg.norm(all_txt[vi], axis=1).mean()
        print(f"  mean L2 norm (first 1000 valid): img={in_:.4f} text={tn_:.4f} (want ~1.0)")


if __name__ == "__main__":
    main()