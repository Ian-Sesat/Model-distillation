# Model Distillation

Knowledge distillation of a strong CLIP teacher (MobileCLIP2) into a smaller CLIP
student (e.g. OpenAI RN50), following Apple's Dataset-Reinforcement formulation.
The teacher's image and text embeddings are cached once over an image–text
dataset, and the student is trained to reproduce the teacher's cross-modal
similarity structure — lifting a weak student on both image-to-image (I2I) and
text-to-image (T2I) retrieval.

## Method

The distillation loss combines two cross-modal KL terms with a standard CLIP
anchor on ground-truth pairs:

```
L_total = (1 − λ)·L_CLIP  +  λ·L_distill

L_distill = KL( S_τt(Ψ_img, Ψ_txt) ‖ S_τs(Φ_img, Φ_txt) )   image → text
          + KL( S_τt(Ψ_txt, Ψ_img) ‖ S_τs(Φ_txt, Φ_img) )   text → image
          + w_ii · L_ii                                       (optional)
```

where Ψ are the frozen **teacher** embeddings, Φ are the trainable **student**
embeddings, and `S_τ(U,V)` is the row-wise softmax of `U·Vᵀ / τ`. Both student
towers are trainable, so the text tower follows the pulled image space and T2I is
preserved.

- **Cross-modal terms** (`L_it`, `L_ti`) transfer the teacher's image–text
  geometry — this is the core of Apple's Eq. 1 and lifts both retrieval
  directions.
- **Optional image–image term** (`L_ii`, weight `w_ii`) directly matches the
  student's image–image similarity structure to the teacher's, targeting I2I
  retrieval geometry the way MCIP-style fine-tuning does. Its diagonal is masked
  (self-similarity is uninformative and would otherwise dominate the softmax).
  `w_ii = 0` recovers the pure Apple cross-modal loss.

The teacher is never loaded as a model during training — only its precomputed
embeddings are used, which is what makes training efficient.

## Repository layout

| File | Purpose |
|------|---------|
| `distill_extract_teacher.py` | Cache the teacher's image + text embeddings over the dataset (one-time). fp32, NaN-safe, resumable. |
| `distill_matrices.py` | Similarity-matrix helpers (image–image, text–text, image–text) used by the loss. |
| `distill_loss_apple.py` | The distillation loss: two cross-modal KL directions + CLIP anchor, plus the optional diagonal-masked image–image term. |
| `distill_loss.py` | The earlier three-term relational loss. **Superseded** by `distill_loss_apple.py`; kept for reference and not imported by training. |
| `distill_dataset.py` | Dataset that yields `(image, caption, original_pair_index)`; the index keeps student batches aligned to the teacher cache. |
| `distill_train.py` | Training loop: live student encode, teacher targets gathered from the cache by index, loss, optimizer, epoch checkpoints. |

## Pipeline

### 1. Cache the teacher embeddings (one-time)

```bash
python distill_extract_teacher.py \
  --teacher_model mobileclip2_s2 \
  --pairs_dir  /path/to/cc12m \
  --out_dir    /path/to/teacher_cache
```

Runs two passes (image then text) and writes
`img_feats_*`, `text_feats_*`, `valid_*`, `captions_*` to `--out_dir`. It is
resumable and skips work already cached.

### 2. Train the student

```bash
python distill_train.py \
  --student rn50 --clip_lib --clip_lib_name RN50 \
  --teacher_dir   /path/to/teacher_cache \
  --teacher_model mobileclip2_s2 \
  --pairs_dir     /path/to/cc12m \
  --out_dir       /path/to/output \
  --batch 64 --epochs 5 --lr 1e-5 --lam 0.7 --w_ii 3.0
```

One checkpoint is saved per epoch (`distill_<student>_epoch<N>.pth`).

## Key options (`distill_train.py`)

| Flag | Meaning |
|------|---------|
| `--student` | Student model key (loaded via `mcip_train.MODELS`). |
| `--clip_lib` / `--clip_lib_name` | Load the student through the original CLIP library (`clip.load`) instead of open_clip — for a deployment-faithful OpenAI RN50 (QuickGELU, fp32). |
| `--teacher_dir` / `--teacher_model` | Location and tag of the cached teacher embeddings. |
| `--pairs_dir` | Image–text dataset (e.g. CC12M: `captions.tsv` + images). |
| `--batch`, `--epochs`, `--lr` | Standard training controls. |
| `--lam` | Weight on distillation vs. the CLIP anchor (default 0.7). |
| `--w_ii` | Weight on the optional image–image term (0 = pure cross-modal). |
| `--tau_teacher`, `--tau_student` | Softmax temperatures for teacher/student distributions. |
| `--resume` | Continue from an existing student checkpoint (e.g. add `w_ii` on top of a cross-modal run). |
| `--grad_checkpoint` | Enable gradient checkpointing to save memory. |

## Notes

- **Deployment-faithful student.** With `--clip_lib`, the student is the plain
  OpenAI CLIP-library RN50 (QuickGELU), so the distilled weights are a direct
  swap for a deployment model using the same variant. Without it, the student
  loads through open_clip (standard GELU), which is a different variant.
- **Relation to Apple's Eq. 1.** The loss implements Apple's cross-modal KL
  structure. It uses a single teacher (no K-teacher ensemble) and a `batchmean`
  normalization that differs from the paper's exact constant by a factor; a
  constant loss scale only rescales the effective learning rate. The image–image
  term is an addition, not part of Eq. 1.
- **Dependencies.** PyTorch, `open_clip` and/or the OpenAI `clip` library, `tqdm`,
  NumPy, and a local `mcip_train` module providing `MODELS` / `load_model`.

## Status

The pipeline distills MobileCLIP2-S2 into RN50 and lifts the student on both
retrieval directions over its baseline. The cross-modal loss improves both
directions together; the image–image term adds further I2I without costing T2I.
Gains saturate with epochs, indicating the teacher's own accuracy is the binding
ceiling — reaching further requires a stronger or ensemble teacher rather than
more training.
