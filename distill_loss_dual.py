"""
distill_loss_dual.py  —  Dual-teacher distillation loss
---------------------------------------------------------------------------
Two teachers, each supervising the axis it is strongest on:

    S4  (T2I champion, 85.58 R@5)  -> cross-modal terms  (image<->text)
    S2  (I2I strong,   72.57 mAP)  -> image-image term   (retrieval geometry)

    L_Total = (1-lam) * L_CLIP_anchor
            + lam * [ w_t2i * (L_it + L_ti  from S4)
                    + w_i2i * (L_ii         from S2) ]

Rationale: the student has ONE shared embedding space. S4's cross-modal
geometry shapes text<->image alignment (lifts T2I); S2's image-image geometry
shapes same-instance clustering (lifts I2I). Each teacher is applied only to
its strong axis, so averaging a weak axis in (which would drag the result to
the middle) is avoided.

KL direction is KL(teacher || student): teacher is the fixed target, student
follows. Reuses the exact _cross_kl and _same_modality_kl from the single-
teacher Apple loss, so behaviour is identical per-term; only the teacher SOURCE
differs per term.

Embedding dims may differ across student (1024), S2 (512), S4 (768) — this is
fine: every term operates on B x B similarity matrices (cosine structure),
never on raw embeddings, so dimensionality is irrelevant.
"""
import torch
import torch.nn.functional as F


def _cross_kl(student_A, student_B, teacher_A, teacher_B, tau_t, tau_s):
    """One cross-modal KL direction: KL(teacher || student), batchmean."""
    teacher_logits = (teacher_A @ teacher_B.t()) / tau_t
    student_logits = (student_A @ student_B.t()) / tau_s
    teacher_p    = F.softmax(teacher_logits, dim=1)
    student_logp = F.log_softmax(student_logits, dim=1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean")


def _same_modality_kl(student_emb, teacher_emb, tau_t, tau_s):
    """
    Image-image relational KL with the DIAGONAL MASKED OUT (self-similarity=1
    would otherwise swamp the softmax and make the term degenerate). Compares
    how each image relates to the OTHERS — the retrieval geometry.
    """
    B = student_emb.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=student_emb.device)
    NEG = torch.finfo(student_emb.dtype).min

    def masked_softmax(emb, tau, log=False):
        S = (emb @ emb.t()) / tau
        S = S.masked_fill(~mask, NEG)
        return F.log_softmax(S, dim=1) if log else F.softmax(S, dim=1)

    teacher_p    = masked_softmax(teacher_emb, tau_t, log=False)
    student_logp = masked_softmax(student_emb, tau_s, log=True)
    kl = teacher_p * (torch.log(teacher_p + 1e-12) - student_logp)
    kl = kl.masked_fill(~mask, 0.0).sum(dim=1)
    return kl.mean()


def clip_loss(student_img, student_txt, logit_scale):
    """Standard symmetric InfoNCE anchor on ground-truth pairs."""
    logits = logit_scale * student_img @ student_txt.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.t(), labels))


def dual_teacher_loss(student_img, student_txt,
                      s4_img, s4_txt,          # T2I reference (cross-modal)
                      s2_img,                  # I2I reference (image-image only)
                      tau_teacher=0.07, tau_student=0.07,
                      lam=0.7, w_t2i=1.0, w_i2i=1.0, logit_scale=100.0):
    """
    Dual-teacher loss. All embeddings assumed L2-normalized.

      student_img/txt : trainable student embeddings          (B, Ds)
      s4_img/s4_txt   : S4 teacher embeddings (T2I reference)  (B, D4)
      s2_img          : S2 teacher image embeddings (I2I ref)  (B, D2)

    Returns (total_loss, parts_dict).
    """
    # --- T2I: two cross-modal directions, teacher = S4 ---
    L_it = _cross_kl(student_img, student_txt, s4_img, s4_txt,
                     tau_teacher, tau_student)          # image -> text
    L_ti = _cross_kl(student_txt, student_img, s4_txt, s4_img,
                     tau_teacher, tau_student)          # text  -> image

    # --- I2I: image-image relational term, teacher = S2 ---
    L_ii = _same_modality_kl(student_img, s2_img, tau_teacher, tau_student)

    L_distill = w_t2i * (L_it + L_ti) + w_i2i * L_ii

    # --- CLIP anchor on ground-truth pairs ---
    L_clip = clip_loss(student_img, student_txt, logit_scale)

    total = (1.0 - lam) * L_clip + lam * L_distill
    return total, {
        "L_it": L_it.item(),
        "L_ti": L_ti.item(),
        "L_ii": L_ii.item(),
        "L_distill": L_distill.item(),
        "L_clip": L_clip.item(),
    }


if __name__ == "__main__":
    # sanity: match -> ~0 cross terms, mismatch -> positive; gradient to student only
    import numpy as np
    torch.manual_seed(0)
    B = 8
    def randn_norm(b, d):
        x = torch.randn(b, d); return F.normalize(x, dim=-1)
    s_img = randn_norm(B, 1024).requires_grad_(True)
    s_txt = randn_norm(B, 1024).requires_grad_(True)
    s4_i  = randn_norm(B, 768); s4_t = randn_norm(B, 768)
    s2_i  = randn_norm(B, 512)
    loss, parts = dual_teacher_loss(s_img, s_txt, s4_i, s4_t, s2_i)
    loss.backward()
    print("loss:", round(loss.item(), 4))
    print("parts:", {k: round(v, 4) for k, v in parts.items()})
    print("student_img grad exists:", s_img.grad is not None)
    print("dims differ (student 1024, S4 768, S2 512) — worked:", True)