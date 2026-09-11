import torch
import torch.nn.functional as F


def _cross_kl(student_A, student_B, teacher_A, teacher_B, tau_t, tau_s):
    teacher_logits = (teacher_A @ teacher_B.t()) / tau_t
    student_logits = (student_A @ student_B.t()) / tau_s
    teacher_p    = F.softmax(teacher_logits, dim=1)
    student_logp = F.log_softmax(student_logits, dim=1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean")


def _same_modality_kl(student_emb, teacher_emb, tau_t, tau_s):
    B = student_emb.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=student_emb.device)
    NEG = torch.finfo(student_emb.dtype).min

    def masked_logsoftmax(emb, tau, log=False):
        S = (emb @ emb.t()) / tau
        S = S.masked_fill(~mask, NEG)          # diagonal -> -inf before softmax
        return F.log_softmax(S, dim=1) if log else F.softmax(S, dim=1)

    teacher_p    = masked_logsoftmax(teacher_emb, tau_t, log=False)
    student_logp = masked_logsoftmax(student_emb, tau_s, log=True)
    # zero out the (masked) diagonal contributions, then batchmean over rows
    kl = teacher_p * (torch.log(teacher_p + 1e-12) - student_logp)
    kl = kl.masked_fill(~mask, 0.0).sum(dim=1)
    return kl.mean()


def clip_loss(img, txt, logit_scale):
    logits = logit_scale * (img @ txt.t())          # (B, B)
    labels = torch.arange(img.size(0), device=img.device)
    loss_i = F.cross_entropy(logits, labels)        # image -> text
    loss_t = F.cross_entropy(logits.t(), labels)    # text  -> image
    return 0.5 * (loss_i + loss_t)


def apple_distillation_loss(student_img, student_txt,
                            teacher_img, teacher_txt,
                            tau_teacher=0.07, tau_student=0.07,
                            lam=0.7, logit_scale=100.0, w_ii=0.0):
    L_it = _cross_kl(student_img, student_txt, teacher_img, teacher_txt,
                     tau_teacher, tau_student)     # image -> text
    L_ti = _cross_kl(student_txt, student_img, teacher_txt, teacher_img,
                     tau_teacher, tau_student)     # text  -> image
    L_distill = L_it + L_ti
    L_ii = _same_modality_kl(student_img, teacher_img, tau_teacher, tau_student)
    if w_ii > 0.0:
        L_distill = L_distill + w_ii * L_ii

    L_clip = clip_loss(student_img, student_txt, logit_scale)

    total = (1.0 - lam) * L_clip + lam * L_distill
    return total, {
        "L_it": L_it.item(),
        "L_ti": L_ti.item(),
        "L_ii": L_ii.item(),
        "L_distill": L_distill.item(),
        "L_clip": L_clip.item(),
    }

# Self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 8, 512
    t_img = F.normalize(torch.randn(B, D), dim=-1)
    t_txt = F.normalize(torch.randn(B, D), dim=-1)

    print("=== TEST 1: student==teacher -> distill ~0 (CLIP term stays >0) ===")
    tot, parts = apple_distillation_loss(t_img, t_txt, t_img, t_txt, lam=1.0)
    print("  parts:", {k: round(v, 6) for k, v in parts.items()})
    print("  L_distill near zero:", parts["L_distill"] < 1e-5)

    print("\n=== TEST 2: mismatched student -> distill positive ===")
    s_img = F.normalize(torch.randn(B, D), dim=-1)
    s_txt = F.normalize(torch.randn(B, D), dim=-1)
    tot2, parts2 = apple_distillation_loss(s_img, s_txt, t_img, t_txt, lam=1.0)
    print("  parts:", {k: round(v, 6) for k, v in parts2.items()})
    print("  L_distill positive:", parts2["L_distill"] > 0)

    print("\n=== TEST 3: distill loss grows with disagreement (lam=1) ===")
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        si = F.normalize((1 - a) * t_img + a * s_img, dim=-1)
        st = F.normalize((1 - a) * t_txt + a * s_txt, dim=-1)
        tot, p = apple_distillation_loss(si, st, t_img, t_txt, lam=1.0)
        print(f"  alpha={a:.2f} -> L_distill={p['L_distill']:.5f}")

    print("\n=== TEST 4: CLIP anchor lower when pairs align ===")
    # perfectly aligned student pairs (img_i == txt_i direction) -> low CLIP loss
    aligned = F.normalize(torch.randn(B, D), dim=-1)
    _, pa = apple_distillation_loss(aligned, aligned, t_img, t_txt, lam=0.0)
    _, pm = apple_distillation_loss(s_img, s_txt, t_img, t_txt, lam=0.0)
    print(f"  CLIP loss aligned : {pa['L_clip']:.4f}")
    print(f"  CLIP loss random  : {pm['L_clip']:.4f}")
    print("  aligned < random  :", pa["L_clip"] < pm["L_clip"])

    print("\n=== TEST 5: gradient flows to student ===")
    sg_i = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
    sg_t = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
    tot, _ = apple_distillation_loss(sg_i, sg_t, t_img, t_txt)
    tot.backward()
    print("  student_img grad:", sg_i.grad is not None)
    print("  student_txt grad:", sg_t.grad is not None)
