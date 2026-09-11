import torch
import torch.nn.functional as F

from distill_matrices import similarity_matrices


def _row_kl(student_S, teacher_S, tau):
    """
    KL(teacher || student) averaged over rows, on two (B,B) similarity matrices.
    """
    student_logp = F.log_softmax(student_S / tau, dim=1)   # (B,B) log-probs
    teacher_p    = F.softmax(teacher_S / tau, dim=1)       # (B,B) probs
    return F.kl_div(student_logp, teacher_p, reduction="batchmean")


def distillation_loss(student_img, student_txt,
                      teacher_img, teacher_txt,
                      tau=0.07, w_ii=1.0, w_tt=1.0, w_it=1.0):
                          
    S_ii_s, S_tt_s, S_it_s = similarity_matrices(student_img, student_txt)
    S_ii_t, S_tt_t, S_it_t = similarity_matrices(teacher_img, teacher_txt)

    L_ii = _row_kl(S_ii_s, S_ii_t, tau)
    L_tt = _row_kl(S_tt_s, S_tt_t, tau)
    L_it = _row_kl(S_it_s, S_it_t, tau)

    total = w_ii * L_ii + w_tt * L_tt + w_it * L_it
    return total, {"L_ii": L_ii.item(), "L_tt": L_tt.item(), "L_it": L_it.item()}

# Self-test: the two properties that MUST hold.
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 8, 512

    # a random teacher embedding set
    t_img = F.normalize(torch.randn(B, D), dim=-1)
    t_txt = F.normalize(torch.randn(B, D), dim=-1)

    print("=== TEST 1: student == teacher -> loss ~ 0 ===")
    total, parts = distillation_loss(t_img, t_txt, t_img, t_txt)
    print("parts:", {k: round(v, 8) for k, v in parts.items()})
    print("total:", round(total.item(), 8))
    print("near zero:", total.item() < 1e-6)

    print("\n=== TEST 2: mismatched student -> positive loss ===")
    s_img = F.normalize(torch.randn(B, D), dim=-1)   # unrelated
    s_txt = F.normalize(torch.randn(B, D), dim=-1)
    total2, parts2 = distillation_loss(s_img, s_txt, t_img, t_txt)
    print("parts:", {k: round(v, 6) for k, v in parts2.items()})
    print("total:", round(total2.item(), 6))
    print("positive:", total2.item() > 0)

    print("\n=== TEST 3: loss grows with disagreement ===")
    # interpolate student from teacher (match) toward random (mismatch)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        si = F.normalize((1 - alpha) * t_img + alpha * s_img, dim=-1)
        st = F.normalize((1 - alpha) * t_txt + alpha * s_txt, dim=-1)
        tot, _ = distillation_loss(si, st, t_img, t_txt)
        print(f"  alpha={alpha:.2f} (0=match,1=random) -> loss={tot.item():.5f}")

    print("\n=== TEST 4: gradient flows to student only ===")
    s_img_g = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
    s_txt_g = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
    tot, _ = distillation_loss(s_img_g, s_txt_g, t_img, t_txt)
    tot.backward()
    print("student_img grad exists:", s_img_g.grad is not None)
    print("teacher has no grad    :", t_img.requires_grad == False)
