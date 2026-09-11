import torch

def similarity_matrices(img, txt):
    if img.shape != txt.shape:
        raise ValueError(f"img {tuple(img.shape)} and txt {tuple(txt.shape)} "
                         f"must have the same shape")

    S_ii = img @ img.t()
    S_tt = txt @ txt.t()
    S_it = img @ txt.t()
    return S_ii, S_tt, S_it

if __name__ == "__main__":
    import torch.nn.functional as F

    import math
    def unit(theta):
        return [math.cos(theta), math.sin(theta)]

    img = torch.tensor([
        unit(math.radians(0)),     # img1
        unit(math.radians(37)),    # img2  cos(37 deg) ~ 0.7986 vs img1
        unit(math.radians(90)),    # img3  cos(90) = 0 vs img1
    ], dtype=torch.float32)

    # Texts: make text_i roughly aligned with image_i but not identical, so
    # S_it has a strong (but < 1) diagonal — like a real CLIP matrix.
    txt = torch.tensor([
        unit(math.radians(10)),    # text1 near img1
        unit(math.radians(45)),    # text2 near img2
        unit(math.radians(80)),    # text3 near img3
    ], dtype=torch.float32)

    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)

    S_ii, S_tt, S_it = similarity_matrices(img, txt)

    torch.set_printoptions(precision=4, sci_mode=False)
    print("S_ii (image-image):\n", S_ii)
    print("\nS_tt (text-text):\n", S_tt)
    print("\nS_it (image-text):\n", S_it)

    print("\n--- structural checks ---")
    # 1. diagonals of S_ii and S_tt are 1.0
    print("S_ii diag ~1 :", torch.allclose(S_ii.diag(), torch.ones(3), atol=1e-4))
    print("S_tt diag ~1 :", torch.allclose(S_tt.diag(), torch.ones(3), atol=1e-4))
    # 2. S_ii and S_tt symmetric
    print("S_ii symmetric:", torch.allclose(S_ii, S_ii.t(), atol=1e-5))
    print("S_tt symmetric:", torch.allclose(S_tt, S_tt.t(), atol=1e-5))
    # 3. S_it NOT necessarily symmetric
    print("S_it symmetric:", torch.allclose(S_it, S_it.t(), atol=1e-5),
          "(expected False)")

    print("\n--- hand-checkable values ---")
    # cos between img1 (0 deg) and img2 (37 deg) should be cos(37 deg)
    print(f"S_ii[0,1] = {S_ii[0,1]:.4f}   (expect cos 37 deg = {math.cos(math.radians(37)):.4f})")
    # img1 (0) vs img3 (90) = cos 90 = 0
    print(f"S_ii[0,2] = {S_ii[0,2]:.4f}   (expect cos 90 deg = 0.0000)")
    # S_it diagonal: img1(0) vs text1(10) = cos 10 deg
    print(f"S_it[0,0] = {S_it[0,0]:.4f}   (expect cos 10 deg = {math.cos(math.radians(10)):.4f})")
    # off-diagonal cross: img1(0) vs text3(80) = cos 80 deg
    print(f"S_it[0,2] = {S_it[0,2]:.4f}   (expect cos 80 deg = {math.cos(math.radians(80)):.4f})")
