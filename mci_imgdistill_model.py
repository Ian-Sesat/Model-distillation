import torch
import torch.nn as nn
import torch.nn.functional as F

MCI_TAGS = {
    "mci0": ("fastvit_mci0.apple_mclip2_dfndr2b", 1024),
    "mci2": ("fastvit_mci2.apple_mclip2_dfndr2b", 1280),
}

class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)

class MCiImgDistill(nn.Module):
    def __init__(self, which="mci0", shared_dim=512, device="cuda"):
        super().__init__()
        self.device = device
        tag, in_dim = MCI_TAGS[which]
        import timm
        self.image_encoder = timm.create_model(tag, pretrained=True, num_classes=0)
        self.image_proj = Projector(in_dim, shared_dim)

    def image_transform(self, image_size=256):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),          # identity norm for MCi
        ])

    def encode_image(self, images):
        feats = self.image_encoder(images)      # (B, in_dim) trainable
        feats = self.image_proj(feats)          # (B, 512)   trainable
        return F.normalize(feats, dim=-1)

    forward = encode_image


def image_image_kl(student_img, teacher_img, tau_t=0.07, tau_s=0.07):
    """Diagonal-masked image-image relational KL: match student's image-image
    similarity structure to the teacher's. (Same term as the dual-teacher L_ii.)"""
    B = student_img.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=student_img.device)
    NEG = torch.finfo(student_img.dtype).min
    def msm(emb, tau, log=False):
        S = (emb @ emb.t()) / tau
        S = S.masked_fill(~mask, NEG)
        return F.log_softmax(S, dim=1) if log else F.softmax(S, dim=1)
    tp = msm(teacher_img, tau_t, log=False)
    sp = msm(student_img, tau_s, log=True)
    kl = tp * (torch.log(tp + 1e-12) - sp)
    return kl.masked_fill(~mask, 0.0).sum(dim=1).mean()


if __name__ == "__main__":
    m = MCiImgDistill("mci0", shared_dim=512, device="cpu").to("cpu")
    x = torch.randn(4, 3, 256, 256)
    e = m.encode_image(x)
    print("mci0 projected emb:", e.shape, "(expect (4,512))  L2:", e.norm(dim=-1)[:2])
    # loss sanity
    t = F.normalize(torch.randn(4, 768), dim=-1)   # teacher img emb (768-d), different dim OK
    print("image_image_kl (random):", round(image_image_kl(e, t).item(), 4), "(>0)")
    print("trainable params:", sum(p.numel() for p in m.parameters() if p.requires_grad))