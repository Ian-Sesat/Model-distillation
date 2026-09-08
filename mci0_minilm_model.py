"""
mci0_minilm_model.py
===============================================================================
A NEW small/fast vision-language model:
    image tower: MCi0  (MobileCLIP2-S0's image encoder, via timm)  -> 1024-d
    text  tower: all-MiniLM-L6-v2 (sentence-transformers)          ->  384-d
    projectors : two trainable MLPs mapping each tower into a shared 512-d space

Trained (encoders + projectors, all trainable) by distilling from MCIP-SigLIP2.

KEY DESIGN POINTS (easy to get wrong):
 1. MCi0 preprocessing = IDENTITY norm (mean 0, std 1), 256px, bicubic. NOT ImageNet
    norm. (Verified earlier from the MobileCLIP2 pretrained_cfg.)
 2. MiniLM outputs TOKEN embeddings -> must MEAN-POOL over tokens (masked by the
    attention mask) to get one 384-d sentence vector, THEN project. This is the
    sentence-transformers pooling; skipping/mis-masking it silently wrecks text quality.
 3. Both projected embeddings are L2-normalized before the loss (cosine space).
 4. Everything trainable: MCi0, MiniLM, both projectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Small MLP projector: in_dim -> hidden -> out_dim, with GELU + LayerNorm.
    An MLP (not a bare linear) gives the alignment a bit more capacity, which
    matters because MiniLM was trained for text-similarity, not image alignment."""
    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def masked_mean_pool(token_embeddings, attention_mask):
    """Mean-pool MiniLM token embeddings over the sequence, respecting the mask.
    token_embeddings: (B, T, 384); attention_mask: (B, T) of 0/1.
    This is the standard sentence-transformers mean pooling."""
    mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
    summed = (token_embeddings * mask).sum(dim=1)         # (B, 384)
    counts = mask.sum(dim=1).clamp(min=1e-9)              # (B, 1)
    return summed / counts


class MCi0MiniLM(nn.Module):
    def __init__(self, shared_dim=512, device="cuda"):
        super().__init__()
        self.device = device
        self.shared_dim = shared_dim

        # ---- IMAGE tower: MCi0 via timm (MobileCLIP2 contrastive weights) ----
        import timm
        self.image_encoder = timm.create_model(
            "fastvit_mci0.apple_mclip2_dfndr2b", pretrained=True, num_classes=0)
        img_dim = 1024                                    # MCi0 output dim

        # ---- TEXT tower: MiniLM via HF transformers ----
        from transformers import AutoModel, AutoTokenizer
        self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer     = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        txt_dim = 384                                     # MiniLM output dim

        # ---- projectors (trainable, random init) ----
        self.image_proj = Projector(img_dim, shared_dim)
        self.text_proj  = Projector(txt_dim, shared_dim)

        # CLIP-style learnable temperature for the anchor loss
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    # ---- image preprocessing transform (use in the dataloader) ----
    def image_transform(self, image_size=256):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                        # -> [0,1], identity norm (MCi0)
        ])

    def encode_image(self, images):
        feats = self.image_encoder(images)                # (B, 1024)
        feats = self.image_proj(feats)                    # (B, 512)
        return F.normalize(feats, dim=-1)

    def encode_text(self, captions):
        tok = self.tokenizer(list(captions), padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(self.device)
        out = self.text_encoder(**tok)
        pooled = masked_mean_pool(out.last_hidden_state, tok["attention_mask"])  # (B, 384)
        feats = self.text_proj(pooled)                    # (B, 512)
        return F.normalize(feats, dim=-1)

    def forward(self, images, captions):
        return self.encode_image(images), self.encode_text(captions)


if __name__ == "__main__":
    # smoke test (run on your machine): confirm dims + pooling are sane
    m = MCi0MiniLM(shared_dim=512, device="cpu").to("cpu").eval()
    print("image_encoder out dim expected 1024; text 384; shared 512")
    import torch
    dummy_img = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        ie = m.image_encoder(dummy_img)
        print("MCi0 raw image feat:", ie.shape, "(expect (2,1024))")
        img_emb = m.encode_image(dummy_img)
        txt_emb = m.encode_text(["a photo of a cat", "a dog running"])
        print("projected image:", img_emb.shape, "(expect (2,512))")
        print("projected text :", txt_emb.shape, "(expect (2,512))")
        print("image L2 norms:", img_emb.norm(dim=-1))
        print("text  L2 norms:", txt_emb.norm(dim=-1))
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"total trainable params: {n_train:,}")