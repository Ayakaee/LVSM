from dinov2.models.vision_transformer import vit_base, DinoVisionTransformer
import torch
import torch.nn as nn
from functools import partial
from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

def make_dinov2_model(model_path):
    vit_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1
    )
    model = vit_base(**vit_kwargs)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model

class DinoEncoder(DinoVisionTransformer):
    def __init__(self, model_path):
        super().__init__(
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            block_fn=partial(Block, attn_class=MemEffAttention),
            num_register_tokens=0,
            img_size=518,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            interpolate_antialias=False,
            interpolate_offset=0.1
        )
        state_dict = torch.load(model_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)
        self.patch_embed = RGBP_PatchEmbed(img_size=518, patch_size=14, in_chans=3, embed_dim=768)

    def forward(self, x):
        return self.forward_features(x)

    def forward_features(self, x):
        return super().forward_features(x)

class RGBP_PatchEmbed(PatchEmbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.full((self.embed_dim,), 1e-4))
        self.plucker_proj = nn.Conv2d(6, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.plucker_norm = nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        rgb, p = x.split([3, 6], dim=1)
        rgb = self.proj(rgb) # B C H W
        H, W = rgb.size(2), rgb.size(3)
        rgb = rgb.flatten(2).transpose(1, 2)  # B HW C
        rgb = self.norm(rgb)

        p = self.plucker_proj(p)
        p = p.flatten(2).transpose(1, 2)  # B HW C
        p = self.plucker_norm(p)
        x = rgb + self.alpha * p
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
