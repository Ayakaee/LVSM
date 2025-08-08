import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception_models.core.vision_encoder.pe import VisionTransformer

class PEEncoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_features(
        self,
        x: torch.Tensor,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False,
        repa_type: str = None,
        repa_label: dict = None
    ):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                dim=1,
            )

        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)

        if self.use_rope2d:
            self.rope.update_grid(x.device, grid_h, grid_w)

        x = self.ln_pre(x)

        stop_idx = layer_idx

        for i, r in enumerate(self.transformer.resblocks):
            x = r(x, attn_mask=None)
            if repa_type is not None:
                if i + 1 in repa_label.keys():
                    repa_label[i + 1] = x[:, 1:, :]
            if i + 1 == stop_idx:
                break

        if norm:
            x = self.ln_post(x)

        if strip_cls_token and self.use_cls_token:
            x = x[:, 1:, :]

        return x

