import sys
sys.path.append('dinov3')
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dinov3.hub.backbones import _make_dinov3_vit
# model = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=True)
model = _make_dinov3_vit(pretrained=True)