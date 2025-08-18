import torch
import sys
sys.path.append('dinov3')
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dinov3.hub.backbones import _make_dinov3_vit
# model = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=True)
# model = _make_dinov3_vit(pretrained=True)
model = torch.hub.load(
    repo_or_dir='dinov3',
    model='dinov3_vitb16',
    source='local',
    pretrained=False
)

state_dict = torch.load(f"pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", map_location="cpu")
model.load_state_dict(state_dict)

model.to("cuda")
model.eval()