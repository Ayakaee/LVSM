import torch
import sys
sys.path.append('dinov3')
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dinov3.hub.backbones import _make_dinov3_vit
# model = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=True)
# model = _make_dinov3_vit(pretrained=True)
model = torch.hub.load(
    repo_or_dir=REPO_DIR,
    model=MODEL_NAME,
    source='local',
    pretrained=False
)

state_dict = torch.load(f"{WEIGHT_DIR}/{MODEL_WEIGHTS}", map_location="cpu")
model.load_state_dict(state_dict)

model.to("cuda")
model.eval()