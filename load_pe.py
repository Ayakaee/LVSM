import torch
from PIL import Image
import sys
sys.path.append('perception_models')
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

print("PE configs:", pe.VisionTransformer.available_configs())
# PE configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224', 'PE-Lang-G14-448', 'PE-Lang-L14-448', 'PE-Spatial-G14-448']

model = pe.VisionTransformer.from_config("PE-Core-L14-336", pretrained=True)  # Loads from HF
model = model.cuda()
preprocess = transforms.get_image_transform(448)
image_path = 're10k/train/images/21e794f71e31becb/00000.png'
image = preprocess(Image.open(image_path)).unsqueeze(0).cuda()

out = model.forward_features(image)  # pass layer_idx=<idx> to get a specific layer's output!
print(out.shape)
# torch.Size([1, 1025, 1024])

torch.save(out.cpu(), 'pe_feature.pt')