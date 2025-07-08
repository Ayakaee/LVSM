from model.encoder import load_encoders, preprocess_raw_image
from einops import rearrange
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

encoders, encoder_types, architectures = load_encoders('dinov2-vit-b', 'cuda', 256)
img_path = 're10k/train/images/21e794f71e31becb/00000.png'
raw_image_ = Image.open(img_path).convert('RGB')  # 保证是3通道

# 新增：先resize到256x256
resize = transforms.Resize((256, 256))
raw_image_ = resize(raw_image_)

to_tensor = transforms.ToTensor()
raw_image_ = to_tensor(raw_image_)  # [C, H, W]
raw_image_ = raw_image_.unsqueeze(0)  # [1, C, H, W]
raw_image_ = raw_image_.to('cuda')
zs_label = []
for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
    # raw_image_ = rearrange(target.image, 'b v c h w -> (b v) c h w')
    raw_image_ = preprocess_raw_image(raw_image_, encoder_type)
    with torch.no_grad():
        z = encoder.forward_features(raw_image_)
        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
    zs_label.append(z)
print(zs_label[0].shape)


import matplotlib.pyplot as plt
import numpy as np

# 假设 z 的 shape 是 [batch, num_patches, feature_dim]
latent = z[0].cpu().numpy()  # shape: [num_patches, feature_dim]

# plt.figure(figsize=(10, 6))
# plt.imshow(latent, aspect='auto', cmap='hot')
# plt.colorbar()
# plt.xlabel('Feature Dimension')
# plt.ylabel('Patch Index')
# plt.title('DINOv2 Encoder Latent Heatmap')
# plt.savefig('vis.png')

# latent_map = latent[:, 0].reshape(16, 16)  # 取第0个feature，reshape成16x16
# plt.imshow(latent_map, cmap='hot')
# plt.colorbar()
# plt.title('Feature 0 Spatial Heatmap')
# plt.savefig('feature0_spatial.png')

# latent_mean = latent.mean(axis=1).reshape(16, 16)  # 对feature维度求均值
# plt.imshow(latent_mean, cmap='hot')
# plt.colorbar()
# plt.title('Mean Feature Spatial Heatmap')
# plt.savefig('mean_feature_spatial.png')

# feature_map = latent  # latent 已经是 numpy 数组，shape [256, 768]
# gray_scale = np.sum(feature_map, axis=1).reshape(16, 16)  # 对 feature 维度求和，axis=1
# gray_scale = gray_scale / feature_map.shape[1]  # 除以特征数，做归一化
# # print(gray_scale)
# plt.imshow(gray_scale, cmap='gray')
# plt.colorbar()
# plt.title('Gray-scale Feature Map')
# plt.savefig('gray_scale_feature_map.png')

# features = zs_label[0]
# # 计算所有通道在空间位置上的均值
# heatmap = features.mean(dim=2).squeeze()  # 形状变为[256]

# # 重塑为16x16的网格
# heatmap = heatmap.reshape(16, 16).cpu().numpy()

# # 可视化
# plt.figure(figsize=(10, 10))
# plt.imshow(heatmap, cmap='viridis')  # 使用与示例图相似的蓝绿色调
# plt.colorbar()
# plt.title("DINOv2 Feature Activation (Channel Mean)")
# plt.axis('off')
# plt.savefig('vis.png')

from sklearn.decomposition import PCA
features = zs_label[0]
# 展平特征并应用PCA
flatten_features = features.squeeze().cpu().numpy()
pca = PCA(n_components=3)
pca_features = pca.fit_transform(flatten_features)

# 归一化到[0,1]范围
normalized = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
rgb_map = normalized.reshape(16, 16, 3)
plt.figure(figsize=(6, 6))
plt.imshow(rgb_map)
plt.axis('off')
plt.title('DINOv2 Patch Feature PCA RGB Map')
plt.savefig('pca_rgb_map.png')
# plt.show()