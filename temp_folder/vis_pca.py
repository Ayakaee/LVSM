import numpy as np
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

# 读取fg_patches.npy文件
fg_patches = np.load('extracted_features/batch_0000_input_self_attn_5.npy')
print(f"原始数据形状: {fg_patches.shape}")

# 假设1024个patches可以排列成32x32的网格
# 如果1024 = 32 * 32，那么 h_patches = w_patches = 32
# 如果1024 = 16 * 64，那么 h_patches = 16, w_patches = 64
# 这里我们假设是正方形排列
h_patches = int(np.sqrt(fg_patches.shape[1]))  # 32
w_patches = fg_patches.shape[1] // h_patches   # 32

print(f"假设的patch网格: {h_patches} x {w_patches}")

# 选择第一个样本进行可视化
x = fg_patches[0]  # 形状: (1024, 768)
print(f"选择的样本形状: {x.shape}")

# 创建PCA模型
pca = PCA(n_components=3, whiten=True)
pca.fit(x)

# 应用PCA变换
projected_image = torch.from_numpy(pca.transform(x)).view(h_patches, w_patches, 3)
print(f"PCA变换后形状: {projected_image.shape}")

# 颜色增强
projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
print(f"颜色增强后形状: {projected_image.shape}")

# 创建简单的前景掩码（这里假设所有区域都是前景）
# 在实际应用中，fg_score_mf应该从其他地方获取
fg_score_mf = torch.ones(h_patches, w_patches)
projected_image *= (fg_score_mf.unsqueeze(0) > 0.5)

# 可视化
plt.figure(figsize=(10, 8), dpi=300)
plt.imshow(projected_image.permute(1, 2, 0))
plt.title('PCA Visualization of Self-Attention Features (Layer 5)')
plt.axis('off')
plt.savefig('pca_fg_patches.png', bbox_inches='tight', dpi=300)
plt.show()

print("PCA可视化完成！结果保存在: pca_fg_patches.png")