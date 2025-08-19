# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from pathlib import Path
import os

def visualize_features_pca(feature_file, output_dir='feature_visualizations', layer_name=None, batch_id=0, view=0):
    """
    使用PCA可视化特征
    
    Args:
        feature_file: 特征文件路径
        output_dir: 输出目录
        layer_name: 层名称（用于文件名）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载特征数据
    features = np.load(feature_file)
    print(f"加载特征文件: {feature_file}")
    print(f"特征形状: {features.shape}")
    
    if layer_name is None:
        layer_name = Path(feature_file).stem
    
    # 选择第一个样本进行可视化
    x = features[batch_id][view]  # 形状: (1024, 768)
    layer_name = f'{layer_name}-view{view}'
    print(f"选择的样本形状: {x.shape}")
    
    # 计算patch网格尺寸
    # 假设1024个patches可以排列成正方形网格
    n_patches = x.shape[0]
    h_patches = int(np.sqrt(n_patches))
    w_patches = n_patches // h_patches
    
    print(f"假设的patch网格: {h_patches} x {w_patches}")
    
    # 方法1: 直接PCA可视化（所有patches）
    print("=== 方法1: 直接PCA可视化 ===")
    pca = PCA(n_components=3, whiten=True)
    pca.fit(x)
    
    # 应用PCA变换
    projected_image = torch.from_numpy(pca.transform(x)).view(h_patches, w_patches, 3)
    
    # 颜色增强
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
    
    # 可视化
    plt.figure(figsize=(12, 10), dpi=300)
    plt.imshow(projected_image.permute(1, 2, 0))
    plt.title(f'{layer_name} - PCA Visualization (All Patches)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{layer_name}_pca_all_patches.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 方法2: 基于特征强度选择前景patches
    print("=== 方法2: 基于特征强度选择前景patches ===")
    
    # 计算每个patch的特征强度（L2范数）
    feature_intensities = np.linalg.norm(x, axis=1)  # 形状: (1024,)
    
    # 选择特征强度较高的patches作为前景
    intensity_threshold = np.percentile(feature_intensities, 70)  # 选择前30%的patches
    foreground_selection = feature_intensities > intensity_threshold
    
    print(f"前景patches数量: {np.sum(foreground_selection)} / {len(foreground_selection)}")
    
    # 只对前景patches进行PCA
    fg_patches = x[foreground_selection]
    
    if len(fg_patches) > 3:  # 确保有足够的patches进行PCA
        pca_fg = PCA(n_components=3, whiten=True)
        pca_fg.fit(fg_patches)
        
        # 对所有patches应用PCA变换
        projected_all = torch.from_numpy(pca_fg.transform(x)).view(h_patches, w_patches, 3)
        projected_all = torch.nn.functional.sigmoid(projected_all.mul(2.0)).permute(2, 0, 1)
        
        # 创建前景掩码
        fg_mask = torch.zeros(h_patches, w_patches)
        fg_indices = np.where(foreground_selection)[0]
        for idx in fg_indices:
            h_idx = idx // w_patches
            w_idx = idx % w_patches
            if h_idx < h_patches and w_idx < w_patches:
                fg_mask[h_idx, w_idx] = 1
        
        # 应用前景掩码
        projected_all *= fg_mask.unsqueeze(0)
        
        # 可视化
        plt.figure(figsize=(12, 10), dpi=300)
        plt.imshow(projected_all.permute(1, 2, 0))
        plt.title(f'{layer_name} - PCA Visualization (Foreground Patches Only)')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{layer_name}_pca_foreground.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
    
    # 方法3: 特征强度热力图
    print("=== 方法3: 特征强度热力图 ===")
    intensity_map = feature_intensities.reshape(h_patches, w_patches)
    
    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(intensity_map, cmap='viridis')
    plt.colorbar(label='Feature Intensity (L2 Norm)')
    plt.title(f'{layer_name} - Feature Intensity Heatmap')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{layer_name}_intensity_heatmap.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 方法4: 特征分布分析
    print("=== 方法4: 特征分布分析 ===")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 特征强度分布
    axes[0, 0].hist(feature_intensities, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(intensity_threshold, color='red', linestyle='--', 
                      label=f'Threshold ({intensity_threshold:.3f})')
    axes[0, 0].set_title('Feature Intensity Distribution')
    axes[0, 0].set_xlabel('Feature Intensity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 特征值分布
    axes[0, 1].hist(x.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Feature Value Distribution')
    axes[0, 1].set_xlabel('Feature Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 每个patch的平均特征值
    mean_features = np.mean(x, axis=1)
    axes[1, 0].hist(mean_features, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Mean Feature Value per Patch')
    axes[1, 0].set_xlabel('Mean Feature Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 特征维度方差
    feature_variance = np.var(x, axis=0)
    axes[1, 1].hist(feature_variance, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Feature Dimension Variance')
    axes[1, 1].set_xlabel('Variance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{layer_name}_feature_analysis.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"可视化完成！结果保存在: {output_dir}")

def visualize_multiple_layers(feature_dir='extracted_features', output_dir='feature_visualizations', batch_id=0):
    """
    可视化多个层的特征
    
    Args:
        feature_dir: 特征文件目录
        output_dir: 输出目录
    """
    feature_files = list(Path(feature_dir).glob('*.npy'))
    
    print(f"找到 {len(feature_files)} 个特征文件")
    
    for feature_file in feature_files:
        layer_name = feature_file.stem
        print(f"\n处理层: {layer_name}")
        if 'cross' in layer_name:
            view = [0]
        else:
            view = [0]
        for v in view:
            visualize_features_pca(str(feature_file), output_dir, layer_name, batch_id=batch_id, view=v)

if __name__ == "__main__":
    # 可视化单个层的特征
    # print("=== 可视化单个层特征 ===")
    # visualize_features_pca('extracted_features/batch_0000_input_self_attn_5.npy', 
    #                       'feature_visualizations', 'input_self_attn_5')
    
    # 可视化多个层的特征
    print("\n=== 可视化多个层特征 ===")
    visualize_multiple_layers('extracted_features', 'feature_visualizations', batch_id=3) 