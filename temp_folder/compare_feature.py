import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import numpy as np

class FeatureVisualizationOrganizer:
    def __init__(self, input_dir="feature_visualizations", output_dir="organized_visualizations"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def organize_layer_features(self, view):
        """将不同层的特征可视化放在一起"""
        
        # 定义需要组织的层类型和文件模式
        layer_patterns = {
            'input_self_attn_intensity': rf'input_self_attn_(\d+)-view{view}_intensity_heatmap\.png',
            'input_self_attn_feature': rf'input_self_attn_(\d+)-view{view}_feature_analysis\.png',
            'input_self_attn_pca_all': rf'input_self_attn_(\d+)-view{view}_pca_all_patches\.png',
            'input_self_attn_pca_fg': rf'input_self_attn_(\d+)-view{view}_pca_foreground\.png',
            'self_cross_intensity': rf'self_cross_(\d+)-view{view}_intensity_heatmap\.png',
            'self_cross_feature': rf'self_cross_(\d+)-view{view}_feature_analysis\.png',
            'self_cross_pca_all': rf'self_cross_(\d+)-view{view}_pca_all_patches\.png',
            'self_cross_pca_fg': rf'self_cross_(\d+)-view{view}_pca_foreground\.png',
            'extra_enc_intensity': rf'extra_enc_(\d+)-view{view}_intensity_heatmap\.png',
            'extra_enc_feature': rf'extra_enc_(\d+)-view{view}_feature_analysis\.png',
            'extra_enc_pca_all': rf'extra_enc_(\d+)-view{view}_pca_all_patches\.png',
            'extra_enc_pca_fg': rf'extra_enc_(\d+)-view{view}_pca_foreground\.png',
            'transformer_layer_intensity': rf'transformer_layer_(\d+)-view{view}_intensity_heatmap\.png',
            'transformer_layer_feature': rf'transformer_layer_(\d+)-view{view}_feature_analysis\.png',
            'transformer_layer_pca_all': rf'transformer_layer_(\d+)-view{view}_pca_all_patches\.png',
            'transformer_layer_pca_fg': rf'transformer_layer_(\d+)-view{view}_pca_foreground\.png',
        }
        
        # 获取所有文件
        files = os.listdir(self.input_dir)
        
        for layer_type, pattern in layer_patterns.items():
            print(f"Processing {layer_type}...")
            
            # 找到匹配的文件并按层号排序
            matched_files = []
            for file in files:
                match = re.match(pattern, file)
                if match:
                    layer_num = int(match.group(1))
                    matched_files.append((layer_num, file))
            
            if not matched_files:
                continue
                
            # 按层号排序
            matched_files.sort(key=lambda x: x[0])
            
            # 创建组合图像
            self._create_layer_comparison(matched_files, layer_type, view=view)
    
    def organize_batch_views(self):
        """将同一batch不同视角的input和target放在一起"""
        
        files = os.listdir(self.input_dir)
        
        # 提取batch信息
        batch_data = defaultdict(lambda: {'input': [], 'target': []})
        
        for file in files:
            # 匹配input文件
            input_match = re.match(r'input-batch(\d+)-(\d+)\.png', file)
            if input_match:
                batch_num = int(input_match.group(1))
                view_num = int(input_match.group(2))
                batch_data[batch_num]['input'].append((view_num, file))
            
            # 匹配target文件
            target_match = re.match(r'target-batch(\d+)-(\d+)\.png', file)
            if target_match:
                batch_num = int(target_match.group(1))
                view_num = int(target_match.group(2))
                batch_data[batch_num]['target'].append((view_num, file))
        
        # 为每个batch创建组合图像
        for batch_num in sorted(batch_data.keys()):
            print(f"Processing batch {batch_num}...")
            self._create_batch_comparison(batch_data[batch_num], batch_num)
    
    def _create_layer_comparison(self, matched_files, layer_type, view):
        """创建层对比图像"""
        if not matched_files:
            return
        
        n_layers = len(matched_files)
        if n_layers <= 3:
            w, h = n_layers, 1
        elif n_layers % 2 == 0:
            w, h = n_layers // 2, 2
        else:
            w, h = n_layers // 2 + 1, 2
        fig_width = min(20, 4 * w)
        fig_height = 5 * h
        
        fig, axes = plt.subplots(h, w, figsize=(fig_width, fig_height))
        if n_layers == 1:
            axes = [[axes]]
        if h == 1:
            axes = [axes]
        
        fig.suptitle(f'{layer_type.replace("_", " ").title()} Across Layers', fontsize=16, y=0.95)
        
        for i, (layer_num, filename) in enumerate(matched_files):
            img_path = os.path.join(self.input_dir, filename)
            try:
                img = Image.open(img_path)
                axes[i // w][i % w].imshow(img)
                axes[i // w][i % w].set_title(f'Layer {layer_num}', fontsize=12)
                axes[i // w][i % w].axis('off')
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                axes[i // w][i % w].text(0.5, 0.5, f'Error\nLayer {layer_num}', 
                           ha='center', va='center', transform=axes[i // w][i % w].transAxes)
                axes[i // w][i % w].axis('off')
        
        # 调整布局并保存
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{layer_type}-view{view}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def _create_batch_comparison(self, batch_info, batch_num):
        """创建批次对比图像"""
        input_files = sorted(batch_info['input'], key=lambda x: x[0])
        target_files = sorted(batch_info['target'], key=lambda x: x[0])
        
        if not input_files and not target_files:
            return
        
        n_input = len(input_files)
        n_target = len(target_files)
        total_images = n_input + n_target
        
        if total_images == 0:
            return
        
        # 计算网格布局
        n_cols = 3  # 最多5列
        n_rows = 2
        
        fig_width = 4 * n_cols
        fig_height = 4 * n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        fig.suptitle(f'Batch {batch_num} - Input and Target Views', fontsize=16, y=0.95)
        
        img_idx = 0
        
        # 添加input图像
        for view_num, filename in input_files:
            row = 0
            col = img_idx
            
            img_path = os.path.join(self.input_dir, filename)
            try:
                img = Image.open(img_path)
                axes[row][col].imshow(img)
                axes[row][col].set_title(f'Input View {view_num}', fontsize=12)
                axes[row][col].axis('off')
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                axes[row][col].text(0.5, 0.5, f'Error\nInput {view_num}', 
                                  ha='center', va='center', transform=axes[row][col].transAxes)
                axes[row][col].axis('off')
            
            img_idx += 1
        
        # 添加target图像
        for view_num, filename in target_files:
            row = 1
            col = img_idx - 2
            
            img_path = os.path.join(self.input_dir, filename)
            try:
                img = Image.open(img_path)
                axes[row][col].imshow(img)
                axes[row][col].set_title(f'Target View {view_num}', fontsize=12)
                axes[row][col].axis('off')
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                axes[row][col].text(0.5, 0.5, f'Error\nTarget {view_num}', 
                                  ha='center', va='center', transform=axes[row][col].transAxes)
                axes[row][col].axis('off')
            
            img_idx += 1
        
        # 隐藏空的子图
        for i in range(img_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row][col].axis('off')
        
        # 调整布局并保存
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'batch_{batch_num:02d}_views.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def create_summary_report(self):
        """创建总结报告"""
        files = os.listdir(self.input_dir)
        
        report = []
        report.append("Feature Visualization Organization Summary")
        report.append("=" * 50)
        report.append("")
        
        # 统计不同类型的文件
        patterns = {
            'Input Self Attention': r'input_self_attn_\d+-view0',
            'Self Cross Attention': r'self_cross_\d+-view0',
            'Extra Encoder': r'extra_enc_\d+-view0',
            'Input Images': r'input-batch\d+-\d+',
            'Target Images': r'target-batch\d+-\d+'
        }
        
        for category, pattern in patterns.items():
            count = len([f for f in files if re.search(pattern, f)])
            report.append(f"{category}: {count} files")
        
        report.append("")
        report.append("Generated organized visualizations in: " + self.output_dir)
        
        # 保存报告
        with open(os.path.join(self.output_dir, 'organization_report.txt'), 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
    
    def run_all(self):
        """运行所有组织任务"""
        print("Starting feature visualization organization...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        # 组织层特征
        print("\n1. Organizing layer features...")
        for view in range(0,9):
            self.organize_layer_features(view=view)
        
        # 组织批次视图
        print("\n2. Organizing batch views...")
        self.organize_batch_views()
        
        # 创建总结报告
        print("\n3. Creating summary report...")
        self.create_summary_report()
        
        print("\n✅ Organization complete!")

if __name__ == "__main__":
    # 使用示例
    organizer = FeatureVisualizationOrganizer(
        input_dir="feature_visualizations/8.1-baseline",
        output_dir="feature_visualizations-o/8.1-baseline"
    )
    
    organizer.run_all()