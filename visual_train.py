import csv
import matplotlib.pyplot as plt

# 假设有两个CSV文件路径
csv_file1 = 'experiments/checkpoints/9.1-dinov3-8-2i/ametrics-eval.csv'
csv_file2 = 'experiments/checkpoints/8.1-baseline/ametrics-eval.csv'

def read_csv_data(filename):
    """从CSV读取数据并返回训练时间和PSNR列表"""
    hours = []
    psnr = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            hours.append(float(row[0]))  # 第一列是训练时间
            psnr.append(float(row[1]))  # 第二列是PSNR
    return hours, psnr

# 读取两个文件的数据
hours1, psnr1 = read_csv_data(csv_file1)
hours2, psnr2 = read_csv_data(csv_file2)

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(hours1, psnr1, 'o-', label='Efficient-LVSM', linewidth=2, markersize=8)
plt.plot(hours2, psnr2, 's--', label='LVSM Decoder-Only', linewidth=2, markersize=8)

# 添加标签和标题
plt.xlabel('Training Time (hours)', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.title('PSNR vs Training Time Comparison', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 优化显示范围
plt.xlim(min(hours1 + hours2) - 0.5, max(hours1 + hours2) + 0.5)
plt.ylim(min(psnr1 + psnr2) - 1, max(psnr1 + psnr2) + 1)

# 保存和显示
plt.tight_layout()
plt.savefig('v-psnr_vs_time.png', dpi=300)