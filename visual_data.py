import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np

# 读取CSV数据
def read_csv_data(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            data.append([float(x) for x in row])
    return header, data

# 按input分组数据
def group_by_input(data):
    groups = defaultdict(list)
    for row in data:
        input_val = int(row[0])
        groups[input_val].append(row)
    return groups

# 按target分组数据
def group_by_target(data):
    groups = defaultdict(list)
    for row in data:
        target_val = int(row[1])
        groups[target_val].append(row)
    return groups

# 读取三个模型的数据
model_files = ['elvsm.csv', '../lvsm-ori/lvsm.csv', '../lvsm-ori/lvsm-ed.csv']  # 请替换为实际文件名
model_files = ['elvsm-train.csv', '../lvsm-ori/lvsm-train.csv', '../lvsm-ori/lvsm-ed-train.csv']  # 请替换为实际文件名
model_names = ['Efficient-LVSM', 'LVSM Decoder-Only', 'LVSM Encoder-Decoder']  # 模型名称
model_data = {}

# 加载所有模型数据
for i, filename in enumerate(model_files):
    try:
        header, data = read_csv_data(filename)
        model_data[model_names[i]] = {
            'data': data,
            'grouped_by_input': group_by_input(data),
            'grouped_by_target': group_by_target(data)
        }
        print(f"成功加载 {model_names[i]}: {len(data)} 个数据点")
    except FileNotFoundError:
        print(f"警告: 文件 {filename} 未找到，跳过模型 {model_names[i]}")
        continue
print(model_data['LVSM Decoder-Only']['grouped_by_input'][4])
# 如果只有一个模型的数据，复制数据来演示多模型对比
if len(model_data) == 1:
    print("只有一个模型数据，创建模拟数据用于演示...")
    base_model = list(model_data.keys())[0]
    base_data = model_data[base_model]['data']
    
    # 创建模拟的第二个模型（稍高的内存和时间）
    model2_data = [[row[0], row[1], row[2]*1.1, row[3]*1.2, row[4]*1.3] for row in base_data]
    model_data['Model 2'] = {
        'data': model2_data,
        'grouped_by_input': group_by_input(model2_data),
        'grouped_by_target': group_by_target(model2_data)
    }
    
    # 创建模拟的第三个模型（更高的内存和时间）
    model3_data = [[row[0], row[1], row[2]*0.9, row[3]*1.5, row[4]*1.1] for row in base_data]
    model_data['Model 3'] = {
        'data': model3_data,
        'grouped_by_input': group_by_input(model3_data),
        'grouped_by_target': group_by_target(model3_data)
    }

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Multi-Model Performance Comparison', fontsize=18, fontweight='bold')

# 颜色和样式设置
model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
model_markers = ['o', 's', '^']
model_linestyles = ['-', '--', '-.']

# 分析参数
selected_input = 4
selected_target = 8

# 1. Memory vs Target 对比 (选择input=1)
ax1 = axes[0, 0]
for i, (model_name, model_info) in enumerate(model_data.items()):
    if selected_input in model_info['grouped_by_input']:
        group_data = model_info['grouped_by_input'][selected_input]
        targets = [int(row[1]) for row in group_data]
        memory = [row[3] for row in group_data]
        print(targets, memory)
        
        ax1.plot(targets, memory, marker=model_markers[i], color=model_colors[i], 
                linewidth=3, markersize=8, label=model_name,
                linestyle=model_linestyles[i], alpha=0.8)

ax1.set_xlabel('Target', fontsize=12, fontweight='bold')
ax1.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
ax1.set_title(f'Memory Usage vs Target (Input={selected_input})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# 2. FLOPs vs Target 对比 (选择input=1)
ax2 = axes[0, 1]
for i, (model_name, model_info) in enumerate(model_data.items()):
    if selected_input in model_info['grouped_by_input']:
        group_data = model_info['grouped_by_input'][selected_input]
        targets = [int(row[1]) for row in group_data]
        flops = [row[2] for row in group_data]
        
        ax2.plot(targets, flops, marker=model_markers[i], color=model_colors[i], 
                linewidth=3, markersize=8, label=model_name,
                linestyle=model_linestyles[i], alpha=0.8)

ax2.set_xlabel('Target', fontsize=12, fontweight='bold')
ax2.set_ylabel('GFLOPs', fontsize=12, fontweight='bold')
ax2.set_title(f'GFLOPs vs Target (Input={selected_input})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# 3. FLOPs vs Target 对比 (选择input=1)
ax2 = axes[1, 1]
for i, (model_name, model_info) in enumerate(model_data.items()):
    if selected_input in model_info['grouped_by_input']:
        group_data = model_info['grouped_by_input'][selected_input]
        targets = [int(row[1]) for row in group_data]
        flops = [row[4] for row in group_data]
        
        ax2.plot(targets, flops, marker=model_markers[i], color=model_colors[i], 
                linewidth=3, markersize=8, label=model_name,
                linestyle=model_linestyles[i], alpha=0.8)

ax2.set_xlabel('Target', fontsize=12, fontweight='bold')
ax2.set_ylabel('GFLOPs', fontsize=12, fontweight='bold')
ax2.set_title(f'Time vs Target (Input={selected_input})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# 3. Memory vs Input 对比 (选择target=8)
# ax3 = axes[1, 0]
# bar_width = 0.25
# input_positions = None

# for i, (model_name, model_info) in enumerate(model_data.items()):
#     if selected_target in model_info['grouped_by_target']:
#         group_data = model_info['grouped_by_target'][selected_target]
#         inputs = [int(row[0]) for row in group_data]
#         memory = [row[3] for row in group_data]
        
#         if input_positions is None:
#             input_positions = np.array(inputs)
        
#         # 计算每个模型的柱状图位置偏移
#         positions = input_positions + (i - 1) * bar_width
#         bars = ax3.bar(positions, memory, bar_width, 
#                       color=model_colors[i], alpha=0.7, label=model_name)
        
#         # 在柱状图上标注数值
#         for bar, mem_val in zip(bars, memory):
#             height = bar.get_height()
#             ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
#                     f'{mem_val:.0f}', ha='center', va='bottom', fontsize=9, rotation=0)

# ax3.set_xlabel('Input', fontsize=12, fontweight='bold')
# ax3.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
# ax3.set_title(f'Memory Usage vs Input (Target={selected_target})', fontsize=14, fontweight='bold')
# ax3.grid(True, alpha=0.3, axis='y')
# ax3.set_xticks(input_positions)
# ax3.legend(fontsize=11)

# # 4. FLOPs vs Input 对比 (选择target=8)
# ax4 = axes[1, 1]
# for i, (model_name, model_info) in enumerate(model_data.items()):
#     if selected_target in model_info['grouped_by_target']:
#         group_data = model_info['grouped_by_target'][selected_target]
#         inputs = [int(row[0]) for row in group_data]
#         flops = [row[2] for row in group_data]
        
#         # 计算每个模型的柱状图位置偏移
#         positions = input_positions + (i - 1) * bar_width
#         bars = ax4.bar(positions, flops, bar_width, 
#                       color=model_colors[i], alpha=0.7, label=model_name)
        
#         # 在柱状图上标注数值
#         for bar, flop_val in zip(bars, flops):
#             height = bar.get_height()
#             ax4.text(bar.get_x() + bar.get_width()/2., height + 20,
#                     f'{flop_val:.0f}', ha='center', va='bottom', fontsize=9, rotation=0)

# ax4.set_xlabel('Input', fontsize=12, fontweight='bold')
# ax4.set_ylabel('GFLOPs', fontsize=12, fontweight='bold')
# ax4.set_title(f'GFLOPs vs Input (Target={selected_target})', fontsize=14, fontweight='bold')
# ax4.grid(True, alpha=0.3, axis='y')
# ax4.set_xticks(input_positions)
# ax4.legend(fontsize=11)

plt.tight_layout()

# 打印对比统计信息
print("=== 多模型对比统计 ===")
for model_name, model_info in model_data.items():
    data = model_info['data']
    print(f"\n{model_name}:")
    print(f"  总数据点数: {len(data)}")
    print(f"  Memory范围: {min(row[3] for row in data):.2f} - {max(row[3] for row in data):.2f} MB")
    print(f"  FLOPs范围: {min(row[2] for row in data):.2f} - {max(row[2] for row in data):.2f}")
    print(f"  Time范围: {min(row[4] for row in data):.2f} - {max(row[4] for row in data):.2f} ms")

# 性能对比分析
print(f"\n=== 性能对比分析 (Input={selected_input}, Target={selected_target}) ===")
comparison_data = {}

for model_name, model_info in model_data.items():
    # 获取特定input和target的数据点
    target_data = None
    if selected_input in model_info['grouped_by_input']:
        for row in model_info['grouped_by_input'][selected_input]:
            if int(row[1]) == selected_target:
                target_data = row
                break
    
    if target_data:
        comparison_data[model_name] = {
            'memory': target_data[3],
            'flops': target_data[2],
            'time': target_data[4]
        }

if len(comparison_data) > 1:
    print(f"在Input={selected_input}, Target={selected_target}条件下:")
    
    # 找出最佳性能的模型
    best_memory_model = min(comparison_data.keys(), key=lambda k: comparison_data[k]['memory'])
    best_flops_model = min(comparison_data.keys(), key=lambda k: comparison_data[k]['flops'])
    best_time_model = min(comparison_data.keys(), key=lambda k: comparison_data[k]['time'])
    
    print(f"最低Memory: {best_memory_model} ({comparison_data[best_memory_model]['memory']:.2f} MB)")
    print(f"最低FLOPs: {best_flops_model} ({comparison_data[best_flops_model]['flops']:.2f})")
    print(f"最短Time: {best_time_model} ({comparison_data[best_time_model]['time']:.2f} ms)")
    
    # 计算相对性能
    base_model = list(comparison_data.keys())[0]
    print(f"\n相对于{base_model}的性能对比:")
    for model_name, metrics in comparison_data.items():
        if model_name != base_model:
            memory_ratio = metrics['memory'] / comparison_data[base_model]['memory']
            flops_ratio = metrics['flops'] / comparison_data[base_model]['flops']
            time_ratio = metrics['time'] / comparison_data[base_model]['time']
            print(f"{model_name}:")
            print(f"  Memory: {memory_ratio:.2f}x ({'+' if memory_ratio > 1 else ''}{(memory_ratio-1)*100:.1f}%)")
            print(f"  FLOPs: {flops_ratio:.2f}x ({'+' if flops_ratio > 1 else ''}{(flops_ratio-1)*100:.1f}%)")
            print(f"  Time: {time_ratio:.2f}x ({'+' if time_ratio > 1 else ''}{(time_ratio-1)*100:.1f}%)")

plt.savefig('v-train.png', dpi=300, bbox_inches='tight')