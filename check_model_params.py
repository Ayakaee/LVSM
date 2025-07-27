import importlib
import os
import torch
from setup import init_config, init_logging
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from setup import init_config, init_distributed, init_logging

# 设置环境变量来启用详细的分布式调试信息
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# Load config and read(override) arguments from CLI
config = init_config()
logger = init_logging('logs/debug_unused_params.log')
ddp_info = init_distributed(seed=777)
# 导入模型
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, logger)

print("=" * 80)
print("模型参数分析")
print("=" * 80)

# 统计参数
total_params = 0
trainable_params = 0
frozen_params = 0

print("\n所有参数列表:")
print("-" * 80)
for idx, (name, param) in enumerate(model.named_parameters()):
    param_count = param.numel()
    total_params += param_count
    
    if param.requires_grad:
        trainable_params += param_count
        status = "✅ 可训练"
    else:
        frozen_params += param_count
        status = "❌ 冻结"
    
    print(f"Index {idx:3d}: {name}")
    print(f"       形状: {list(param.shape)}, 参数数量: {param_count:,}, 状态: {status}")
    print()

print("=" * 80)
print("参数统计:")
print(f"总参数数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
print(f"冻结参数: {frozen_params:,}")
print(f"可训练比例: {trainable_params/total_params*100:.2f}%")

# 按模块分组统计
print("\n" + "=" * 80)
print("按模块分组统计:")
print("-" * 80)

module_stats = {}
for name, param in model.named_parameters():
    module_name = name.split('.')[0]  # 取第一个点之前的部分作为模块名
    if module_name not in module_stats:
        module_stats[module_name] = {'trainable': 0, 'frozen': 0}
    
    if param.requires_grad:
        module_stats[module_name]['trainable'] += param.numel()
    else:
        module_stats[module_name]['frozen'] += param.numel()

for module_name, stats in module_stats.items():
    total = stats['trainable'] + stats['frozen']
    trainable_ratio = stats['trainable'] / total * 100 if total > 0 else 0
    print(f"{module_name:20s}: 可训练 {stats['trainable']:>10,}, 冻结 {stats['frozen']:>10,}, 可训练比例 {trainable_ratio:>6.2f}%")

# 检查特定的可能问题模块
print("\n" + "=" * 80)
print("检查可能的问题模块:")
print("-" * 80)

# 检查 image_encoder 相关参数
image_encoder_params = []
for name, param in model.named_parameters():
    if 'image_encoder' in name:
        image_encoder_params.append((name, param.requires_grad, param.numel()))

if image_encoder_params:
    print("Image Encoder 参数:")
    for name, requires_grad, numel in image_encoder_params:
        status = "可训练" if requires_grad else "冻结"
        print(f"  {name}: {status}, {numel:,} 参数")
else:
    print("没有找到 image_encoder 相关参数")

# 检查 repa 相关参数
repa_params = []
for name, param in model.named_parameters():
    if 'repa' in name:
        repa_params.append((name, param.requires_grad, param.numel()))

if repa_params:
    print("\nREPA 相关参数:")
    for name, requires_grad, numel in repa_params:
        status = "可训练" if requires_grad else "冻结"
        print(f"  {name}: {status}, {numel:,} 参数")
else:
    print("\n没有找到 REPA 相关参数")

print("\n" + "=" * 80) 