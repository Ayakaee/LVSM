import importlib
import os
import torch
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
model = LVSM(config, logger).to(ddp_info.device)

# 打印模型的所有参数名称和索引
print("=" * 80)
print("所有模型参数:")
print("=" * 80)
for idx, (name, param) in enumerate(model.named_parameters()):
    print(f"Index {idx:3d}: {name} - requires_grad: {param.requires_grad}")

# 启用 find_unused_parameters=True 来检测未使用的参数
model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)

# 创建一个简单的测试数据
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
data = next(iter(dataloader))
batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}

print("\n" + "=" * 80)
print("开始前向传播测试...")
print("=" * 80)

try:
    # 执行前向传播
    input, target = model.module.process_data(batch, has_target_image=True, target_has_input=config.training.target_has_input, compute_rays=True)
    ret_dict = model(batch, input, target)
    
    # 计算损失
    loss = ret_dict.loss_metrics.loss
    print(f"Loss computed successfully: {loss.item()}")
    
    # 反向传播
    loss.backward()
    print("反向传播完成")
    
    # 检查哪些参数没有梯度
    print("\n" + "=" * 80)
    print("检查参数梯度:")
    print("=" * 80)
    
    unused_params = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            if param.grad is None:
                unused_params.append((idx, name))
                print(f"❌ Index {idx:3d}: {name} - 没有梯度")
            else:
                grad_norm = param.grad.norm().item()
                print(f"✅ Index {idx:3d}: {name} - 梯度范数: {grad_norm:.6f}")
    
    if unused_params:
        print(f"\n❌ 发现 {len(unused_params)} 个未使用的参数:")
        for idx, name in unused_params:
            print(f"  Index {idx}: {name}")
    else:
        print("\n✅ 所有参数都有梯度")
        
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

dist.destroy_process_group() 