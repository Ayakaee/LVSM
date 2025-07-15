import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_logging
from utils.metric_utils import export_results, summarize_evaluation

# Load config and read(override) arguments from CLI
config = init_config()
logger = init_logging('logs/debug.log')
ddp_info = init_distributed(seed=777)
print(config)
model_path = config.inference.checkpoint_dir.replace("evaluation", "checkpoints")
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, logger).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
model.module.load_ckpt(model_path)

# 在 inference.py 中，模型加载后立即添加
print(f"Model loaded from: {model_path}")
print(f"Model state dict keys count: {len(model.state_dict())}")
print(f"Model state dict keys: {model.state_dict().keys()}")

# 检查几个关键层的权重
print(f"norm: {model.module.self_attn_blocks[21].mlp.mlp[0].weight.norm()}")
# print(f"Image decoder weight norm: {model.image_token_decoder[1].weight.norm()}")