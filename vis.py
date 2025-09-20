# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_logging
from utils.metric_utils import export_results, summarize_evaluation
import argparse
import numpy as np
from get_attn_map import get_attention_weights_external, visualize_attention_weights
from torchvision import transforms

def visualize_attention_maps(model, images, layer_idx=-1):
    """
    可视化Vision Transformer的注意力图
    """
    model.eval()
    attention_maps = []
    
    # 注册hook来获取注意力权重
    def hook_fn(module, input, output):
        attention_maps.append(output)  # attention weights
    
    # 注册hook到指定层
    handle = model.blocks[layer_idx].attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(images)
    
    handle.remove()
    
    return attention_maps[0]

def create_attention_visualization(image, attention_map, model_name):
    # 将注意力图调整到图像尺寸
    h, w = image.shape[:2]
    attention_resized = cv2.resize(attention_map, (w, h))
    
    # 创建热力图
    heatmap = plt.cm.plasma(attention_resized)
    
    return heatmap


# 添加命令行参数解析
parser = argparse.ArgumentParser(description='LVSM模型推理')
parser.add_argument('--extract_features', action='store_true', help='是否提取模型每一层的特征')
parser.add_argument('--feature_save_dir', type=str, default='extracted_features', help='特征保存目录')
args, _ = parser.parse_known_args()

def save_image(image, name, path):
    for batch_id, images in enumerate(image):
        tensors = images.detach().cpu()
        for idx, tensor in enumerate(tensors):
            to_pil = transforms.ToPILImage()
            pil_img = to_pil(tensor)
            pil_img.save(os.path.join(path, f'{name}-batch{batch_id}-{idx}.png'))
    
# Load config and read(override) arguments from CLI
config = init_config()
config.training.num_views = config.training.num_input_views + config.training.num_target_views
log_file = config.training.get("log_file", f'logs/{config.inference.checkpoint_dir.split("/")[-1]}_eval.log')
logger = init_logging(log_file)
config.inference.feature_save_dir = os.path.join(config.inference.feature_save_dir, config.inference.checkpoint_dir.split('/')[2])

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}


# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

dataset = torch.utils.data.Subset(dataset, range(32))

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler
)
dataloader_iter = iter(dataloader)

dist.barrier()

# Import model and load checkpoint
model_path = config.inference.checkpoint_dir.replace("evaluation", "checkpoints")
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, logger).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
model.module.load_ckpt(model_path)

if ddp_info.is_main_process:  
    print(f"Running inference; save results to: {config.inference.checkpoint_dir}")
    if config.inference.extract_features:
        print(f"特征提取已启用，特征将保存到: {config.inference.feature_save_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


# datasampler.set_epoch(0)
model.eval()

# 如果启用特征提取，创建保存目录
if config.inference.extract_features and ddp_info.is_main_process:
    os.makedirs(config.inference.feature_save_dir, exist_ok=True)
with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):  
    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        input, target = model.module.process_data(batch, has_target_image=True, target_has_input = config.training.target_has_input, compute_rays=True)
        
        save_image(input.image, 'input', config.inference.feature_save_dir)
        save_image(target.image, 'target', config.inference.feature_save_dir)
        
        if config.inference.extract_features:
            result = model(batch, input, target, train=False, extract_features=True)
            
            # 保存特征到文件
            if ddp_info.is_main_process and result.layer_features is not None:
                for layer_name, features in result.layer_features.items():
                    if features is not None:
                        # 保存特征到numpy文件
                        feature_path = os.path.join(config.inference.feature_save_dir, f'{layer_name}.npy')
                        # 修复 BFloat16 类型问题：先转换为 float32，再转换为 numpy
                        features = features.view(config.training.batch_size_per_gpu, -1, 1024, 768)
                        if features.dtype == torch.bfloat16:
                            np_features = features.float().cpu().numpy()
                        else:
                            np_features = features.cpu().numpy()
                        np.save(feature_path, np_features)
                        print(f"保存特征 {layer_name} 到 {feature_path}, 形状: {np_features.shape}")
        else:
            result = model(batch, input, target, train=False)
        # result, attention_weights = get_attention_weights_external(model, batch, input, target, layer_idx=5, block_type='self_cross')
        if config.inference.get("render_video", False):
            result= model.module.render_video(result, **config.inference.render_video_config)
        export_results(result, config.inference.checkpoint_dir, compute_metrics=config.inference.get("compute_metrics"))
        # 清理GPU内存
        if config.inference.extract_features:
            del result.layer_features
        torch.cuda.empty_cache()
        
        if ddp_info.is_main_process:
            print(f"处理批次 {batch_idx + 1} 完成")
        break
# print(attention_weights.shape)
# 把attn weight保存到文件
# np.save('attn_weights.npy', attention_weights.cpu().numpy())

# 可视化注意力图
# visualize_attention_weights(attention_weights, 'input_self_attn_blocks[5].attn', 'attn_weights', save_heatmap=True, save_attention_pattern=True)

dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference.checkpoint_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference.checkpoint_dir}")

dist.barrier()
dist.destroy_process_group()
exit(0)