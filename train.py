# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup, init_logging, init_file_logging
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0
from model.encoder import load_encoders, preprocess_raw_image
from einops import rearrange, repeat
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from datetime import datetime

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

log_file = config.training.get("log_file", f'logs/{config.training.wandb_exp_name}')
logger = init_logging(log_file)
file_only_logger = init_file_logging(log_file)

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up wandb and backup source code
if ddp_info.is_main_process:
    init_wandb_and_backup(config)
dist.barrier()

total_num_epochs = config.training.train_epochs
grad_accum_steps = config.training.grad_accum_steps
batch_size_per_gpu = config.training.batch_size_per_gpu
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_train_steps = int(total_num_epochs * config.training.dataset_len // (total_batch_size))
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
save_every_steps = int(config.training.dataset_len // (total_batch_size) * config.training.checkpoint_every_epoch) + 1

# 记录训练开始时间
training_start_time = datetime.now()
if ddp_info.is_main_process:
    logger.info(f"\n{'='*60}")
    logger.info(f"Training started at: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Training configuration:")
    logger.info(f"  - Total steps: {total_train_steps}")
    logger.info(f"  - Batch size per GPU: {config.training.batch_size_per_gpu}")
    logger.info(f"  - Save every steps: {save_every_steps}")
    logger.info(f"  - World size: {ddp_info.world_size}")
    logger.info(f"  - Learning rate: {config.training.lr}")
    logger.info(f"  - Use compile: {config.training.use_compile}")
    logger.info(f"  - Enable REPA: {config.training.enable_repa}")
    logger.info(f"{'='*60}\n")

# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)



module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, logger).to(ddp_info.device)
if config.training.use_compile:
    model = torch.compile(model)
model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=False)
if config.training.enable_repa:
    encoders, encoder_types, architectures = load_encoders(config.model.encoder_type, ddp_info.device, 256)

optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())

scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)

if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
)

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

start_train_step = cur_train_step
model.train()
print('len:', len(dataset))
# 记录实际训练开始时间（跳过resume的部分）
actual_training_start_time = datetime.now()
if ddp_info.is_main_process:
    logger.info(f"Actual training started at: {actual_training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Starting from step: {cur_train_step}")
    logger.info(f"{'='*60}\n")

while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset) )
    try:
        data = next(dataloader_iter)
    except StopIteration:
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}
    data_time = time.time() - tic
    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        input, target = model.module.process_data(batch, has_target_image=True, target_has_input = config.training.target_has_input, compute_rays=True)
        zs_label = []
        if config.training.enable_repa:
            for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                raw_image_ = rearrange(target.image, 'b v c h w -> (b v) c h w')
                raw_image_ = preprocess_raw_image(raw_image_, encoder_type)
                with torch.no_grad():
                    z = encoder.forward_features(raw_image_)
                    if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                    if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                    if 'PE-Core' in config.model.encoder_type: z = z[:, 1:, :]
                zs_label.append(z)

        ret_dict = model(batch, zs_label, input, target)

    update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
    if update_grads:
        with model.no_sync(): # no sync grads for efficiency
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    else:
        scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    cur_train_step += 1

    export_inter_results = ((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.vis_every == 0)

    skip_optimizer_step = False
    # Skip optimizer step if loss is NaN or Inf
    if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
        print(f"NaN or Inf loss detected, skip this iteration")
        skip_optimizer_step = True
        ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

    total_grad_norm = None
    # Check gradient norm and update optimizer if everything is fine
    if update_grads and (not skip_optimizer_step):
        # Unscales the gradients
        scaler.unscale_(optimizer) 
        # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
        with torch.no_grad():
            for n, p in optimized_param_dict.items():
                if p.requires_grad and (p.grad is not None):
                    p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
    
        # visualize the grad norm of each layer of our transformer (FOR DEBUG)
        if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
            grad_norms = {}  # Dictionary to store norms per layer
            for name, param in model.named_parameters():
                if param.grad is not None:  # Some parameters might not have gradients
                    grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
            for layer_name, grad_norm in grad_norms.items():
                wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

        total_grad_norm = 0.0
        if config.training.grad_clip_norm > 0:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

            if total_grad_norm > config.training.grad_clip_norm * 2.0:
                print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

            allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
            if total_grad_norm > allowed_gradnorm:
                skip_optimizer_step = True
                print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

            # show grad norm in wandb if it's too large
            display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
            if display_grad_norm and ddp_info.is_main_process:
                wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)

        if not skip_optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            cur_param_update_step += 1

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # log and save checkpoint
    if ddp_info.is_main_process:
        loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Total samples: {cur_train_step * total_batch_size} | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | Data Time: {data_time:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            # Add loss values
            for k, v in loss_dict.items():
                if 'lpips' in k:
                    continue
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        # log in wandb
        if (cur_train_step % config.training.wandb_log_every == 0) or (
            cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "total_samples": cur_train_step * total_batch_size,
                "iter": cur_train_step,
                "Data Time": data_time,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            file_only_logger.info(log_dict)
            
            wandb.log(
                log_dict,
                step=cur_train_step * total_batch_size,
            )

        # save checkpoint
        if (cur_train_step % save_every_steps == 0) or (cur_train_step == total_train_steps):
            if isinstance(model, DDP):
                model_weights = model.module.state_dict()
            else:
                model_weights = model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_epoch * 10:03d}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at epoch*10 {cur_epoch * 10} to {os.path.abspath(ckpt_path)}")
        
        # export intermediate visualization results
        if export_inter_results:
            vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
            os.makedirs(vis_path, exist_ok=True)
            # visualize_intermediate_results(vis_path, ret_dict)
            torch.cuda.empty_cache()
            model.train()

            
    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()

# 训练结束，输出时间统计
training_end_time = datetime.now()
training_duration = training_end_time - training_start_time
actual_training_duration = training_end_time - actual_training_start_time

if ddp_info.is_main_process:
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed successfully!")
    logger.info(f"Training ended at: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {training_duration}")
    logger.info(f"Actual training duration: {actual_training_duration}")
    logger.info(f"Final step: {cur_train_step}")
    logger.info(f"Final param update step: {cur_param_update_step}")
    logger.info(f"{'='*60}\n")

dist.barrier()
dist.destroy_process_group()
