# LVSM Finetuning Guide

This guide explains how to finetune your LVSM model from 256 resolution to 512 resolution following the paper's training procedure.

## Training Process Overview

According to the paper, the training process consists of two stages:

1. **Initial Training (256 resolution)**: Train for ~3 days with 256x256 resolution
2. **Finetuning (512 resolution)**: Finetune for 20k iterations with 512x512 resolution, smaller learning rate (1e-4), and total batch size of 128

## Prerequisites

1. **Completed 256 resolution training**: You should have a trained checkpoint from the initial 256 resolution training
2. **512 resolution dataset**: Ensure your dataset supports 512x512 resolution (or update the dataset path in config)

## Usage

### Step 1: Prepare Your Pretrained Checkpoint

First, locate your 256 resolution checkpoint. It should be in your checkpoint directory, typically named like `step_XXXXXXX.pt`.

### Step 2: Update Configuration

Edit `configs/LVSM_scene_decoder_only_finetune_512.yaml` and set the pretrained checkpoint path:

```yaml
training:
  pretrained_checkpoint: "/path/to/your/256_resolution_checkpoint.pt"
```

### Step 3: Run Finetuning

Use the finetune script with the configuration:

```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    finetune.py --config configs/LVSM_scene_decoder_only_finetune_512.yaml
```

### Alternative: Command Line Override

You can also override the pretrained checkpoint path via command line:

```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    finetune.py --config configs/LVSM_scene_decoder_only_finetune_512.yaml \
    training.pretrained_checkpoint="/path/to/your/checkpoint.pt"
```

## Configuration Details

The finetune configuration includes the following key changes from the original 256 resolution training:

### Model Configuration
- `image_tokenizer.image_size: 512` (changed from 256)
- `target_pose_tokenizer.image_size: 512` (changed from 256)

### Training Configuration
- `lr: 0.0001` (1e-4, smaller learning rate for finetuning)
- `train_steps: 20000` (20k iterations)
- `batch_size_per_gpu: 16` (adjusted for 512 resolution and total batch size of 128)
- `warmup: 1000` (reduced warmup steps for finetuning)
- `checkpoint_dir: ./experiments/checkpoints/finetune_512` (separate directory for finetune checkpoints)

### Loss Configuration
- `proj_loss_type: cos` (you can change this to 'l2' or 'sl1' as needed)

## Expected Training Time

The finetuning should take approximately 3 days as mentioned in the paper, depending on your hardware configuration.

## Monitoring

The finetune script will:
- Log training progress every 20 steps
- Save checkpoints every 1000 steps
- Create visualizations every 1000 steps
- Save a final checkpoint when training completes

## Output

The finetuned model will be saved in `./experiments/checkpoints/finetune_512/` with:
- Intermediate checkpoints: `step_XXXXXXX.pt`
- Final checkpoint: `final_checkpoint.pt`

## Troubleshooting

### Memory Issues
If you encounter GPU memory issues with 512 resolution:
- Reduce `batch_size_per_gpu` in the config
- Increase `grad_accum_steps` to maintain the same effective batch size
- Consider using gradient checkpointing

### Dataset Issues
If your dataset doesn't support 512 resolution:
- Update the `dataset_path` in the config to point to 512 resolution data
- Or modify your dataset class to support dynamic resolution

### Checkpoint Loading Issues
If you encounter issues loading the pretrained checkpoint:
- Ensure the checkpoint path is correct and accessible
- Check that the checkpoint was saved properly from the 256 resolution training
- The script uses `strict=False` loading, so some parameter mismatches are expected

## Example Commands

### Single GPU Finetuning (for testing)
```bash
python finetune.py --config configs/LVSM_scene_decoder_only_finetune_512.yaml \
    training.pretrained_checkpoint="/path/to/checkpoint.pt" \
    training.batch_size_per_gpu=4
```

### Multi-GPU Finetuning (recommended)
```bash
torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    finetune.py --config configs/LVSM_scene_decoder_only_finetune_512.yaml \
    training.pretrained_checkpoint="/path/to/checkpoint.pt"
```

## Notes

- The finetune script automatically handles the resolution change from 256 to 512
- It uses a smaller learning rate and reduced warmup for stable finetuning
- The total batch size is configured to be 128 as specified in the paper
- All training metrics and logging are preserved for monitoring progress
