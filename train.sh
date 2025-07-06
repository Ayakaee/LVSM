export NCCL_SOCKET_IFNAME="lo"
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE="disabled"
torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/LVSM_scene_decoder_only.yaml \
    # model.transformer.n_layer = 12 \
    # training.batch_size_per_gpu = 16
