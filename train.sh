export NCCL_SOCKET_IFNAME="lo"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export WANDB_MODE="disabled"
export WANDB_MODE=offline
torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/LVSM_scene_decoder_only.yaml \
    train.wandb_exp_name = 7.7-repa
    # model.transformer.n_layer = 12 \
    # training.batch_size_per_gpu = 16

# echo "Training finished, running occupy_gpu.py..."
# nohup python /inspire/hdd/global_user/chenxinyan-240108120066/youjunqi/occupy_gpu.py > /dev/null 2>&1 &