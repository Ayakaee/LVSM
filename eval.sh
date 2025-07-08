export NCCL_SOCKET_IFNAME="lo"
torchrun --nproc_per_node 4 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
inference.py --config configs/LVSM_scene_decoder_only.yaml \
    training.target_has_input =  false \
    training.num_views = 5 \
    training.square_crop = true \
    training.num_input_views = 2 \
    training.num_target_views = 3 \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
    inference_out_dir = ./experiments/evaluation/test