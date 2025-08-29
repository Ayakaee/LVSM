#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LVSM注意力权重提取示例

这个脚本展示了如何使用外部计算方法获取LVSM模型的注意力权重。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from get_attn_map import get_attention_weights_external
from attention_visualization import visualize_attention_weights

def example_usage():
    """
    使用示例：如何获取和可视化注意力权重
    """
    print("=== LVSM注意力权重提取示例 ===")
    
    # 注意：这里需要根据实际的模型和数据加载逻辑来调整
    print("请根据您的实际环境调整以下代码：")
    
    """
    # 1. 加载模型
    from model.LVSM_scene_decoder_only import Images2LatentScene
    from utils.config import load_config
    
    config = load_config('path/to/config.yaml')
    model = Images2LatentScene(config)
    model.load_ckpt('path/to/checkpoint.pt')
    model.eval()
    
    # 2. 准备数据
    # 这里需要根据您的数据格式来准备data_batch
    data_batch = prepare_data_batch('path/to/data')
    
    # 3. 获取注意力权重
    layer_idx = 0  # 第0层
    block_type = 'cross'  # cross attention
    
    attention_weights = get_attention_weights_external(
        model, data_batch, layer_idx, block_type
    )
    
    # 4. 可视化注意力权重
    if attention_weights is not None:
        output_dir = f'attention_vis_layer_{layer_idx}'
        visualize_attention_weights(
            attention_weights, 
            f"{block_type}_layer_{layer_idx}", 
            output_dir
        )
        print(f"注意力可视化完成！结果保存在: {output_dir}")
    else:
        print("未能获取注意力权重")
    """
    
    print("\n主要步骤说明：")
    print("1. 加载LVSM模型")
    print("2. 准备数据批次（包含输入图像和相机参数）")
    print("3. 调用get_attention_weights_external()获取注意力权重")
    print("4. 使用visualize_attention_weights()进行可视化")
    
    print("\n可用的block_type:")
    print("- 'self': Self-attention blocks")
    print("- 'cross': Cross-attention blocks") 
    print("- 'self_cross': Self-cross attention blocks")
    
    print("\n注意事项:")
    print("- 确保模型处于eval模式")
    print("- 使用torch.no_grad()来节省内存")
    print("- 注意力权重形状为 [batch, num_heads, seq_len, seq_len]")

def test_attention_computation():
    """
    测试注意力权重计算函数
    """
    print("\n=== 测试注意力权重计算 ===")
    
    # 创建测试数据
    batch_size = 2
    num_heads = 8
    seq_len = 256
    head_dim = 64
    
    # 创建Q, K, V张量
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    print(f"Q形状: {q.shape}")
    print(f"K形状: {k.shape}")
    print(f"V形状: {v.shape}")
    
    # 测试注意力权重计算
    from get_attn_map import compute_attention_weights
    
    attention_weights = compute_attention_weights(q, k, v)
    
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重统计:")
    print(f"  均值: {attention_weights.mean():.4f}")
    print(f"  标准差: {attention_weights.std():.4f}")
    print(f"  最小值: {attention_weights.min():.4f}")
    print(f"  最大值: {attention_weights.max():.4f}")
    
    # 验证softmax归一化
    attention_sum = attention_weights.sum(dim=-1)
    print(f"每行注意力权重和（应该接近1）: {attention_sum.mean():.4f}")
    
    print("注意力权重计算测试完成！")

if __name__ == "__main__":
    example_usage()
    test_attention_computation() 