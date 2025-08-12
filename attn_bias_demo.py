#!/usr/bin/env python3
"""
演示attn_bias作用的示例脚本
"""

import torch
import torch.nn.functional as F

def demonstrate_attn_bias():
    """演示attn_bias在attention计算中的作用"""
    print("=== Attention Bias 演示 ===")
    
    # 模拟数据
    batch_size, seq_len, dim = 2, 4, 8
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)
    
    print(f"输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # 1. 不使用attn_bias的标准attention
    print("\n1. 标准attention（无bias）:")
    scores = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    print(f"Attention scores形状: {scores.shape}")
    print(f"Attention weights:\n{attn_weights[0]}")
    print(f"输出形状: {output.shape}")
    
    # 2. 使用attn_bias屏蔽某些位置
    print("\n2. 使用attn_bias屏蔽位置[0, 2]和[1, 3]:")
    
    # 创建attn_bias，屏蔽特定位置
    attn_bias = torch.zeros(batch_size, seq_len, seq_len)
    attn_bias[0, :, 2] = float('-inf')  # 屏蔽第0个batch的第2个位置
    attn_bias[1, :, 3] = float('-inf')  # 屏蔽第1个batch的第3个位置
    
    print(f"Attention bias:\n{attn_bias[0]}")
    
    # 计算带bias的attention
    scores_with_bias = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)
    scores_with_bias = scores_with_bias + attn_bias  # 关键步骤！
    attn_weights_with_bias = F.softmax(scores_with_bias, dim=-1)
    output_with_bias = torch.matmul(attn_weights_with_bias, v)
    
    print(f"带bias的attention weights:\n{attn_weights_with_bias[0]}")
    print(f"带bias的输出形状: {output_with_bias.shape}")
    
    # 3. 比较差异
    print("\n3. 比较差异:")
    weight_diff = attn_weights - attn_weights_with_bias
    print(f"权重差异:\n{weight_diff[0]}")
    
    # 检查被屏蔽的位置是否真的变成了0
    print(f"位置[0, 2]的权重: {attn_weights_with_bias[0, :, 2]}")
    print(f"位置[1, 3]的权重: {attn_weights_with_bias[1, :, 3]}")

def demonstrate_view_masking():
    """演示视角屏蔽中的attn_bias使用"""
    print("\n=== 视角屏蔽中的Attention Bias ===")
    
    # 模拟视角数据
    batch_size, v_input, v_target, n_patches, dim = 2, 4, 2, 3, 8
    
    # 创建input和target tokens
    input_tokens = torch.randn(batch_size, v_input * n_patches, dim)
    target_tokens = torch.randn(batch_size * v_target, n_patches, dim)
    
    print(f"Input tokens形状: {input_tokens.shape}")
    print(f"Target tokens形状: {target_tokens.shape}")
    
    # 创建attention mask，屏蔽第一个视角
    attn_mask = torch.zeros(batch_size, v_target, n_patches, v_input * n_patches)
    attn_mask[:, :, :, :n_patches] = float('-inf')  # 屏蔽第一个视角的所有patches
    
    print(f"Attention mask形状: {attn_mask.shape}")
    print(f"被屏蔽的patches数量: {torch.isinf(attn_mask).sum().item()}")
    
    # 模拟cross attention计算
    print("\n模拟cross attention计算:")
    
    for b_idx in range(batch_size):
        for t_idx in range(v_target):
            print(f"\nBatch {b_idx}, Target {t_idx}:")
            
            # 获取当前target的tokens
            target_idx = b_idx * v_target + t_idx
            cur_target = target_tokens[target_idx:target_idx+1]  # [1, n_patches, dim]
            
            # 获取对应的mask
            cur_mask = attn_mask[b_idx, t_idx]  # [n_patches, v_input * n_patches]
            
            print(f"  Target tokens形状: {cur_target.shape}")
            print(f"  Mask形状: {cur_mask.shape}")
            
            # 模拟attention计算
            # 这里简化处理，实际应该用Q、K、V
            scores = torch.randn(n_patches, v_input * n_patches)  # 模拟attention scores
            scores_with_bias = scores + cur_mask
            
            # 应用softmax
            attn_weights = F.softmax(scores_with_bias, dim=-1)
            
            print(f"  原始scores范围: [{scores.min():.3f}, {scores.max():.3f}]")
            print(f"  带bias的scores范围: [{scores_with_bias.min():.3f}, {scores_with_bias.max():.3f}]")
            print(f"  Attention weights范围: [{attn_weights.min():.3f}, {attn_weights.max():.3f}]")
            
            # 检查被屏蔽的位置
            masked_positions = torch.isinf(cur_mask)
            if masked_positions.any():
                print(f"  被屏蔽位置的权重: {attn_weights[masked_positions]}")
                print(f"  被屏蔽位置的权重和: {attn_weights[masked_positions].sum():.6f}")

def demonstrate_different_bias_values():
    """演示不同bias值的效果"""
    print("\n=== 不同Bias值的效果 ===")
    
    # 模拟attention scores
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [2.0, 1.0, 2.0, 3.0],
                          [3.0, 2.0, 1.0, 2.0],
                          [4.0, 3.0, 2.0, 1.0]])
    
    print(f"原始scores:\n{scores}")
    
    # 不同的bias值
    bias_values = [0.0, -1.0, -5.0, float('-inf')]
    
    for bias in bias_values:
        print(f"\nBias = {bias}:")
        
        # 创建bias矩阵，对第2列应用bias
        bias_matrix = torch.zeros_like(scores)
        bias_matrix[:, 1] = bias
        
        # 计算带bias的scores
        scores_with_bias = scores + bias_matrix
        attn_weights = F.softmax(scores_with_bias, dim=-1)
        
        print(f"带bias的scores:\n{scores_with_bias}")
        print(f"Attention weights:\n{attn_weights}")
        print(f"第2列的权重和: {attn_weights[:, 1].sum():.6f}")

if __name__ == "__main__":
    print("Attention Bias 详细演示")
    print("=" * 60)
    
    demonstrate_attn_bias()
    demonstrate_view_masking()
    demonstrate_different_bias_values()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n关键要点:")
    print("1. attn_bias在attention分数计算后、softmax前添加")
    print("2. float('-inf')会导致对应位置的attention权重变成0")
    print("3. 负值会降低对应位置的attention权重")
    print("4. 正值会提高对应位置的attention权重") 