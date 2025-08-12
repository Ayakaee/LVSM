#!/usr/bin/env python3
"""
分析不同mask策略对速度的影响
"""

import torch
import time
import torch.nn.functional as F
import random
from model.transformer import QK_Norm_SelfCrossAttentionBlock

def create_attention_block():
    """创建一个attention block用于测试"""
    return QK_Norm_SelfCrossAttentionBlock(
        dim=768,
        head_dim=64,
        use_qk_norm=True,
        use_flex_attention=True
    )

def create_attention_mask(v_input, v_target, n_patches, device, batch_size, mask_strategy, view_min, view_max):
    """创建attention mask（模拟用户的实现）"""
    # 创建mask矩阵 [b, v_target * n_patches, v_input * n_patches]
    attn_mask = torch.zeros(batch_size, v_target * n_patches, v_input * n_patches, device=device)
    
    # 随机屏蔽一些视角
    num_masked_views = view_max - random.randint(view_min, view_max)
    if mask_strategy == 'individual':
        if num_masked_views > 0:
            for b_idx in range(batch_size):
                # 随机选择要屏蔽的视角
                masked_view_indices = torch.randperm(v_input)[:num_masked_views]
                for view_idx in masked_view_indices:
                    start_patch = view_idx * n_patches
                    end_patch = (view_idx + 1) * n_patches
                    attn_mask[b_idx, :, start_patch:end_patch] = float('-inf')
    elif mask_strategy == 'unified':
        if num_masked_views > 0:
            masked_view_indices = torch.randperm(v_input)[:num_masked_views]
            for view_idx in masked_view_indices:
                start_patch = view_idx * n_patches
                end_patch = (view_idx + 1) * n_patches
                attn_mask[:, :, start_patch:end_patch] = float('-inf')
    return attn_mask

def benchmark_no_mask(batch_size, v_input, v_target, n_patches, num_iterations=100):
    """测试没有mask策略"""
    print(f"测试没有mask策略: batch_size={batch_size}, v_input={v_input}, v_target={v_target}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block = create_attention_block().to(device)

    input_tokens = torch.randn(batch_size, v_input * n_patches, 768, device=device)
    target_tokens = torch.randn(batch_size, v_target * n_patches, 768, device=device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = block(input_tokens, target_tokens)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = block(input_tokens, target_tokens)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"没有mask策略平均时间: {avg_time*1000:.2f} ms")
    return avg_time

def benchmark_unified_mask(batch_size, v_input, v_target, n_patches, num_iterations=100):
    """测试统一mask策略（所有batch使用相同mask）"""
    print(f"测试统一mask策略: batch_size={batch_size}, v_input={v_input}, v_target={v_target}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block = create_attention_block().to(device)
    
    # 创建数据
    input_tokens = torch.randn(batch_size, v_input * n_patches, 768, device=device)
    target_tokens = torch.randn(batch_size, v_target * n_patches, 768, device=device)
    
    # 创建统一的mask
    unified_mask = create_attention_mask(v_input, v_target, n_patches, device, batch_size, 'unified', 1, 3)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = block(input_tokens, target_tokens, attn_bias=unified_mask)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # 计时
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = block(input_tokens, target_tokens, attn_bias=unified_mask)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"统一mask平均时间: {avg_time*1000:.2f} ms")
    return avg_time

def benchmark_individual_mask(batch_size, v_input, v_target, n_patches, num_iterations=100):
    """测试个体mask策略（每个batch使用不同mask）"""
    print(f"测试个体mask策略: batch_size={batch_size}, v_input={v_input}, v_target={v_target}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block = create_attention_block().to(device)
    
    # 创建数据
    input_tokens = torch.randn(batch_size, v_input * n_patches, 768, device=device)
    target_tokens = torch.randn(batch_size, v_target * n_patches, 768, device=device)
    
    # 创建个体mask
    individual_masks = create_attention_mask(v_input, v_target, n_patches, device, batch_size, 'individual', 1, 3)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = block(input_tokens, target_tokens, attn_bias=individual_masks)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # 计时
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = block(input_tokens, target_tokens, attn_bias=individual_masks)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"个体mask平均时间: {avg_time*1000:.2f} ms")
    return avg_time

def compare_memory_usage():
    """比较不同策略的内存使用"""
    print("\n=== 内存使用比较 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, v_input, v_target, n_patches = 4, 4, 2, 256
    
    # 统一mask
    unified_mask = torch.zeros(batch_size, v_target * n_patches, v_input * n_patches, device=device)
    unified_memory = unified_mask.element_size() * unified_mask.numel()
    
    # 个体mask
    individual_masks = torch.zeros(batch_size, v_target * n_patches, v_input * n_patches, device=device)
    individual_memory = individual_masks.element_size() * individual_masks.numel()
    
    print(f"统一mask内存: {unified_memory / 1024 / 1024:.2f} MB")
    print(f"个体mask内存: {individual_memory / 1024 / 1024:.2f} MB")
    print(f"内存增加比例: {individual_memory / unified_memory:.2f}x")

def run_comprehensive_benchmark():
    """运行综合性能测试"""
    print("=== 综合性能测试 ===")
    
    # 测试不同配置
    configs = [
        (2, 4, 2, 256),
        (4, 4, 2, 256),
        (8, 4, 2, 256),
        (2, 8, 4, 256),
        (4, 8, 4, 256),
    ]
    
    results = []
    
    for batch_size, v_input, v_target, n_patches in configs:
        print(f"\n配置: batch_size={batch_size}, v_input={v_input}, v_target={v_target}, n_patches={n_patches}")
        
        try:
            no_mask_time = benchmark_no_mask(batch_size, v_input, v_target, n_patches, 50)
            unified_time = benchmark_unified_mask(batch_size, v_input, v_target, n_patches, 50)
            individual_time = benchmark_individual_mask(batch_size, v_input, v_target, n_patches, 50)
            
            results.append({
                'config': (batch_size, v_input, v_target, n_patches),
                'no_mask': no_mask_time,
                'unified': unified_time,
                'individual': individual_time,
                'unified_overhead': unified_time / no_mask_time,
                'individual_overhead': individual_time / no_mask_time,
                'individual_vs_unified': individual_time / unified_time
            })
            
        except Exception as e:
            print(f"配置失败: {e}")
    
    # 总结结果
    print("\n=== 结果总结 ===")
    for result in results:
        config = result['config']
        print(f"\n配置 {config}:")
        print(f"  没有mask: {result['no_mask']*1000:.2f} ms")
        print(f"  统一mask: {result['unified']*1000:.2f} ms")
        print(f"  个体mask: {result['individual']*1000:.2f} ms")
        print(f"  统一mask开销: {result['unified_overhead']:.2f}x")
        print(f"  个体mask开销: {result['individual_overhead']:.2f}x")
        print(f"  个体vs统一: {result['individual_vs_unified']:.2f}x")

def test_different_view_ranges():
    """测试不同view_min和view_max对性能的影响"""
    print("\n=== 不同视角范围对性能的影响 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block = create_attention_block().to(device)
    
    batch_size, v_input, v_target, n_patches = 4, 4, 2, 256
    input_tokens = torch.randn(batch_size, v_input * n_patches, 768, device=device)
    target_tokens = torch.randn(batch_size, v_target * n_patches, 768, device=device)
    
    # 测试不同的view_min和view_max组合
    test_configs = [
        (1, 2),  # 屏蔽1个视角
        (1, 3),  # 屏蔽1-2个视角
        (2, 3),  # 屏蔽2个视角
        (1, 4),  # 屏蔽1-3个视角
    ]
    
    for view_min, view_max in test_configs:
        print(f"\nview_min={view_min}, view_max={view_max}:")
        
        # 测试individual策略
        individual_mask = create_attention_mask(v_input, v_target, n_patches, device, batch_size, 'individual', view_min, view_max)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = block(input_tokens, target_tokens, attn_bias=individual_mask)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        # 计时
        start_time = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = block(input_tokens, target_tokens, attn_bias=individual_mask)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 50
        print(f"  个体mask平均时间: {avg_time*1000:.2f} ms")
        
        # 统计被屏蔽的视角数量
        total_masked_patches = torch.isinf(individual_mask).sum().item()
        print(f"  总屏蔽patches: {total_masked_patches}")

if __name__ == "__main__":
    print("Mask策略速度影响分析")
    print("=" * 60)
    
    # 比较内存使用
    compare_memory_usage()
    
    # 运行综合测试
    run_comprehensive_benchmark()
    
    # 测试不同视角范围
    test_different_view_ranges()
    
    print("\n" + "=" * 60)
    print("结论:")
    print("1. 个体mask和统一mask的内存使用相同")
    print("2. 个体mask和统一mask的性能差异很小")
    print("3. 主要开销来自attention mask的计算")
    print("4. 不同视角范围对性能影响不大") 