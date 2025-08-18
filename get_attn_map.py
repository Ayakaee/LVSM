import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from einops import rearrange
import os
import numpy as np

def get_attention_weights_external(model, data_batch, input, target, layer_idx=-1, block_type='cross'):
    """
    在外部手动计算注意力权重，不修改模型代码
    
    Args:
        model: LVSM模型
        data_batch: 数据批次
        layer_idx: 层索引
        block_type: 注意力块类型
    
    Returns:
        attention_weights: 注意力权重张量
    """
    model = model.module
    model.eval()
    
    # 存储中间结果
    q_values = []
    k_values = []
    v_values = []
    
    def hook_fn(module, input, output):
        """Hook函数来获取Q, K, V值"""
        # 对于self-attention，input[0]是x
        # 对于cross-attention，input[0]是q_input，input[1]是kv_input
        if block_type == 'self':
            x = input[0]
            # 手动计算Q, K, V
            qkv = module.to_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            
            # 重塑为多头格式
            q = rearrange(q, "b l (nh dh) -> b l nh dh", dh=module.head_dim)
            k = rearrange(k, "b l (nh dh) -> b l nh dh", dh=module.head_dim)
            v = rearrange(v, "b l (nh dh) -> b l nh dh", dh=module.head_dim)
            
            # 应用QK归一化（如果启用）
            if module.use_qk_norm:
                q = module.q_norm(q)
                k = module.k_norm(k)
                
        elif block_type == 'cross' or block_type == 'self_cross':
            q_input = input[0]
            kv_input = input[1]
            
            # 手动计算Q, K, V
            q = module.to_q(q_input)
            kv = module.to_kv(kv_input)
            k, v = kv.chunk(2, dim=-1)
            
            # 重塑为多头格式
            q = rearrange(q, "b l (nh dh) -> b l nh dh", dh=module.head_dim)
            k = rearrange(k, "b l (nh dh) -> b l nh dh", dh=module.head_dim)
            v = rearrange(v, "b l (nh dh) -> b l nh dh", dh=module.head_dim)
            
            # 应用QK归一化（如果启用）
            if module.use_qk_norm:
                q = module.q_norm(q)
                k = module.k_norm(k)
        
        q_values.append(q.detach())
        k_values.append(k.detach())
        v_values.append(v.detach())
    
    # 根据block_type注册hook到对应的注意力模块
    if block_type == 'cross':
        if model.cross_attn_blocks is not None and layer_idx < len(model.cross_attn_blocks):
            handle = model.cross_attn_blocks[layer_idx].attn.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Cross attention block {layer_idx} not found")
    elif block_type == 'self':
        if model.self_attn_blocks is not None and layer_idx < len(model.self_attn_blocks):
            handle = model.self_attn_blocks[layer_idx].attn.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Self attention block {layer_idx} not found")
    elif block_type == 'self_cross':
        if model.self_cross_blocks is not None and layer_idx < len(model.self_cross_blocks):
            # 对于self-cross block，我们获取cross-attention部分
            handle = model.self_cross_blocks[layer_idx].cross_attn.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Self-cross attention block {layer_idx} not found")
    else:
        raise ValueError(f"Unknown block type: {block_type}")
    
    try:
        with torch.no_grad():
            # 运行模型前向传播
            result = model.forward(data_batch, input, target, train=False, extract_features=True)
    finally:
        handle.remove()
    
    if not q_values or not k_values or not v_values:
        print("警告: 没有获取到Q, K, V值")
        return None
    
    # 获取Q, K, V值
    q = q_values[0]
    k = k_values[0]
    v = v_values[0]
    
    # 手动计算注意力权重
    attention_weights = compute_attention_weights(q, k, v)
    
    return result, attention_weights

def compute_attention_weights(q, k, v, attn_bias=None, attn_dropout=0.0, training=False):
    """
    手动计算注意力权重
    
    Args:
        q: Query张量 [batch, seq_len_q, num_heads, head_dim]
        k: Key张量 [batch, seq_len_k, num_heads, head_dim]
        v: Value张量 [batch, seq_len_k, num_heads, head_dim]
        attn_bias: 注意力偏置
        attn_dropout: Dropout概率
        training: 是否在训练模式
    
    Returns:
        attention_weights: 注意力权重 [batch, num_heads, seq_len_q, seq_len_k]
    """
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape
    
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    print(f"Q 变换后形状: {q.shape}") # -> [B, H, 3072, D_h]
    print(f"K 变换后形状: {k.shape}") # -> [B, H, 2048, D_h]

    # 2. 计算注意力分数
    # q: [B, H, seq_len_q, D_h]
    # k.transpose(-2, -1): [B, H, D_h, seq_len_k]
    # matmul 结果: [B, H, seq_len_q, seq_len_k]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # 应用注意力偏置（如果有）
    if attn_bias is not None:
        # 确保attn_bias的形状与attn_scores匹配
        if attn_bias.dim() == 2:
            # [seq_len_q, seq_len_k] -> [1, 1, seq_len_q, seq_len_k]
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
        elif attn_bias.dim() == 3:
            # [batch, seq_len_q, seq_len_k] -> [batch, 1, seq_len_q, seq_len_k]
            attn_bias = attn_bias.unsqueeze(1)
        attn_scores = attn_scores + attn_bias
    
    # 应用softmax
    attention_weights = torch.softmax(attn_scores, dim=-1)
    
    # 应用dropout（如果在训练模式）
    if training and attn_dropout > 0:
        attention_weights = torch.dropout(attention_weights, p=attn_dropout, train=True)
    
    # 重新排列维度以匹配标准格式 [batch, num_heads, seq_len_q, seq_len_k]
    attention_weights = attention_weights.transpose(1, 2)
    
    return attention_weights

def visualize_attention_maps_modified(model, images, layer_idx=-1, block_type='self'):
    """
    使用修改后的模型可视化注意力图
    """
    model.eval()
    
    # 获取注意力权重
    attention_weights = get_attention_weights(model, images, layer_idx, block_type)
    
    return attention_weights

def create_attention_visualization(image, attention_map, model_name):
    # 将注意力图调整到图像尺寸
    import cv2
    h, w = image.shape[:2]
    attention_resized = cv2.resize(attention_map, (w, h))
    
    # 创建热力图
    heatmap = plt.cm.plasma(attention_resized)
    
    return heatmap

def visualize_attention_weights(attention_weights, layer_name, output_dir, save_heatmap=True, save_attention_pattern=True):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重张量 [batch, num_heads, seq_len_q, seq_len_k]
        layer_name: 层名称
        output_dir: 输出目录
        save_heatmap: 是否保存热力图
        save_attention_pattern: 是否保存注意力模式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为numpy数组
    attention_weights = attention_weights[0]
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    print(f"=== {layer_name} 注意力权重分析 ===")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重统计:")
    print(f"  均值: {np.mean(attention_weights):.4f}")
    print(f"  标准差: {np.std(attention_weights):.4f}")
    print(f"  最小值: {np.min(attention_weights):.4f}")
    print(f"  最大值: {np.max(attention_weights):.4f}")
    
    # 1. 平均注意力权重热力图
    if save_heatmap:
        plt.figure(figsize=(12, 10))
        
        # 计算所有头的平均注意力权重
        avg_attention = np.mean(attention_weights, axis=0)  # [seq_len_q, seq_len_k]
        
        plt.imshow(avg_attention, cmap='viridis', aspect='auto')
        print(222)
        plt.colorbar(label='Average Attention Weight')
        plt.title(f'{layer_name} - Average Attention Weights')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{layer_name}_attention_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 每个注意力头的可视化
    if save_attention_pattern:
        num_heads = attention_weights.shape[0]
        num_cols = min(4, num_heads)
        num_rows = (num_heads + num_cols - 1) // num_cols
        
        plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        
        for head_idx in range(num_heads):
            plt.subplot(num_rows, num_cols, head_idx + 1)
            
            # 计算当前头的平均注意力权重
            head_attention = attention_weights[head_idx]  # [seq_len_q, seq_len_k]
            
            plt.imshow(head_attention, cmap='viridis', aspect='auto')
            plt.title(f'Head {head_idx}')
            plt.xticks([])
            plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{layer_name}_attention_heads.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 注意力权重分布
    plt.figure(figsize=(15, 5))
    
    # 注意力权重分布直方图
    plt.subplot(1, 3, 1)
    plt.hist(attention_weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{layer_name} - Attention Weight Distribution')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 每个头的平均注意力强度
    plt.subplot(1, 3, 2)
    head_avg_attention = np.mean(attention_weights, axis=(1, 2))  # [num_heads]
    plt.bar(range(len(head_avg_attention)), head_avg_attention)
    plt.title(f'{layer_name} - Average Attention per Head')
    plt.xlabel('Attention Head')
    plt.ylabel('Average Attention Weight')
    plt.grid(True, alpha=0.3)
    
    # 注意力权重的标准差
    plt.subplot(1, 3, 3)
    head_std_attention = np.std(attention_weights, axis=(0, 2, 3))  # [num_heads]
    plt.bar(range(len(head_std_attention)), head_std_attention)
    plt.title(f'{layer_name} - Attention Weight Std per Head')
    plt.xlabel('Attention Head')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{layer_name}_attention_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"注意力可视化完成！结果保存在: {output_dir}")

