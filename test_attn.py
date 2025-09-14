from thop import profile
from model.transformer import QK_Norm_SelfAttentionBlock, QK_Norm_CrossAttentionBlock, QK_Norm_SelfCrossAttentionBlock, QK_Norm_FFNBlock, init_weights
import torch.nn as nn
import torch
import math

# class NativeAttentionBlock(nn.Module):
#     """
#     Transformer block using PyTorch's native MultiheadAttention
#     """
#     def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
        
#         # Layer normalization
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
        
#         # Native PyTorch multihead attention
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True  # Important: set batch_first=True for (batch, seq, dim) format
#         )
        
#         # MLP (Feed Forward Network)
#         hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
    
#     def forward(self, x, attn_mask=None, key_padding_mask=None):
#         # Self-attention with residual connection
#         norm_x = self.norm1(x)
#         attn_out, _ = self.attn(
#             query=norm_x, 
#             key=norm_x, 
#             value=norm_x,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask
#         )
#         x = x + attn_out
        
#         # MLP with residual connection
#         x = x + self.mlp(self.norm2(x))
#         return x

class NaiveAttention(nn.Module):
    """
    朴素版本的自注意力机制实现
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # QKV投影层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # 输出投影层
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        
        # 计算Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, dim)
        k = self.k_proj(x)  # (batch, seq_len, dim)
        v = self.v_proj(x)  # (batch, seq_len, dim)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        # 计算注意力分数: Q @ K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len, seq_len)
        
        # 应用mask（如果有的话）
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        out = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        
        # 重塑回原始维度
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)  # (batch, seq_len, dim)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out

class NaiveAttentionBlock(nn.Module):
    """
    使用朴素注意力的Transformer块
    """
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 朴素自注意力
        self.attn = NaiveAttention(dim, num_heads, dropout)
        
        # MLP (Feed Forward Network)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        x = x + self.attn(self.norm1(x), mask=mask)
        
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        
        return x

class TorchMultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # 输入 x: (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)  # MultiheadAttention 需要 (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        return x.transpose(0, 1)  # 恢复为 (batch_size, seq_len, embed_dim)

# self_attn_blocks = nn.Sequential(*[
#     QK_Norm_SelfAttentionBlock(
#         768, 12, use_qk_norm=True, use_flex_attention=True
#     ) for _ in range(24)
# ])

# self_attn_blocks = nn.Sequential(*[
#     TorchMultiheadAttentionBlock(
#         768, 12
#     ) for _ in range(24)
# ])

# Create the model using native PyTorch attention
self_attn_blocks = nn.Sequential(*[
    NaiveAttentionBlock(
        dim=768, 
        num_heads=12,  # 768 / 12 = 64 head_dim, matching your original setup
        mlp_ratio=4,
        dropout=0.1
    ) for _ in range(24)
])

vi, vt = (4, 3)
data = torch.ones((vt,1024*vi, 768))
flops, params = profile(self_attn_blocks, inputs=(data,))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs") # GFLOPs
print(f"参数量: {params / 1e6:.2f} M") # M (Million)