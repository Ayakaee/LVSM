# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention
import math

try:
    import xformers.ops as xops
except ImportError:
    print('failed to import xofrmers')



def init_weights(module, std=0.02):
    """Initialize weights for linear and embedding layers.
    
    Args:
        module: Module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)



# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)



class MLP(nn.Module):
    """
    Multi-Layer Perceptron block.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    """
    
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        bias=False,
        dropout=0.0,
        activation=nn.GELU,
        mlp_dim=None,
    ):
        """
        Args:
            dim: Input dimension
            mlp_ratio: Multiplier for hidden dimension
            bias: Whether to use bias in linear layers
            dropout: Dropout probability
            activation: Activation function
            mlp_dim: Optional explicit hidden dimension (overrides mlp_ratio)
        """
        super().__init__()
        hidden_dim = mlp_dim if mlp_dim is not None else int(dim * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            activation(),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)



class QK_Norm_SelfAttention(nn.Module):
    """
    Self-attention with optional Q-K normalization.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
        use_flex_attention=False,
    ):
        """
        Args:
            dim: Input dimension
            head_dim: Dimension of each attention head
            qkv_bias: Whether to use bias in QKV projection
            fc_bias: Whether to use bias in output projection
            attn_dropout: Dropout probability for attention weights
            fc_dropout: Dropout probability for output projection
            use_qk_norm: Whether to use Q-K normalization
            use_flex_attention: Whether to use flex attention
            use_learnable_scale: Whether to use learnable scale factor for QK scores
        We use flash attention V2 for efficiency.
        """
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm
        self.use_flex_attention = use_flex_attention

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        if self.use_flex_attention:
            self.compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

    def forward(self, x, attn_bias=None, score_mod=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attn_bias: Optional attention bias mask
            score_mod: Optional score modification function for flex attention
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim) for t in (q, k, v))
        
        # Apply qk normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_flex_attention:
            q, k, v = (rearrange(t, "b s h d -> b h s d") for t in (q, k, v))
            if score_mod is None:
                # scaled product score
                x = self.compiled_flex_attention(q, k, v)
            else:
                x = self.compiled_flex_attention(q, k, v, score_mod=score_mod)
            
            x = rearrange(x, "b h s d -> b s h d")
        else:
            x = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        
        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        
        return x

class QK_Norm_CrossAttention(nn.Module):
    """
    Cross-attention with optional Q-K normalization.
    Q 来自 target，K/V 来自 input。
    """

    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
        use_flex_attention=False,
        use_log_scale=None,
    ):
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"

        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm
        self.use_flex_attention = use_flex_attention
        self.use_log_scale = use_log_scale

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        if self.use_log_scale == 'none':
            self.use_log_scale = None
        if isinstance(self.use_log_scale, int):
            self.scale = math.log(self.use_log_scale)
        elif self.use_log_scale == 'auto':
            self.scale = nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=True)

        # Learnable scale factor for QK scores (additional to the default head_dim scaling)

        if self.use_flex_attention:
            self.compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

    def forward(self, q_input, kv_input, v_input=None, attn_bias=None, score_mod=None):
        """
        Args:
            q_input: (batch, n_target, dim)  # target tokens
            kv_input: (batch, n_input, dim)  # input tokens
            attn_bias: attention bias
            score_mod: Optional score modification function for flex attention
        Returns:
            (batch, n_target, dim)
        """
        q = self.to_q(q_input)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
        k = rearrange(k, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
        v = rearrange(v, "b l (nh dh) -> b l nh dh", dh=self.head_dim)

        # QK Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply learnable scale factor
        if self.use_log_scale is not None:
            log_views = torch.log(torch.clamp(torch.tensor(v_input, dtype=torch.bfloat16), min=2))
            scale = log_views - self.scale + 1
            q = q * scale
            k = k * scale

        # Cross Attention
        if self.use_flex_attention:
            q, k, v = (rearrange(t, "b s h d -> b h s d") for t in (q, k, v))

            if score_mod is None:
                x = self.compiled_flex_attention(q, k, v)
            else:
                x = self.compiled_flex_attention(q, k, v, score_mod=score_mod)
            
            x = rearrange(x, "b h s d -> b s h d")
        else:
            x = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )

        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        return x


class QK_Norm_SelfAttentionBlock(nn.Module):
    """
    Standard transformer block with pre-normalization architecture.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    """

    def __init__(
        self,
        dim,
        head_dim,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
        use_flex_attention=False,
        use_log_scale=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.attn = QK_Norm_SelfAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
            use_flex_attention=use_flex_attention,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )


    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x



class QK_Norm_CrossAttentionBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        head_dim, 
        ln_bias=False, 
        attn_qkv_bias=False, 
        attn_dropout=0.0, 
        attn_fc_bias=False, 
        attn_fc_dropout=0.0, 
        mlp_ratio=4, 
        mlp_bias=False, 
        mlp_dropout=0.0, 
        use_qk_norm=True,
        use_flex_attention=False,
        use_log_scale=None,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.norm_kv = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.attn = QK_Norm_CrossAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
            use_flex_attention=use_flex_attention,
            use_log_scale=use_log_scale,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )

    def forward(self, kv, q, attn_bias=None):
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv), attn_bias=attn_bias)
        q = q + self.mlp(self.norm2(q))
        return q

class QK_Norm_SelfCrossAttentionBlock(nn.Module):
    """
    Block that alternates between self-attention and cross-attention within the same block.
    Self-attention -> Cross-attention -> FFN
    """
    def __init__(
        self, 
        dim, 
        head_dim, 
        ln_bias=False, 
        attn_qkv_bias=False, 
        attn_dropout=0.0, 
        attn_fc_bias=False,
        attn_fc_dropout=0.0, 
        mlp_ratio=4, 
        mlp_bias=False, 
        mlp_dropout=0.0, 
        use_qk_norm=True,
        use_flex_attention=False,
        use_log_scale=None,
    ):
        super().__init__()
        # Self-attention components
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.self_attn = QK_Norm_SelfAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
            use_flex_attention=use_flex_attention
        )
        
        # Cross-attention components
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.norm_kv = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.cross_attn = QK_Norm_CrossAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
            use_flex_attention=use_flex_attention,
            use_log_scale=use_log_scale,
        )
        
        # Shared FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )

    def forward(self, input, target, attn_bias=None):
        """
        Args:
            input: [b, seq*in, dim]
            target: [b*out, seq, dim]
            attn_bias: Optional attention bias mask for cross-attention
        """
        target = target + self.self_attn(self.norm1(target))
        bs, seq_in, _ = input.shape
        bs_out, seq, dim = target.shape
        target = target.view(bs, seq * bs_out // bs, dim)

        target = target + self.cross_attn(self.norm_q(target), self.norm_kv(input), v_input=seq_in // seq, attn_bias=attn_bias)
        target = target + self.mlp(self.norm2(target))

        target = target.view(bs_out, seq, dim)
        return target

class QK_Norm_FFNBlock(nn.Module):
    """
    只有FFN，没有Attention的Transformer Block。
    """
    def __init__(
        self,
        dim,
        head_dim=None,  # 兼容接口，实际不用
        ln_bias=False,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        **kwargs  # 兼容多余参数
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )

    def forward(self, x):
        x = x + self.mlp(self.norm(x))
        return x