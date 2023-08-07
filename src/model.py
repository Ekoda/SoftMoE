import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from .config import Config


class Norm(nn.Module):
    def __init__(self, config: Config, include_bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.dimension))
        self.bias = nn.Parameter(torch.zeros(config.dimension)) if include_bias else None

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, config: Config, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.dimension, 2) * (-np.log(10000.0) / config.dimension))
        pe = torch.zeros(1, max_len, config.dimension)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiQueryAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.head_dimension = config.dimension // config.n_heads
        self.n_heads = config.n_heads
        self.q_projection = nn.Linear(config.dimension, config.dimension, bias=False)
        self.kv_projection = nn.Linear(config.dimension, self.head_dimension * 2, bias=False)
        self.linear = nn.Linear(config.dimension, config.dimension, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape
        Q = self.q_projection(x).view(batch_size, seq_len, self.n_heads, self.head_dimension).transpose(1, 2)
        K, V = self.kv_projection(x).unsqueeze(1).expand(-1, self.n_heads, -1, -1).split(self.head_dimension, dim=-1)
        heads = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.config.dropout, is_causal=False)
        concat = heads.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        linear = self.linear(concat)
        return self.dropout(linear)


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.dimension, config.dimension * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dimension * 4, config.dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SoftMoe(nn.Module):
    """
    Soft Mixture of Experts (SoftMoE) implementation as described in:
    "From Sparse to Soft Mixtures of Experts" by Joan Puigcerver, Carlos Riquelme, 
    Basil Mustafa, Neil Houlsby at Google DeepMind.

    The soft MoE is neither sparse, nor dense, but a mixture.
    The input activates all experts as in a dense MoE,
    but it is done so only fractionally as in a sparse MoE.
    
    This class provides a fully-differentiable sparse Transformer that addresses 
    challenges in MoEs like training instability, token dropping, inability to scale 
    the number of experts, or ineffective fine-tuning.

    https://arxiv.org/abs/2308.00951
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_experts)])
        self.phi = nn.Parameter(torch.randn(config.dimension, config.n_experts * config.slots_per_expert))
        
    def forward(self, x: torch.Tensor):
        logits = torch.matmul(x, self.phi) # (batch_size, seq_len, slots)
        dispatch_weights = F.softmax(logits, dim=-1)
        combine_weights = F.softmax(logits, dim=1)
        xs = torch.bmm(dispatch_weights.transpose(1, 2), x)
        ys = torch.cat(
            [expert(xs[:, i * self.config.slots_per_expert : (i + 1) * self.config.slots_per_expert, :]) 
                          for i, expert in enumerate(self.experts)],
            dim=1
            )
        y = torch.bmm(combine_weights, ys)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: Config):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attention = MultiQueryAttention(config)
        self.moe = SoftMoe(config)
        self.attention_norm = Norm(config)
        self.moe_norm = Norm(config)

    def forward(self, x: torch.Tensor):
        x = self.attention_norm(x)
        x = self.attention(x) + x
        x = self.moe_norm(x)
        x = self.moe(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Linear(config.flattened_patch_size(), config.dimension, bias=False)
        self.positional_encoding = PositionalEncoding(config)
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.dimension))
        self.layers = nn.ModuleList([TransformerBlock(i, config) for i in range(config.n_layers)])
        self.mlp_head = nn.Sequential(
            nn.Linear(config.dimension, config.dimension * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dimension * 4, config.n_outputs),
        )

    def forward(self, x: torch.Tensor):
        batch_size, _, _ = x.shape
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = torch.cat([self.class_token.repeat(batch_size, 1, 1), x], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.mlp_head(x[:, 0])
        return x
