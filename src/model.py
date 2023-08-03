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


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: Config):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attention = MultiQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = Norm(config)
        self.ff_norm = Norm(config)

    def forward(self, x: torch.Tensor):
        x = self.attention_norm(x)
        x = self.attention(x) + x
        x = self.ff_norm(x)
        x = self.feed_forward(x) + x
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