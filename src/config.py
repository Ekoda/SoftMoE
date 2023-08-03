import torch
from dataclasses import dataclass

@dataclass
class Config:
    dimension: int = 256
    n_layers: int = 6
    n_heads: int = 8
    
    img_height: int = 28
    img_width: int = 28
    patch_size: int = 2*2
    n_channels: int = 1
    batch_size: int = 16
    n_outputs: int = 10

    n_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    beta_1: float = 0.9
    beta_2: float = 0.95
    epsilon: float = 1e-8
    dropout: float = 0
    grad_clip: float = 1.0
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def seq_len(self) -> int:
        return (self.img_height * self.img_width) // (self.patch_size ** 2)

    def flattened_patch_size(self) -> int:
        return self.patch_size * self.n_channels