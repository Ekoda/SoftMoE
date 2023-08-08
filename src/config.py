import torch
import yaml
from dataclasses import dataclass, asdict, fields


@dataclass
class Config():
    dimension: int = 512
    n_layers: int = 2
    n_heads: int = 8
    n_experts: int = 8
    slots_per_expert: int = 1
    n_outputs: int = 10
    n_channels: int = 1
    
    img_height: int = 28
    img_width: int = 28
    patch_size: int = 2*2
    batch_size: int = 16

    n_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    beta_1: float = 0.9
    beta_2: float = 0.95
    epsilon: float = 1e-8
    dropout: float = 0
    grad_clip: float = 1.0

    should_load: bool = False
    load_path: str = 'params/512.pt'
    
    should_save: bool = True
    save_path: str = 'params/512.pt'

    device: str = 'mps'

    def seq_len(self) -> int:
        return (self.img_height * self.img_width) // (self.patch_size ** 2)

    def flattened_patch_size(self) -> int:
        return self.patch_size * self.n_channels

    model_params = ['dimension', 'n_layers', 'n_heads', 'n_experts', 'slots_per_expert', 'n_outputs', 'n_channels']

    def save_to_yaml(self, train_loss: float, val_loss: float):
        data = {f: getattr(self, f) for f in self.model_params}
        data['info'] = {
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        with open(f'{self.save_path.removesuffix(".pt")}.yaml', 'w') as file:
            yaml.dump(data, file)

    def load_model_params(self):
        with open(f'{self.load_path.removesuffix(".pt")}.yaml', 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        for k, v in data.items():
            setattr(self, k, v)


