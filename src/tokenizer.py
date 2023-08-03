import torch
from torchvision import transforms
from PIL import Image
from src import Config
from torch.utils.data import Dataset, DataLoader


class ImageTokenizer:
    def __init__(self, config: Config):
        self.patch_size = config.patch_size
        self.channels = config.n_channels
        self.transform = transforms.Compose([
            transforms.Resize((config.img_height, config.img_width)),
            transforms.ToTensor()
        ])

    def encode(self, image: Image.Image or torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = self.transform(image)
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, patches.size(-1)*patches.size(-2)*patches.size(-3))
        return patches # (n_patches, patch_size*patch_size*n_channels)


class TokenizedImagesDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        tokenized_image = self.tokenizer.encode(image)
        return tokenized_image, label

    def to_dataloader(self, config: Config):
        return DataLoader(self, batch_size=config.batch_size, shuffle=True)

