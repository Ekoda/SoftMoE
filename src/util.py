from torch.utils.data import DataLoader, random_split
from src import Config, ImageTokenizer, TokenizedImagesDataset


def prepare_data(data, tokenizer: ImageTokenizer, config: Config, val_split: float = 0.2) -> (DataLoader, DataLoader):
    train_size = int(len(data)*(1-val_split))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])
    tokenized_train, tokenized_val = TokenizedImagesDataset(train_data, tokenizer), TokenizedImagesDataset(val_data, tokenizer)
    train_data_loader, val_data_loader = tokenized_train.to_dataloader(config), tokenized_val.to_dataloader(config)
    return train_data_loader, val_data_loader