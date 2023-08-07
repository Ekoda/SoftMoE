import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from PIL import Image
from src import Transformer, Config, ImageTokenizer, TokenizedImagesDataset
from tqdm import tqdm, tqdm_notebook


class VisionTransformer:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = ImageTokenizer(config)
        if config.should_load:
            self.load_model()
        else:
            self.model = Transformer(config).to(config.device)

    def load_model(self):
        self.config.load_model_params()
        self.model = Transformer(self.config).to(self.config.device)
        self.model.load_state_dict(torch.load(self.config.load_path))

    def train(self, train_data: DataLoader, val_data: DataLoader):
        model = self.model
        config = self.config

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta_1, config.beta_2),
            eps=config.epsilon
            )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(config.n_epochs):

            model.train()
            start_time = time.time()
            train_loss = 0
            for i, batch in tqdm(enumerate(train_data), total=len(train_data), desc="Training"):
                inputs, labels = batch
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                optimizer.zero_grad()
                logits = model.forward(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                if config.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_data)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_data), total=len(val_data), desc="Validation"):
                    inputs, labels = batch
                    inputs, labels = inputs.to(config.device), labels.to(config.device)
                    logits = model.forward(inputs)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                val_loss = val_loss / len(val_data)

            end_time = time.time()
            epoch_time = end_time - start_time
            print(
                f'Epoch {epoch}:\n'
                    f'\tTrain Loss: {train_loss}\n'
                    f'\tVal Loss: {val_loss}\n'
                    f'\tTime: {epoch_time:.3f}s'
                )

            if config.should_save:
                torch.save(model.state_dict(), config.save_path)
                config.save_to_yaml(train_loss, val_loss)


    def predict(self, x: Image.Image or torch.Tensor) -> int:
        if len(x.shape) != 2:
            x = self.tokenizer.encode(x)
        model = self.model.eval()
        x = x.to(self.config.device).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probabilities = torch.softmax(logits, dim=-1)[0]
            prediction = torch.argmax(probabilities, dim=-1)

        return prediction.item()
