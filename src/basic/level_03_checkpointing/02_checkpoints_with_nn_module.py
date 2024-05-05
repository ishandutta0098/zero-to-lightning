# Documentation Link:
# https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#save-a-checkpoint

import os

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LitConvClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.model = ConvClassifier()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def prepare_dataloaders():
    train_dataset = MNIST(
        "./", download=True, train=True, transform=transforms.ToTensor()
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    seed = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=seed
    )

    test_dataset = MNIST(
        "./", download=True, train=False, transform=transforms.ToTensor()
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    return train_dataloader, val_dataloader, test_dataloader


train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders()

# Train the Model
model = LitConvClassifier()

trainer = pl.Trainer(max_epochs=1, default_root_dir="experiments/")

trainer.fit(model, train_dataloader, val_dataloader)

# Load the module using the state dict

checkpoint_path = (
    "experiments/lightning_logs/version_0/checkpoints/epoch=0-step=1500.ckpt"
)

checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint["state_dict"]

classifier_model = ConvClassifier()
classifier_model.load_state_dict(state_dict)
