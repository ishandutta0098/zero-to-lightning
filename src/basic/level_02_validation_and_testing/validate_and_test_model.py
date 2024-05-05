# Documentation Link
# https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html

import os

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class LitConvClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Define the forward pass through the network
        # Input shape: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 32, 28, 28)
        x = F.max_pool2d(x, 2)  # Shape: (batch_size, 32, 14, 14)
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 64, 14, 14)
        x = F.max_pool2d(x, 2)  # Shape: (batch_size, 64, 7, 7)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 64*7*7)
        x = F.relu(self.fc1(x))  # Shape: (batch_size, 128)
        x = self.fc2(x)  # Shape: (batch_size, 10)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # The validation step is performed once per batch of data from the validation set.
        # It's used to check the model's performance on the validation set during training.
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        # The test step is performed once per batch of data from the test set.
        # It's used to assess the model's performance on unseen data after training is complete.
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


train_dataset = MNIST(
    "./", download=True, train=True, transform=transforms.ToTensor()
)

# Calculate training and validation split
# We will keep 80% data for training and 20% for validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

# Split the dataset into training and validation
seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size], generator=seed
)

test_dataset = MNIST(
    "./", download=True, train=False, transform=transforms.ToTensor()
)

# Create data loaders for loading the data in batches
train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

model = LitConvClassifier()

trainer = pl.Trainer(max_epochs=1)

trainer.fit(model, train_dataloader, val_dataloader)

# Test the model on the test set after training is complete
trainer.test(model, test_dataloader)
