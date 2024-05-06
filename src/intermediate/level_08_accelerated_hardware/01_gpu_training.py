# Documentation Link
# https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html

import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class LitConvClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 1, 28, 28)

        self.learning_rate = learning_rate

        # Define blocks of layers as submodules
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self(x)

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

model = LitConvClassifier()

# The below code will train the model on the GPU if unavailable it will throw an error.
# A Graphics Processing Unit (GPU), is a specialized hardware accelerator designed to speed up
# mathematical computations used in gaming and deep learning.

# Following are the different configuration examples which you can use to train your model on GPU
# based on your hardware configuration.

# run on as many GPUs as available by default
trainer = pl.Trainer(
    max_epochs=5,
    default_root_dir="experiments/",
    accelerator="auto",
    devices="auto",
    strategy="auto",
)
# equivalent to
trainer = pl.Trainer(max_epochs=5, default_root_dir="experiments/")

# run on one GPU
trainer = pl.Trainer(
    max_epochs=5, default_root_dir="experiments/", accelerator="gpu", devices=1
)

# run on multiple GPUs
trainer = pl.Trainer(
    max_epochs=5, default_root_dir="experiments/", accelerator="gpu", devices=8
)

# choose the number of devices automatically
trainer = pl.Trainer(
    max_epochs=5, default_root_dir="experiments/", accelerator="gpu", devices="auto"
)

# DEFAULT (int) specifies how many GPUs to use per node
k = 2
trainer = pl.Trainer(
    max_epochs=5, default_root_dir="experiments/", accelerator="gpu", devices=k
)
# equivalent to
trainer = pl.Trainer(
    max_epochs=5,
    default_root_dir="experiments/",
    accelerator="gpu",
    devices=list(range(k)),
)

# Specify which GPUs to use (don't use when running on cluster)
trainer = pl.Trainer(
    max_epochs=5, default_root_dir="experiments/", accelerator="gpu", devices=[0, 1]
)
# equivalent to
trainer = pl.Trainer(
    max_epochs=5, default_root_dir="experiments/", accelerator="gpu", devices="0, 1"
)

# To use all available GPUs put -1 or '-1'
# equivalent to `list(range(torch.cuda.device_count())) and `"auto"`
trainer = pl.Trainer(
    max_epochs=5, default_root_dir="experiments/", accelerator="gpu", devices=-1
)

trainer.fit(model, train_dataloader, val_dataloader)
