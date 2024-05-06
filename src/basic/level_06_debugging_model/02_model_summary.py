# Documentation Link
# https://lightning.ai/docs/pytorch/stable/debug/debugging_basic.html

import os

import lightning.pytorch as pl
import torch

# Used for child modules in the model summary
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


# We have updated the model to use nn.Sequential() and named the blocks of layers.
# This will help us understand the Model Summary output.
class LitConvClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Another debugging tool is to display the intermediate input- and output sizes of
        # all your layers by setting the example_input_array attribute in your LightningModule.
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

# Default
# Whenever the .fit() function gets called,
# the Trainer will print the weights summary for the LightningModule.
print("\n----------------------------------")
print("Default Model Summary")
print("----------------------------------")
trainer = pl.Trainer(
    max_epochs=1,
    default_root_dir="experiments/",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
)
trainer.fit(model, train_dataloader, val_dataloader)

# Child Modules
print("\n----------------------------------")
print("Child Modules Model Summary")
print("----------------------------------")
trainer = pl.Trainer(
    max_epochs=1,
    default_root_dir="experiments/",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
        ModelSummary(max_depth=-1),
    ],
)
trainer.fit(model, train_dataloader, val_dataloader)

# Turn off model summary
print("\n----------------------------------")
print("Turn off Model Summary")
print("----------------------------------")
trainer = pl.Trainer(
    max_epochs=1,
    default_root_dir="experiments/",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    enable_model_summary=False,
)
trainer.fit(model, train_dataloader, val_dataloader)
