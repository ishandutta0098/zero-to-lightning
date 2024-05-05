# Documentation Link
#  https://lightning.ai/docs/pytorch/stable/common/early_stopping.html


import os

import lightning.pytorch as pl
import torch

# Import the early stopping callback
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

        self.learning_rate = learning_rate
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # First we log the loss of interest
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

# Then pass the callback to the trainer
trainer = pl.Trainer(
    max_epochs=3,
    default_root_dir="experiments/",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
)
trainer.fit(model, train_dataloader, val_dataloader)

# Or customize the early stopping callback and pass it to the trainer
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True)
trainer = pl.Trainer(
    max_epochs=3, default_root_dir="experiments/", callbacks=[early_stopping]
)

trainer.fit(model, train_dataloader, val_dataloader)

# Additional parameters that stop training at extreme points:
# --> stopping_threshold: Stops training immediately once the monitored quantity reaches this threshold.
#                         It is useful when we know that going beyond a certain optimal value does not further benefit us.

# --> divergence_threshold: Stops training as soon as the monitored quantity becomes worse than this threshold.
#                           When reaching a value this bad, we believes the model cannot recover anymore
#                           and it is better to stop early and run with different initial conditions.

# --> check_finite: When turned on, it stops training if the monitored metric becomes NaN or infinite.

# --> check_on_train_epoch_end: When turned on, it checks the metric at the end of a training epoch.
#                               Use this only when you are monitoring any metric logged within training-specific
#                               hooks on epoch-level.
