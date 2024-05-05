# Documentation Link
# https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)


# Steps to enable Manual Optimization
# 1. Set `self.automatic_optimization=False`` in your LightningModule’s __init__.

# 2. Use the following functions and call them manually:

# 2.1 `self.optimizers()` to access your optimizers (one or multiple)

# 2.2 `optimizer.zero_grad()` to clear the gradients from the previous training step

# 2.3 `self.manual_backward(loss)` instead of loss.backward()

# 2.4 `optimizer.step()` to update your model parameters

# 2.5 `self.toggle_optimizer()` and `self.untoggle_optimizer()` if needed


class LitConvClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 1, 28, 28)

        self.learning_rate = learning_rate

        # Enable manual optimization
        self.automatic_optimization = False

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

    # Define the compute_loss method
    def compute_loss(self, batch):
        x, y = batch
        logits = self(x)  # Pass inputs through the model
        return F.cross_entropy(logits, y)  # Calculate cross-entropy loss

    # Here are three examples of how to use manual optimization in Lightning
    # Uncomment one of the examples to try it out!

    # Example 1: Basic Manual Optimization
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()

        return loss

    # # Example 2: Gradient Accumulation
    # def training_step(self, batch, batch_idx, N=5):
    #     opt = self.optimizers()

    #     # scale losses by 1/N (for N batches of gradient accumulation)
    #     loss = self.compute_loss(batch) / N
    #     self.manual_backward(loss)

    #     # accumulate gradients of N batches
    #     if (batch_idx + 1) % N == 0:
    #         opt.step()
    #         opt.zero_grad()

    #     return loss

    # Example 3: Gradient Clipping
    # def training_step(self, batch, batch_idx):
    #     opt = self.optimizers()

    #     # compute loss
    #     loss = self.compute_loss(batch)

    #     opt.zero_grad()
    #     self.manual_backward(loss)

    #     # clip gradients
    #     self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

    #     opt.step()

    #     return loss

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


data_module = MNISTDataModule()
model = LitConvClassifier()

trainer = pl.Trainer(
    max_epochs=3,
    default_root_dir="experiments",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
        ModelSummary(max_depth=-1),
    ],
    precision="16-mixed",
    limit_train_batches=0.1,
    limit_val_batches=0.01,
)

trainer.fit(model, data_module)

# Get Predictions
predictions = trainer.predict(model, data_module)
print(len(predictions))
