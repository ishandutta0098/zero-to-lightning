# Documentation Link
# https://lightning.ai/docs/pytorch/stable/integrations/hpu/basic.html

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# Import the HPUAccelerator
from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.strategies import HPUParallelStrategy
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


data_module = MNISTDataModule()
model = LitConvClassifier()

# Run on as many Gaudi devices as available by default
trainer = pl.Trainer(
    max_epochs=3,
    default_root_dir="experiments",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
        ModelSummary(max_depth=-1),
    ],
    precision="bf16-mixed",
    limit_train_batches=0.1,
    limit_val_batches=0.01,
    accelerator="auto",
    devices="auto",
    strategy="auto",
)

# equivalent to
trainer = pl.Trainer(
    max_epochs=3,
    default_root_dir="experiments",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
        ModelSummary(max_depth=-1),
    ],
    precision="bf16-mixed",
    limit_train_batches=0.1,
    limit_val_batches=0.01,
)

# Run on one Gaudi device
trainer = pl.Trainer(
    max_epochs=3,
    default_root_dir="experiments",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
        ModelSummary(max_depth=-1),
    ],
    precision="bf16-mixed",
    limit_train_batches=0.1,
    limit_val_batches=0.01,
    accelerator=HPUAccelerator(),
    devices="1",
)

# Run on multiple Gaudi devices
trainer = pl.Trainer(
    max_epochs=3,
    default_root_dir="experiments",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
        ModelSummary(max_depth=-1),
    ],
    precision="bf16-mixed",
    limit_train_batches=0.1,
    limit_val_batches=0.01,
    accelerator=HPUAccelerator(),
    devices="8",
)

# To train a Lightning model using multiple HPU nodes,
# set the num_nodes parameter with the available nodes in the Trainer class.
hpus = 8
parallel_hpus = [torch.device("hpu")] * hpus

trainer = pl.Trainer(
    max_epochs=3,
    default_root_dir="experiments",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
        ModelSummary(max_depth=-1),
    ],
    precision="bf16-mixed",
    limit_train_batches=0.1,
    limit_val_batches=0.01,
    accelerator=HPUAccelerator(),
    devices=hpus,
    strategy=HPUParallelStrategy(parallel_devices=parallel_hpus),
    num_nodes=2,
)

trainer.fit(model, data_module)

# Get Predictions
predictions = trainer.predict(model, data_module)
print(len(predictions))
