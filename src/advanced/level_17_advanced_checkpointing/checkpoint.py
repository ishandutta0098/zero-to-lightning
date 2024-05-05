# Documentation Link
# https://lightning.ai/docs/pytorch/stable/common/checkpointing_advanced.html

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


# In this example we will learn how to modify a checkpoint
# We create a custom attribute train_batches_processed and increment it in the training_step
# We then modify the checkpoint to save this attribute
class LitConvClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 1, 28, 28)

        self.learning_rate = learning_rate

        # Custom attribute to keep track of training batches processed
        self.train_batches_processed = 0

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

        # Increment custom attribute train_batches_processed
        self.train_batches_processed += 1
        self.log("train_batches_processed", self.train_batches_processed)

        return loss

    def on_save_checkpoint(self, checkpoint):
        # Add the custom attribute to the checkpoint
        checkpoint["train_batches_processed"] = self.train_batches_processed

    def on_load_checkpoint(self, checkpoint):
        # Load the custom attribute from the checkpoint
        self.train_batches_processed = checkpoint.get("train_batches_processed", 0)

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

# Manually load the saved checkpoint
checkpoint_path = trainer.checkpoint_callback.best_model_path
print(f"\nLoading checkpoint from: {checkpoint_path}")

# Load the model from the checkpoint
loaded_model = LitConvClassifier.load_from_checkpoint(checkpoint_path)

# Print the custom attribute stored in the checkpoint
# This is to check if the custom attribute is stored and loaded correctly
print(
    f"\nTrain batches processed (from checkpoint): {loaded_model.train_batches_processed}"
)

# Get Predictions
predictions = trainer.predict(model, data_module)
print(len(predictions))
