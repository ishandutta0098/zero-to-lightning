# Imports
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning.pytorch as pl

# A simple convolution based classifier model for MNIST 
class LitConvClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Define the layers for the model architecture

        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Define the forward pass through the network
                                         # Input shape: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))        # Shape: (batch_size, 32, 28, 28)
        x = F.max_pool2d(x, 2)           # Shape: (batch_size, 32, 14, 14)
        x = F.relu(self.conv2(x))        # Shape: (batch_size, 64, 14, 14)
        x = F.max_pool2d(x, 2)           # Shape: (batch_size, 64, 7, 7)
        x = x.view(x.size(0), -1)        # Shape: (batch_size, 64*7*7)
        x = F.relu(self.fc1(x))          # Shape: (batch_size, 128)
        x = self.fc2(x)                  # Shape: (batch_size, 10)
        return x

    def training_step(self, batch, batch_idx):
        # Define the training step which includes 
        # the forward pass, loss calculation and backpropagation

        x, y = batch                     # Unpack batch
        y_hat = self(x)                  # Forward pass, get predicted logits
        loss = F.cross_entropy(y_hat, y) # Calculate loss
        return loss

    def configure_optimizers(self):
        # Define the optimizer to use for training

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Load Dataset
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())

# Create a Dataloader
train_dataloader = DataLoader(dataset, batch_size=32)

# Initialise the model
model = LitConvClassifier()

# Initialise the trainer
trainer = pl.Trainer(max_epochs=5)

# Train the model
trainer.fit(model, train_dataloader)