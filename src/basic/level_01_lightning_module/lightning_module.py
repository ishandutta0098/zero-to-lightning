# Documentation Link
# https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


# A simple convolution based classifier model for MNIST
class LitConvClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Define the layers for the model architecture

        # Convolutional layer with 32 filters of size 3x3
        # ReLU activation function introduces non-linearity to the model, enabling it to learn more complex patterns
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # Second convolutional layer with 64 filters of size 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        # Fully connected layers for classification
        # The input size 64*7*7 corresponds to the flattened output of the last convolutional layer
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
        # Define the training step which includes
        # the forward pass, loss calculation and backpropagation

        x, y = batch  # Unpack batch
        y_hat = self(x)  # Forward pass, get predicted logits

        # Calculate loss using cross-entropy, which is suitable for multi-class classification
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        # Define the optimizer to use for training
        # Adam is a popular choice due to its adaptive learning rate and momentum
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Load Dataset
# MNIST is a widely used dataset for handwritten digit recognition
dataset = MNIST("./", download=True, transform=transforms.ToTensor())

# Create a Dataloader with batch size of 32
# Batch size is a hyperparameter that defines the number of
# samples to work through before updating the model's weights
train_dataloader = DataLoader(dataset, batch_size=32)

# Initialise the model
model = LitConvClassifier()

# Initialise the trainer with 1 epoch
# An epoch is a complete pass through the entire training dataset
trainer = pl.Trainer(max_epochs=1)

# Train the model
trainer.fit(model, train_dataloader)
