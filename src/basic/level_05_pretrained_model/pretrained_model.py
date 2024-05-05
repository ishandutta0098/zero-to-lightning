# Documentation Link
# https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html

import os

import lightning.pytorch as pl
import torch
import torchvision.models as models
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


# Define the Lightning Module
class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Data preparation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalization for Imagenet data
    ]
)

train_dataset = datasets.CIFAR10(
    root="./", train=True, transform=transform, download=True
)
# Use a subset of the training data for demonstration purposes
train_dataset = torch.utils.data.Subset(train_dataset, indices=list(range(100)))

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

# Training
model = ImagenetTransferLearning()
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_dataloader)

# Save the model checkpoint
trainer.save_checkpoint("example_model.ckpt")

# Inference
loaded_model = ImagenetTransferLearning.load_from_checkpoint("example_model.ckpt")
loaded_model.freeze()

# Load some CIFAR10 images for prediction (assuming you're using the same transform as above)
test_dataset = datasets.CIFAR10(root="./", train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=5
)  # Loading 5 images for demonstration
some_images_from_cifar10, _ = next(iter(test_dataloader))

predictions = loaded_model(some_images_from_cifar10)
print(predictions.argmax(dim=1))
