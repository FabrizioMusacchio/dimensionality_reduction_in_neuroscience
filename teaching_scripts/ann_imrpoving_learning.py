""" 
Just a script to demonstrate how to improve learning in a neural network 
using techniques such as data augmentation, dropout, batch normalization, 
and early stopping.

author: Fabrizio Musacchio
date: Okt 27, 2024
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

# set global properties for all plots:
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% DEFINE PATHS
RESULTSPATH = '../results/teaching_material/'
# check whether the results path exists, if not, create it:
if not os.path.exists(RESULTSPATH):
    os.makedirs(RESULTSPATH)
# %% HYPERPARAMETERS
learning_rate = 0.001
batch_size = 64
epochs = 10
weight_decay = 1e-4  # L2 regularization strength
# %% DATASET & DATA AUGMENTATION
transform = transforms.Compose([
    transforms.RandomRotation(10),          # data Augmentation
    transforms.RandomHorizontalFlip(),      # data Augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))    # normalize to [-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

test_data = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
# %% MODEL DEFINITION WITH DROPOUT, BATCH NORM, AND CUSTOM WEIGHT INIT
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)   # Batch Normalization
        self.dropout1 = nn.Dropout(0.3)  # Dropout for Regularization
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 10)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # He Initialization (good for ReLU activations)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# instantiate model, loss function, and optimizer:
device= torch.device('mps')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# %% TRAINING WITH EARLY STOPPING AND TRACKING LOSSES
train_losses = []
val_losses = []
patience = 3
early_stop_count = 0
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()

    train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # validation: (no gradient calculation needed)
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

    val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # early stopping check:
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break

# plot training and validation losses:
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(RESULTSPATH + 'ann_tuning_loss_curve.png', dpi=300)
plt.show()

# %% TESTING AND PREDICTION VISUALIZATION
model.eval()
correct = 0
total = 0
predictions = []
images_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # collect a few images and predictions for visualization:
        if len(images_list) < 10:
            images_list.extend(images[:5])
            predictions.extend(predicted[:5])

print(f'Test accuracy: {100 * correct / total:.2f}%')

# plot some sample predictions:
plt.figure(figsize=(10, 5))
for i, (img, pred) in enumerate(zip(images_list, predictions)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img.cpu().numpy().squeeze(), cmap='gray')
    plt.title(f"Pred: {pred.item()}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(RESULTSPATH + 'ann_tuning_predictions.png', dpi=300)
plt.show()
# %% END