""" 
Just a script to demonstrate optimal, overfitting, and underfitting configurations in a neural network.

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
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# set device
device = torch.device('mps')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
num_epochs = 50
batch_size = 100
learning_rate = 0.001 # 0.001 for optimal and overfitting configuration, 0.01 for underfitting configuration
configurations = ['optimal', 'overfitting', 'underfitting', ]
configuration = configurations[0] # choose one of the configurations
# %% DATASET
# Transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # data augmentation (optimal configuration)
    transforms.RandomCrop(32, padding=4),   # data augmentation (optimal configuration)
    transforms.ToTensor(),                  # convert the image to a pytorch tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), # normalize the image
                         (0.2023, 0.1994, 0.2010))
])

""" 
The choice of normalization values for transforms.Normalize in the CIFAR-10 dataset is 
based on the mean and standard deviation of the pixel values across the entire dataset, 
rather than normalizing to a standard range like 0.5.
"""


# CIFAR-10 sataset:
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# dataLoader:
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# %% DEFINE CNN MODEL
# define CNN model:
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # first conv layer: 3 input channels, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)# second conv layer: 32 input channels, 64 output channels, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)                          # max pooling layer: 2x2 kernel, stride 2
        self.fc1 = nn.Linear(64 * 8 * 8, 512)                   # fully connected layer: 64*8*8 input features, 512 output features
        self.fc2 = nn.Linear(512, 10)                           # fully connected layer: 512 input features, 10 output features
        self.dropout = nn.Dropout(0.5)                          # dropout layer: 50% dropout rate

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

""" 
In image classification tasks, such as the one we are working on with CIFAR-10, a full encoder-decoder 
structure is typically not necessary. The encoder alone suffices because the goal is to extract features 
from the input image and map them directly to class probabilities. In contrast, tasks like image segmentation 
or image-to-image translation require both encoding and decoding because they involve generating new images 
from the input.
"""

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # reduced filter size
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)  # reduced fully connected layer
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        return x
# %% TRAINING
# instantiate model, loss function, and optimizer:
model = CNN() # optimal and overfitting configuration
#model = SimpleCNN() # underfitting configuration
# set the model to the device:
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001) # optimal configuration
#optimizer = optim.Adam(model.parameters(), lr=learning_rate) # overfitting configuration
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01) # underfitting configuration


# training the model:
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training loop
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # validation loop:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# plotting training and validation loss:
plt.figure(figsize=(7,5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTSPATH + f'ann_demo_{configuration}_loss.png')
plt.show()

# plotting training and validation accuracy:
plt.figure(figsize=(7,5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and validation accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTSPATH + f'ann_demo_{configuration}_acc.png')
plt.show()
# %% END 