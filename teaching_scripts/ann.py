""" 
Just a script to plot various activation functions.

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
from torchvision import datasets, transforms
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
# %% LOAD THE DATA
# load the MNIST dataset:
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

""" 
`transform` converts images to tensors (for PyTorch). We additionally 
normalize the pixel values to the range [0, 1] and split the data into 
training and testing sets.
"""
# %% SIMPLE FNN EXAMPLE

# define the network architecture:
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Flattened input (28x28 image to a vector)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes for MNIST digits

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for the final layer (raw scores)
        return x


# initialize the network, loss function, and optimizer:
model = FeedforwardNN()
# put the model on Apple Silicon's GPU (MPS) if available:
device = torch.device('mps')
model.to(device)
# otherwise, it will be on CUDA or the CPU:
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop:
epochs = 5
for epoch in range(epochs):  # Train for 5 epochs
    for images, labels in train_loader:
        optimizer.zero_grad()           # clear the gradients
        images = images.to(device)      # move images to the same device as the model
        outputs = model(images)         # forward pass
        labels = labels.to(device)      # move labels to the same device as the model
        loss = criterion(outputs, labels) # compute the loss
        loss.backward()                 # backward pass
        optimizer.step()                # update weights

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# test the network:
correct = 0
total = 0
with torch.no_grad():  # No need to calculate gradients during testing
    for images, labels in test_loader:
        images = images.to(device)  # Move images to the same device as the model
        labels = labels.to(device)  # Move labels to the same device as the model
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the test images: {100 * correct / total:.2f}%")

""" 
While the accuracy of the network on the test images is high (97.57%), the network is not very deep or complex and 
thus may not be able to learn more complex patterns in the data or even generalize well to unseen data.
"""

# %% CNN EXAMPLE
# define the CNN architecture:
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (grayscale), 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 output channels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # flattened from 7x7 feature map with 64 channels
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST digits

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)  # down-sample by 2x2
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)  # down-sample by 2x2
        x = x.view(-1, 64 * 7 * 7)     # flatten the feature map
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                # raw scores
        return x

# initialize the CNN, loss function, and optimizer:
cnn_model = SimpleCNN()
# put the model on Apple Silicon's GPU (MPS) if available:
device = torch.device('mps')
cnn_model.to(device)
# otherwise, it will be on CUDA or the CPU:
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# training loop:
epochs = 5
for epoch in range(epochs):  # Train for 5 epochs
    for images, labels in train_loader:
        cnn_optimizer.zero_grad()       # clear the gradients
        images = images.to(device)      # move images to the same device as the model
        outputs = cnn_model(images)     # forward pass
        labels = labels.to(device)      # move labels to the same device as the model
        loss = cnn_criterion(outputs, labels) # compute the loss
        loss.backward()                 # backward pass
        cnn_optimizer.step()            # update weights

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# test the CNN:
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the CNN on the test images: {100 * correct / total:.2f}%")
# %% CNN WITH VISUAL INSPECTION AND LOSS PLOTS
# load and split the MNIST dataset, this time with a validation set:
transform = transforms.Compose([transforms.ToTensor()])
train_data_full = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
train_size = int(0.8 * len(train_data_full))   # 80% for training
val_size = len(train_data_full) - train_size   # 20% for validation
train_data, val_data = random_split(train_data_full, [train_size, val_size])
test_data = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# define the CNN architecture:
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (grayscale), 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 output channels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # flattened from 7x7 feature map with 64 channels
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST digits

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)  # down-sample by 2x2
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)  # down-sample by 2x2
        x = x.view(-1, 64 * 7 * 7)     # flatten the feature map
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                 # raw scores
        return x


# initialize the CNN, loss function, and optimizer:
cnn_model = SimpleCNN().to('mps')  # Adapt device as needed (e.g., 'cuda' or 'cpu')
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# training loop with validation loss tracking:
epochs = 5
train_losses = []
val_losses = []

for epoch in range(epochs):
    cnn_model.train()  # Set model to training mode
    epoch_train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to('mps'), labels.to('mps')
        cnn_optimizer.zero_grad()       # clear the gradients
        outputs = cnn_model(images)     # forward pass
        loss = cnn_criterion(outputs, labels)  # compute the loss
        loss.backward()                 # backward pass
        cnn_optimizer.step()            # update weights
        epoch_train_loss += loss.item()
    
    # calculate and store average training loss for the epoch:
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # validation phase:
    cnn_model.eval()  # set model to evaluation mode
    epoch_val_loss = 0
    with torch.no_grad():  # disable gradient calculations for validation (!)
        for images, labels in val_loader:
            images, labels = images.to('mps'), labels.to('mps')
            outputs = cnn_model(images)
            loss = cnn_criterion(outputs, labels)
            epoch_val_loss += loss.item()
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# plot the training and validation loss over epochs:
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', label="Training loss")
plt.plot(range(1, epochs + 1), val_losses, marker='o', label="Validation loss")
plt.title("Training and validation loss over epochs")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTSPATH + 'cnn_training_validation_loss.png', dpi=300)
plt.show()

# test the CNN and visualize some predictions:
correct = 0
total = 0
predictions = []
images_list = []

# collect some images and their predictions for visualization:
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('mps'), labels.to('mps')
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect a few images and predictions for visualization
        if len(images_list) < 10:
            images_list.extend(images[:5])
            predictions.extend(predicted[:5])

print(f"Accuracy of the CNN on the test images: {100 * correct / total:.2f}%")

# visualize a few test images along with predictions:
plt.figure(figsize=(10, 5))
for i, (img, pred) in enumerate(zip(images_list, predictions)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img.to('cpu').numpy().squeeze(), cmap='gray')
    plt.title(f"Predicted: {pred.item()}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(RESULTSPATH + 'cnn_predictions.png', dpi=300)
plt.show()
# %% END