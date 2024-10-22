"""
A simple script to illustrate autoencoder on digits dataset (MNIST).

author: Fabrizio Musacchio
date: Oct 18, 2024

"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

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
# %% DOWNLOAD AND PREPARE THE MNIST DATASET
# download and prepare the MNIST dataset:
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
""" 
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0–9). 
The dataset is already pre-split into a training set and a test set by default when downloaded from 
torchvision.datasets.

Each image is converted into a tensor and normalized using the transforms.Compose function.
transforms.Normalize((0.5,), (0.5,)): Normalizes the image pixel values to be in the range [-1, 1], 
with a mean of 0.5 and standard deviation of 0.5. This helps with training by standardizing the inputs.
"""

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True)
""" 
After loading the datasets, they are passed to the PyTorch DataLoader to create iterable batches of data.
"""

# plot a few examples from the dataset in a 9x9 grid:
plt.figure(figsize=(6, 6))
for i in range(9):
    for j in range(9):
        plt.subplot(9, 9, i*9+j+1)
        plt.imshow(train_data[i*9+j][0].view(28, 28), cmap='gray')
        plt.axis('off')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'AE_MNIST_samples.png', dpi=300)
plt.show()

# %% AUTOENCODER DEFINITION
# for reproducibility:
torch.manual_seed(1)

# autoencoder definition:
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 2)  # 2 dimensions for the latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# instantiate the model, loss function, and optimizer:
model = Autoencoder()

# on macOS, move the model to the MPS device:
device = torch.device('mps')
model = model.to(device)
learning_rate = 1e-3
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training the autoencoder:
num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(device)
        optimizer.zero_grad()
        encoded, decoded = model(images)
        loss = criterion(decoded, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            encoded, decoded = model(images)
            loss = criterion(decoded, images)
            val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# plot loss curves:
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTSPATH + f'AE_loss_curves.png', dpi=300)
plt.show()
# %% EXAMPLE RECONSTRUCTION

# Select a batch of test images
model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))  # Get a batch of test images
    images = images.view(images.size(0), -1).to(device)
    encoded, decoded = model(images)
    decoded = decoded.view(decoded.size(0), 28, 28).cpu()  # Reshape the decoded image to 28x28

# Plot the first image and its reconstruction
n = 1  # You can change this to visualize a different image in the batch

plt.figure(figsize=(6, 3))

# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(images[n].view(28, 28).cpu().numpy(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(decoded[n].cpu().numpy(), cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'AE_reconstruction.png', dpi=300)
plt.show()
# %% VISUALIZING THE LATENT SPACE
""" 
After training, we switch the model into evaluation mode using model.eval() and disable gradient 
computations with torch.no_grad(). This reduces memory and computation overhead, which is useful 
when we're just encoding data without needing backpropagation. This step ensures we're focusing 
on inference without extra overhead from gradients.

Storing the latent space at every step during training can increase memory usage and complexity 
since training involves many iterations (forward/backward passes) where gradients are computed.
Thus, we decided to store the latent space representation only after training is complete.
"""

# visualizing the latent space:
model.eval()
latents = []
labels = []

# we need to disable gradient computation for this step:
with torch.no_grad():
    for images, lbls in test_loader:
        images = images.view(images.size(0), -1).to(device)
        encoded, _ = model(images)
        latents.append(encoded.cpu().numpy())
        labels.append(lbls.cpu().numpy())

latents = np.concatenate(latents)
labels = np.concatenate(labels)

# plot the latent space, color-coded by digit labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Digit Label')
plt.title('Latent space representation (color-coded by digits)')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'AE_latent_space.png', dpi=300)
plt.show()
# %% END