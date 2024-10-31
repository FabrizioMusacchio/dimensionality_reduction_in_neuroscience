"""
A simple script to illustrate variational autoencoder (VAE) on digits dataset (MNIST).

author: Fabrizio Musacchio
date: Oct 29, 2024

"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# for reproducibility:
torch.manual_seed(1)

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
plt.savefig(RESULTSPATH + f'VAE_MNIST_samples.png', dpi=300)
plt.show()
# %% AUTOENCODER DEFINITION
# for reproducibility:
torch.manual_seed(1)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # encoder:
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mu = nn.Linear(64, latent_dim)
        self.fc3_logvar = nn.Linear(64, latent_dim)
        
        # decoder:
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3_mu(h2), self.fc3_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc4(z))
        h4 = torch.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# set input and latent dimensions:
input_dim = 28 * 28  # flattened size of each MNIST image
latent_dim = 2       # dimension of the latent space
model = VAE(input_dim, latent_dim)

# on macOS, move the model to the MPS device
device = torch.device('mps')
# other devices: 'cpu', 'cuda'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# define the loss function:
def vae_loss(recon_x, x, mu, logvar):
    # Binary Cross-Entropy for reconstruction loss:
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    # KL Divergence:
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total loss:
    return BCE + KLD

# initialize optimizer:
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)


# train the VAE:
num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss = vae_loss(recon_images, images, mu, logvar)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    
    # validation:
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            recon_images, mu, logvar = model(images)
            loss = vae_loss(recon_images, images, mu, logvar)
            val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")


# plot loss curves:
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('VAE Loss')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTSPATH + f'VAE_loss_curves.png', dpi=300)
plt.show()

""" 
If the ELBO loss is negative but the model is learning (i.e., the loss is 
decreasing over epochs), then this behavior is expected and not a cause for 
concern. The key is to ensure that the model is learning and the loss is 
decreasing over time.
"""
# %% EXAMPLE RECONSTRUCTION
# visualize original and reconstructed images:
model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.view(images.size(0), -1).to(device)
    recon_images, _, _ = model(images)
    recon_images = recon_images.view(-1, 28, 28).cpu()  # reshape for visualization

# plot the original and reconstructed images:
n = 5  # number of images to display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original image
    plt.subplot(2, n, i + 1)
    plt.imshow(images[i].view(28, 28).cpu().numpy(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon_images[i].numpy(), cmap='gray')
    plt.title('Reconstr.')
    plt.axis('off')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'VAE_reconstruction.png', dpi=300)
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

# visualize the latent space:
latents = []
labels = []

with torch.no_grad():
    for images, lbls in test_loader:
        images = images.view(images.size(0), -1).to(device)
        mu, _ = model.encode(images)
        latents.append(mu.cpu().numpy())
        labels.append(lbls.cpu().numpy())

latents = np.concatenate(latents)
labels = np.concatenate(labels)

# plot the latent space with color-coded digits:
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Digit Label')
plt.title('Latent Space Representation (color-coded by digits)')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'VAE_latent_space.png', dpi=300)
plt.show()
# %% BETA-VAE
# define the beta-VAE:
class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=1.0):
        super(BetaVAE, self).__init__()
        self.beta = beta
        
        # encoder:
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_mu = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)
        
        # decoder:
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# define loss function with beta parameter:
def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Binary Cross-Entropy (Reconstruction) loss
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence with beta scaling
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with beta parameter
    return recon_loss + beta * kld_loss

# model and optimizer setup:
input_dim = 784  # Example for MNIST flattened input
latent_dim = 20
beta = 20  # Experiment with different values (e.g., 0.1, 0.5, 2, 4, 10)
model = BetaVAE(input_dim, latent_dim, beta=beta)
model.to(device)
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop (same as before):
epochs = 50  # Define your epochs
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        loss = beta_vae_loss(recon_batch, data, mu, logvar, beta=beta)
        
        # Backpropagation
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    
    # Validation (optional)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, input_dim).to(device)
            recon_batch, mu, logvar = model(data)
            loss = beta_vae_loss(recon_batch, data, mu, logvar, beta=beta)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# plot loss curves:
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and validation loss (β-VAE, β = {beta})')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTSPATH + f'BetaVAE_loss_curves_beta{beta}.png', dpi=300)
plt.show()


# visualize original and reconstructed images:
model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.view(images.size(0), -1).to(device)
    recon_images, _, _ = model(images)
    recon_images = recon_images.view(-1, 28, 28).cpu()  # reshape for visualization

# plot the original and reconstructed images:
n = 5  # number of images to display
plt.figure(figsize=(9, 5))
for i in range(n):
    # Original image
    plt.subplot(2, n, i + 1)
    plt.imshow(images[i].view(28, 28).cpu().numpy(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon_images[i].numpy(), cmap='gray')
    plt.title('Reconstr.')
    plt.axis('off')
plt.tight_layout()
plt.suptitle(f'β-VAE reconstruction ($β=${beta})')
plt.savefig(RESULTSPATH + f'BetaVAE_reconstruction_beta{beta}.png', dpi=300)
plt.show()


# visualize the latent space:
latents = []
labels = []

with torch.no_grad():
    for images, lbls in test_loader:
        images = images.view(images.size(0), -1).to(device)
        mu, _ = model.encode(images)
        latents.append(mu.cpu().numpy())
        labels.append(lbls.cpu().numpy())

latents = np.concatenate(latents)
labels = np.concatenate(labels)

# plot the latent space with color-coded digits:
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Digit Label')
plt.title(f'β-VAE latent space representation (color-coded by digits)\nβ = {beta}')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'betaVAE_latent_space_beta{beta}.png', dpi=300)
plt.show()

# plot the latent space in 3D with color-coded digits:
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2], c=labels, cmap='tab10', alpha=0.7)
fig.colorbar(scatter, ax=ax, label='Digit Label')
ax.set_title(f'β-VAE latent space representation (color-coded by digits)\nβ = {beta}')
ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')
ax.set_zlabel('Latent Dimension 3')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'betaVAE_latent_space_3d_beta{beta}.png', dpi=300)
plt.show()
# %% END