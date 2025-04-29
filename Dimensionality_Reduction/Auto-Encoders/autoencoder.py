import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import dataset.loader as loader
import matplotlib.pyplot as plt

# --- Hyperparameters ---
latent_dim = 50
batch_size = 32
epochs = 50
learning_rate = 1e-3

# --- Data Preparation ---
X, y = loader.load_dataset()
X_train, X_test, y_train, y_test = loader.split_dataset(X, y)

# Normalize data to [0, 1] range for both train and test
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert training data to PyTorch tensors
train_tensor = torch.tensor(X_train, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

# --- Autoencoder Definition ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim=10304, latent_dim=50):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Training Function ---
def train_autoencoder(model, dataloader, epochs, optimizer, criterion, device):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "autoencoder1000.pth")
    print("Model saved as autoencoder1000.pth")

# --- Visualization Function ---
def reconstruct_and_plot(model, X_test, n=10):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        reconstructed = model(inputs).cpu().numpy()

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(112, 92), cmap='gray')  # try flipped
        plt.title("Original")
        plt.axis("off")

        # Reconstructed (clipped to [0, 1] for display)
        ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(np.clip(reconstructed[i].reshape(92, 112), 0, 1), cmap='gray')
        plt.imshow(reconstructed[i].reshape(112, 92), cmap='gray')  # try flipped
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(input_dim=10304, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# --- Uncomment to train the model ---
# train_autoencoder(model, train_loader, epochs, optimizer, criterion, device)

# --- Load trained model and visualize results ---
model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
reconstruct_and_plot(model, X_test)
