import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import Data.loader as loader
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, input_dim=10304, latent_dim=128):
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
    
    def get_latent_representation(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded



class AutoencoderTrainer:
    def __init__(self, input_dim=10304, latent_dim=50, lr=1e-3, batch_size=32, epochs=50, model_path="autoencoder.pth", standardize=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.standardize = standardize

    def __call__(self, X):
        """
        Returns the latent representation of input X, mimicking PCA behavior.
        Standardizes the input if self.standardize is True.
        """
        if self.standardize:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        else:
            X = (X - np.mean(X)) / np.std(X)

        return self.get_latent_representation(X)

    def prepare_data(self, X=None, y=None):
        if X is None or y is None:
            X, y = loader.load_dataset()

        X_train, X_test, y_train, y_test = loader.split_dataset(X, y)

        # Feature-wise standardization
        if self.standardize:
            X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
            X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
        else:
            mean = X_train.mean()
            std = X_train.std()
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.train_loader = DataLoader(TensorDataset(train_tensor), batch_size=self.batch_size, shuffle=True)
        self.X_test = X_test 
        self.X_train = X_train

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                x = batch[0].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, x)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        self.save_model()
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved as {self.model_path}")
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print(f"Model loaded from {self.model_path}")
    def reconstruct(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            reconstructed = self.model(inputs).cpu().numpy()
        return reconstructed
    
    def plot_reconstruction(self, n=10, image_shape=(112, 92)):
        reconstructed = self.reconstruct(self.X_test)

        plt.figure(figsize=(20, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.X_test[i*5].reshape(image_shape), cmap='gray')
            plt.title("Original")
            plt.axis("off")

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed[i*5].reshape(image_shape), cmap='gray')
            plt.title("Reconstructed")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def cluster_latent_space(self, X_test, n_clusters=3, method='kmeans'):
        latent_features = self.model.get_latent_representation(torch.tensor(X_test, dtype=torch.float32).to(self.device)).cpu().numpy()

        if method == 'kmeans':
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'gmm':
            clustering_model = GaussianMixture(n_components=n_clusters, random_state=42)
        
        cluster_labels = clustering_model.fit_predict(latent_features)
        
        return cluster_labels
    
    def get_latent_representation(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            latent_representation = self.model.get_latent_representation(inputs).cpu().numpy()
        return latent_representation



if __name__ == "__main__":
    trainer = AutoencoderTrainer()
    trainer.prepare_data()

    # Train if needed
    # trainer.train()

    # Load and visualize
    trainer.load_model()
    trainer.plot_reconstruction()

    # Cluster the latent space using KMeans or GMM
    # cluster_labels = trainer.cluster_latent_space(trainer.X_test, n_clusters=3, method='kmeans')

    # Print or visualize cluster labels
    # print(f"Cluster labels: {cluster_labels}")
