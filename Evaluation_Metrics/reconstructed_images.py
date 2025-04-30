import matplotlib.pyplot as plt

def plot_reconstructed_images(X_original, X_reconstructed, image_shape=(112, 92), n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_original[i * 5].reshape(image_shape), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(X_reconstructed[i * 5].reshape(image_shape), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    plt.tight_layout()
    plt.show()