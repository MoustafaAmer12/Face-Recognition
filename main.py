import argparse
import numpy as np
from Data import loader
#PCA
from Dimensionality_Reduction.PCA import pca
from Dimensionality_Reduction.AutoEncoders.autoencoder import AutoencoderTrainer
#Clustering
from Clustering.GMM.clustering_GMM import GMM
from Clustering.GMM.gmm_accuracy import gmm_accuracy
from Clustering.GMM.gmm_plotter import gmm_plotter
from Clustering.K_Means import kmeans as kmeans_module
#Evaluation and plots
from Evaluation_Metrics import metrics as eval
from Clustering.K_Means import plotting
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


def get_dim_reducer(method, X_train, alpha=0.9):
    if method == "pca":
        reducer = pca.PCA(X_train, var_threshold=alpha)
        reducer()
        return reducer, reducer.map_test_data
    elif method == "autoencoder":
        trainer = AutoencoderTrainer()
        trainer.prepare_data()
        try:
            trainer.load_model()
        except FileNotFoundError:
            trainer.train()
        return trainer, lambda X: trainer(X)
    else:
        raise ValueError(f"Unknown reduction method '{method}'")


def get_clusterer(method, X_train, y_train, k):
    if method == "kmeans":
        return kmeans_module.KMeans(X_train, y_train, k=k, max_iter=100, threshold=1e-10, random_state=42)
    elif method == "gmm":
        gmm_component = GMM(X=X_train, n_components=k)
        #gmm_component.fit(X_train)
        return gmm_component
    else:
        raise ValueError(f"Unknown clustering method '{method}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduction", choices=["pca", "autoencoder"], required=True, help="Dimensionality reduction method")
    parser.add_argument("--cluster", choices=["kmeans", "gmm"], required=True, help="Clustering algorithm")
    args = parser.parse_args()

    X, y = loader.load_dataset()
    y = y.astype(int)
    X_train, X_test, y_train, y_test = loader.split_dataset(X, y)

    K_values = [20, 40, 60]
    alpha_values = [0.8, 0.85, 0.9, 0.95]

    accuracy_matrix = []
    f1_matrix = []

    for k in K_values:
        accuracies = []
        for alpha in alpha_values:
            reducer, transform = get_dim_reducer(args.reduction, X_train, alpha=alpha)
            train_out = reducer(X_train) if args.reduction == "autoencoder" else reducer()
            test_out = transform(X_test)
            clusterer = get_clusterer(args.cluster, train_out, y_train, k)
            clusterer()
            clustered_out = clusterer.test(test_out)
            clustered_train = clusterer.test(train_out)

            
            #accuracy of training data
            accuracy = 0
            if(args.cluster == "gmm"):
                accuracy = gmm_accuracy().compute_clustering_accuracy(y_train, clustered_train)
            else:
                accuracy = eval.accuracy_score(y_train, clustered_train)
            
            accuracies.append(accuracy)

        accuracy_matrix.append(accuracies)

    print(f"Accuracy Matrix: {accuracy_matrix}")
    plotting.plot_accuracy_vs_k_for_all_alphas(K_values, alpha_values, accuracy_matrix)

    # Final run for confusion matrix plot
    final_reducer, transform = get_dim_reducer(args.reduction, X_train, alpha=0.8)
    train_out = final_reducer(X_train) if args.reduction == "autoencoder" else final_reducer()
    test_out = transform(X_test)

    clusterer = get_clusterer(args.cluster, train_out, y_train, k=60)
    clusterer()
    clustered_out = clusterer.test(test_out)

    eval.plot_confusion_matrix(
        y_test, clustered_out,
        title=f"{args.cluster.upper()} Clustering at K=60, Var=0.8",
        figsize=(12, 10),
        save_path=f"plots/{args.cluster}_60_08.png"
    )
    if args.cluster == "gmm":
        print(f"Final Accuracy: {gmm_accuracy().compute_clustering_accuracy(y_test, clustered_out):.4f}")
    else:
        print(f"Final Accuracy: {eval.accuracy_score(y_test, clustered_out):.4f}")
    print(f"Final F1 Score: {eval.f1_score(y_test, clustered_out, average='macro'):.4f}")


if __name__ == "__main__":
    main()
