import argparse
import numpy as np
from Data import loader
from Dimensionality_Reduction.PCA import pca
from Dimensionality_Reduction.AutoEncoders.autoencoder import AutoencoderTrainer
from Clustering.K_Means import kmeans as kmeans_module
from Evaluation_Metrics import metrics, reconstructed_images
from Clustering.K_Means import plotting


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
        return gmm_module.GMM(X_train, y_train, n_components=k)
    else:
        raise ValueError(f"Unknown clustering method '{method}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduction", choices=["pca", "autoencoder"], required=True, help="Dimensionality reduction method")
    parser.add_argument("--cluster", choices=["kmeans", "gmm"], required=True, help="Clustering algorithm")
    args = parser.parse_args()

    X, y = loader.load_dataset()
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

            # Uncomment to plot reconstructed images for Autoencoder or PCA
            # if k == 20 and alpha == 0.9:
                # reconstructed_test = reducer.reconstruct(test_out) if args.reduction == "pca" else trainer.model(inputs).cpu().numpy()
                # reconstructed_images.plot_reconstructed_images(X_test, reconstructed_test)

            clusterer = get_clusterer(args.cluster, train_out, y_train, k)
            clusterer()
            clustered_out = clusterer.test(test_out)

            accuracy = metrics.accuracy_score(y_train, clustered_out)
            f1 = metrics.f1_score(y_test, clustered_out, average='macro')

            accuracies.append(accuracy)
            f1_matrix.append(f1)

        accuracy_matrix.append(accuracies)

    print(f"Accuracy Matrix: {accuracy_matrix}")
    print(f"F1 Matrix: {f1_matrix}")
    plotting.plot_accuracy_vs_k_for_all_alphas(K_values, alpha_values, accuracy_matrix)

    # Final run for confusion matrix plot
    final_reducer, transform = get_dim_reducer(args.reduction, X_train, alpha=0.8)
    train_out = final_reducer(X_train) if args.reduction == "autoencoder" else final_reducer()
    test_out = transform(X_test)

    clusterer = get_clusterer(args.cluster, train_out, y_train, k=60)
    clusterer()
    clustered_out = clusterer.test(test_out)

    metrics.plot_confusion_matrix(
        y_test, clustered_out,
        title=f"{args.cluster.upper()} Clustering at K=60, Var=0.8",
        figsize=(12, 10),
        save_path=f"plots/{args.cluster}_60_08.png"
    )

    print(f"Final Accuracy: {metrics.accuracy_score(y_test, clustered_out):.4f}")
    print(f"Final F1 Score: {metrics.f1_score(y_test, clustered_out, average='macro'):.4f}")


if __name__ == "__main__":
    main()
