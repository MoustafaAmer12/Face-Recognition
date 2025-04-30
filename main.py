from Data import loader
from Dimensionality_Reduction.PCA import pca
from Clustering.K_Means import kmeans
from Evaluation_Metrics import metrics as eval
from Clustering.K_Means import plotting

def main():
    X, y = loader.load_dataset()

    X_train, X_test, y_train, y_test =  loader.split_dataset(X, y)

    K_values = [20, 40, 60]
    alpha_values = [0.8, 0.85, 0.9, 0.95]

    accuracy_matrix = []
    f1_matrix = []
    for k in K_values:
        accuracies = []
        for alpha in alpha_values:
            dim_red = pca.PCA(X_train, var_threshold=alpha)
            train_out = dim_red()
            test_out = dim_red.map_test_data(X_test)

            clusterer = kmeans.KMeans(train_out, y_train, k=k, max_iter=100, threshold=1e-10, random_state=42)
            clusterer()

            clusterd_out = clusterer.test(test_out)

            accuracy = eval.accuracy_score(y_train, clusterd_out)
            f1 = eval.f1_score(y_test, clusterd_out, average='macro')
            accuracies.append(accuracy)
            f1_matrix.append(f1)
        accuracy_matrix.append(accuracies)
        f1_matrix.append(f1)

    print(f"Accuracy Matrix: {accuracy_matrix}")
    print(f"F1 Matrix: {f1_matrix}")
    # plotting.plot_accuracy_vs_k_for_all_alphas(K_values, alpha_values, accuracy_matrix)


    # dim_red = pca.PCA(X_train, var_threshold=0.8)
    # out = dim_red()
    # test_out = dim_red.map_test_data(X_test)

    # clusterer = kmeans.KMeans(out, y_train, k=20, max_iter=100, threshold=1e-10, random_state=42)
    # clusterer()

    # clusterd_out = clusterer.test(test_out)

    # eval.plot_confusion_matrix(y_test, clusterd_out, title='KMeans K=20 Clustering At Variance = 0.8', figsize=(12, 10), save_path='plots/kmeans_20_08.png')
    # accuracy = eval.accuracy_score(y_test, clusterd_out)
    # f1 = eval.f1_score(y_test, clusterd_out, average='macro')
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()