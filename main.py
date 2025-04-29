from dataset import loader
from Dimensionality_Reduction.PCA import pca
from Clustering.K_Means import kmeans
from Evaluation_Metrics import metrics as eval

def main():
    X, y = loader.load_dataset()

    X_train, X_test, y_train, y_test =  loader.split_dataset(X, y)

    
    dim_red = pca.PCA(X_train, var_threshold=0.8)
    out = dim_red()
    test_out = dim_red.map_test_data(X_test)

    clusterer = kmeans.KMeans(out, y_train, k=20, max_iter=100, threshold=1e-10, random_state=42)
    clusterer()

    clusterd_out = clusterer.test(test_out)

    eval.plot_confusion_matrix(y_test, clusterd_out, title='KMeans K=20 Clustering At Variance = 0.8', figsize=(12, 10), save_path='plots/kmeans_20_08.png')
    accuracy = eval.accuracy_score(y_test, clusterd_out)
    f1 = eval.f1_score(y_test, clusterd_out, average='macro')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()