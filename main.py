from dataset import loader
from Dimensionality_Reduction.PCA import pca
from Clustering.K_Means import kmeans

def main():
    X, y = loader.load_dataset()

    X_train, X_test, y_train, y_test =  loader.split_dataset(X, y)

    dim_red = pca.PCA(X_train, var_threshold=0.8)
    out = dim_red()
    test_out = dim_red.map_test_data(X_test)

    clusterer = kmeans.KMeans(out, y_train, k=60, max_iter=100, threshold=1e-10, random_state=42)
    clusterer()

    clusterd_out = clusterer.test(test_out)

if __name__ == "__main__":
    main()