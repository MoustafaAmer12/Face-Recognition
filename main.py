import numpy as np
from dataset import loader
from Dimensionality_Reduction.PCA import pca
from Clustering.GMM.clustering_GMM import GMM
from Accuracy.GMM_Accuracy.clustering_accuracy import clustering_accuracy
from Accuracy.GMM_Accuracy.encoder import encoder

def main():
    X, y = loader.load_dataset()
    y = y.astype(int)
    X_train, X_test, y_train, y_test =  loader.split_dataset(X, y)

    dim_red = pca.PCA(X_train, var_threshold=0.8)
    X_train_PCA = dim_red()
    print("PCA Output:", X_train_PCA)
    X_test_PCA = dim_red.map_test_data(X_test)
    print("PCA Test Output:", X_test_PCA)
    """====================CLustering========================"""
    #GMM
    K = [20, 40, 60]
    X_train_PCA = np.real(X_train_PCA)


    accuracy_results = {}
    for k in K:
        gmm = GMM(k)
        gmm.fit(X_train_PCA)
        responsibilities = gmm.predict(X_train_PCA)
        print(f"for K = {k}:\n responsibility vector:\n{responsibilities}")

        #Compute accuracy
        acc = clustering_accuracy(y_train, responsibilities)
        gmm_accuracy = acc.compute_accuracy()
        accuracy_results[k] = gmm_accuracy
        print(f"Accuracy: {gmm_accuracy*100}%")

if __name__ == "__main__":
    main()