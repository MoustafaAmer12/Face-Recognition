import numpy as np
from dataset import loader
from Dimensionality_Reduction.PCA import pca
from Clustering.GMM.clustering_GMM import GMM
from Clustering.GMM.gmm_accuracy import gmm_accuracy
from Clustering.GMM.gmm_plotter import gmm_plotter
from Evaluation_Metrics import metrics as eval


def main():
    X, y = loader.load_dataset()
    y = y.astype(int)
    X_train, X_test, y_train, y_test =  loader.split_dataset(X, y)

    alphas = [0.8, 0.85, 0.9, 0.95]
    K = [20, 40, 60]
    print(y_train)

    accuracy_results = {}
    for alpha in alphas:
        dim_red = pca.PCA(X_train, var_threshold=alpha)
        X_train_PCA = dim_red()
        X_test_PCA = dim_red.map_test_data(X_test)
        X_train_PCA = np.real(X_train_PCA)
        accuracy_results[alpha] = {}
        print(f"Alpha: {alpha}")
        #GMM
        for k in K:
            gmm = GMM(k)
            gmm.fit(X_train_PCA)
            responsibilities = gmm.predict(X_train_PCA)
            print(f"for K = {k}:\n responsibility vector:\n{responsibilities}")

            #Compute accuracy
            acc = gmm_accuracy()
            acc_value = acc.compute_clustering_accuracy(y_train, responsibilities)
            accuracy_results[alpha][k] = acc_value
            print(f"Accuracy: {acc_value*100}%")
            
            print("----Testing----")
            eval.plot_confusion_matrix(y_test, responsibilities, title=f'Confusion Matrix (α={alpha}, K={k})')
            accuracy = acc.compute_clustering_accuracy(y_test, responsibilities)
            f1 = eval.f1_score(y_test, responsibilities, average='macro')
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
    #plot GMM
    plotter = gmm_plotter()
    plotter.plot_accuracy_vs_k(results=accuracy_results)


if __name__ == "__main__":
    main()