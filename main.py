from dataset import loader
from Clustering.GMM.clustering_GMM import GMM

def main():
    X_train, X_test, y_train, y_test =  loader.load_split_dataset()
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing labels shape:", y_test.shape)
    number_of_clusters = [20,40,60]
    #GMM
    for K in number_of_clusters:
        gmm = GMM(n_components = K)
        gmm.fit(X_train)
        train_labels_pred = gmm.predict(X_train)
        print(f"Trained labels for k={K}")

if __name__ == "__main__":
    main()