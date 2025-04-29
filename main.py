from dataset import loader
from Dimensionality_Reduction.PCA import pca

def main():
    X, y = loader.load_dataset()

    X_train, X_test, y_train, y_test =  loader.split_dataset(X, y)

    dim_red = pca.PCA(X_train, var_threshold=0.8)
    out = dim_red()
    print("PCA Output:", out)
    test_out = dim_red.map_test_data(X_test)
    print("PCA Test Output:", test_out)
if __name__ == "__main__":
    main()