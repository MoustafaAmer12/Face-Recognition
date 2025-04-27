from dataset import loader

def main():
    X_train, X_test, y_train, y_test =  loader.load_split_dataset()
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing labels shape:", y_test.shape)

if __name__ == "__main__":
    main()