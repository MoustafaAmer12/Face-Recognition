import os
import numpy as np
import cv2

def load_split_dataset(path = "Data"):
    """
    Loads the dataset from the specified directory and splits it.

    The function reads images from subdirectories, flattens them,
    and stores them in a matrix, with corresponding labels. Each subdirectory represents a
    different class, and the images are labeled accordingly. The
    dataset is then split into training and testing sets.

    Parameters:
    -----------
        path : str, optional
            The path to the dataset directory. Defaults to "Data".
            The directory should contain subdirectories for each class,
            and each subdirectory should contain images of that class.
            The subdirectory names should be in the format "sX", where
            X is the class label (e.g., "s1", "s2", etc.). The images
            should be in grayscale format and have the same dimensions.

    Returns:
    --------
        train_split : np.array
            Training set with shape (200, 10304)
            Contains samples with odd indices from the original dataset.
        test_split : np.array
            Testing set with shape (200, 10304)
            Contains samples with even indices from the original dataset.
        train_labels : np.array 
            Labels for the training set with shape (200,)
            Corresponding labels for the training samples.
        test_labels : np.array
            Labels for the testing set with shape (200,)
            Corresponding labels for the testing samples.
    """
    dataset = []
    labels = []
    current_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(current_dir, path)
    os.chdir(dataset_dir)

    for file in sorted(os.listdir(dataset_dir)):
        if not os.path.isdir(file):
            continue

        for img_file in os.listdir(file):
            img_path = os.path.join(file, img_file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            vec = img.flatten()

            dataset.append(vec)
            labels.append(file.split(sep='s')[1])        
    
    data_matrix = np.vstack(dataset)
    labels = np.array(labels)

    return split_dataset(data_matrix, labels)

def split_dataset(X: np.array, y: np.array):
    """
    Splits the dataset into training and testing sets.

    The function takes a dataset of 400 samples and splits it into
    training and testing sets. The training set contains samples with
    odd indices, while the testing set contains samples with even
    indices. The function returns the training and testing sets along
    with their corresponding labels.

    Parameters:   
    -----------
        X : np.array
            Data matrix with shape (400, 10304)
            Each row represents a flattened image.
            10304 is the number of pixels in the image (92x112).

        y : np.array
            Data labels with shape (400,)
            Each label corresponds to the integer class of the image.

    Returns:
    --------
        train_split : np.array
            Training set with shape (200, 10304)
            Contains samples with odd indices from the original dataset.
        test_split : np.array
            Testing set with shape (200, 10304)
            Contains samples with even indices from the original dataset.
        train_labels : np.array
            Labels for the training set with shape (200,)
            Corresponding labels for the training samples.
        test_labels : np.array
            Labels for the testing set with shape (200,)
            Corresponding labels for the testing samples.
    """
    train_split = []
    test_split = []
    train_labels = []
    test_labels = []
    for i in range(0, 400):
        if i % 2 == 0:
            test_split.append(X[i])
            test_labels.append(y[i])
        else:
            train_split.append(X[i])
            train_labels.append(y[i])

    train_split = np.array(train_split)
    test_split = np.array(test_split)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_split, test_split, train_labels, test_labels

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split_dataset()