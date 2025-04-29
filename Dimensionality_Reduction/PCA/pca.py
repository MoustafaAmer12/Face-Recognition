import numpy as np
import os

class PCA:
    def __init__(self, X, var_threshold=0.95):
        """
        Initializes the PCA class with the *TRAIN* dataset and variance threshold.

        Parameters:
        ----------
            X : np.array
                The training dataset of shape (n_samples, n_features).
            var_threshold : float
                The variance threshold to determine the number of components.
                Default is 0.95, meaning 95% of the variance should be retained.
        """
        self.X = X
        self.var_threshold = var_threshold
        self.eigenvals = None
        self.eigenvectors = None
        self.num_components = None

        project_dir = os.getcwd()
        self.eigenvals_path = os.path.join(project_dir, "cache", "eigenvals.npy")
        self.eigenvectors_path = os.path.join(project_dir, "cache", "eigenvectors.npy")

    def __call__(self):
        """
        Calls the PCA class to perform PCA on the training dataset.

        The function first standardizes the dataset, then calculates
        the eigenvalues and eigenvectors. It also determines the
        number of components required to achieve the specified
        variance threshold.
        """
        self.get_principal_components()
        Z = self.standardize_data(self.X)
        Z = Z @ self.principal_eigenvectors
        return Z
    
    def map_test_data(self, X):
        """
        Maps a given dataset to the PCA space.

        The function uses the principal eigenvectors to project
        the standardized dataset into the PCA space. The mapped
        data is then returned.

        Parameters:
        ----------
            X : np.array
                The dataset either test or train to be mapped to the PCA space.
        Returns:
        --------
            mapped_data : np.array
                Mapped Dataset
        """
        Z = self.standardize_data(X)
        mapped_data = Z @ self.principal_eigenvectors
        return mapped_data
    
    def get_principal_components(self):
        """
        Returns the Principal Components of the dataset.

        The function uses the eigenvalues to calulcate the
        attained variance. The eigenvalues are then sorted in
        descending order. Then the cumulative variance is calculated
        and the number of components required to achieve the
        specified variance threshold is determined. 

        Returns:
        --------
            e : np.array
                Eigenvalues
            v : np.array
                Eigenvectors
        """
        e, v = self.load_eigenvals_eigenvectors()

        # Calculate the variance explained by each eigenvalue
        explained_variance = np.cumsum(e) / np.sum(e)
        
        self.num_components = np.searchsorted(explained_variance, self.var_threshold) + 1
        print("PCA MODULE\n\tNumber of components to retain:", self.num_components)

        self.principal_eigenvals = e[:self.num_components]
        self.principal_eigenvectors = v[:, :self.num_components]

        return self.principal_eigenvals, self.principal_eigenvectors

    def load_eigenvals_eigenvectors(self):
        """
        Loads The Eigenvalues and Eigenvectors.

        Attempts to load the eigenvalues and eigenvectors from the
        specified paths. If the files are not found or are empty,
        it calculates them using the standardize method and the
        eigenVal method. The calculated values are then saved to
        the specified paths for future use.

        Returns:
        --------
            e : np.array
                Eigenvalues
            v : np.array
                Eigenvectors
        """
        if os.path.exists(self.eigenvals_path) and os.path.exists(self.eigenvectors_path) and \
            os.path.getsize(self.eigenvals_path) > 0 and os.path.getsize(self.eigenvectors_path) > 0:
            self.eigenvals = np.load(self.eigenvals_path)
            self.eigenvectors = np.load(self.eigenvectors_path)
        else:
            print("PCA MODULE\n\tEigenvalues and Eigenvectors not found or empty, calculating them...")
            os.makedirs(os.path.dirname(self.eigenvals_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.eigenvectors_path), exist_ok=True)

            self.eigenvals, self.eigenvectors = self.calculate_eigenvals(self.X)

            np.save(self.eigenvals_path, self.eigenvals)
            np.save(self.eigenvectors_path, self.eigenvectors)
    
        print("PCA MODULE\n\tEigenvalues & Eigenvectors loaded successfully")
        return self.eigenvals, self.eigenvectors

    def calculate_eigenvals(self, X):
        """
        Calculates Eigenvalues and Eigenvectors.

        The covariance matrix is calculated using the
        standardized dataset. The eigenvalues and eigenvectors
        are then calculated using the numpy library. The
        eigenvalues are sorted in descending order and the
        corresponding eigenvectors are also sorted
        accordingly. The eigenvalues and eigenvectors are
        then returned.

        Parameters:
        ----------
            Z : np.array
                Standardized Dataset
        Returns:
        -------
            sorted_e : np.array
                Sorted Eigenvalues descendingly
            sorted_v : np.array
                Corresponding Eigenvectors to sorted eigenvalues 
        """
        Z = self.standardize_data(X)
        c = np.cov(Z, rowvar=False)

        e, v = np.linalg.eigh(c)
        
        # Sort the eigenvalues and eigenvectors in descending order
        idx = np.argsort(e)[::-1]
        sorted_e = e[idx]
        sorted_v = v[:, idx]

        return sorted_e, sorted_v
    
    @staticmethod
    def standardize_data(X):
        """
        Standardizes The Training Dataset

        Converts The pixel values to 0 mean and unity
        variance, so as to preserve the ranges across all
        features.
        Here the data range is 0-255 for each pixel value,
        however, the mean and std are calculated across all
        the pixels in the dataset. Since each pixel might have
        different distributions.

        Returns:
        --------
            Z : np.array 
                Standardized Dataset
        """
        means = X.mean(axis=0)
        std = X.std(axis=0)

        Z = (X - means) / std
        return Z

if __name__ == "__main__":
    # Example Usage
    # X, y = loader.load_dataset()

    # X_train, X_test, y_train, y_test =  loader.split_dataset(X, y)

    # dim_red = PCA(X_train, var_threshold=0.8)
    # out = dim_red()
    # print("PCA Output:", out)
    # test_out = dim_red.map_test_data(X_test)
    # print("PCA Test Output:", test_out)
    pass

# TODO Image Reconstruction