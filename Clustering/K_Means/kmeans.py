import numpy as np
from Clustering.K_Means.cluster import Cluster

class KMeans:
    def __init__(self, X, y, k=40, max_iter=100, threshold = 1e-6, random_state=42):
        self.X = X
        self.y = y
        self.k = k

        self.max_iter = max_iter
        self.threshold = threshold
        self.random_state = random_state

        self.old_centroids = None
        self.clusters = []
        self.n_samples = len(X)
        self.n_features = len(X[0])
    
    def __call__(self):
        """
        Calls the KMeans class to perform KMeans clustering.

        The function initializes the clusters, assigns points to
        the nearest centroid, and updates the centroids. This
        process is repeated until convergence or until the maximum
        number of iterations is reached.

        Returns:
        --------
            clusters : list
                A list of Cluster objects, each representing a cluster.
        """
        self.initialize_clusters()
        
        for _ in range(self.max_iter):
            for cluster in self.clusters:
                cluster.clear_points()
            self.assign_points()
            self.update_centroids()
            if self.check_convergence():
                break
                
        for cluster in self.clusters:
            cluster.get_cluster_label()

        return self.clusters
    
    def test(self, X):
        """
        Test the KMeans clustering on a new dataset.

        The function normalizes the new dataset and assigns
        points to the nearest centroid. The function returns
        the cluster labels for the new dataset.

        Parameters:
        ----------
            X : np.array
                The new dataset to be tested.

        Returns:
        --------
            labels : list
                A list of cluster labels for the new dataset.
        """
        labels = []
        
        for i in range(len(X)):
            point = X[i]
            distances = [self.calculate_distance(point, cluster.centroid) for cluster in self.clusters]
            closest_cluster = np.argmin(distances)
            labels.append(self.clusters[closest_cluster].cluster_class)

        return labels
    
    def initialize_clusters(self):
        """
        Initializes the centroids based on KMeans++.

        The function selects the first centroid randomly from the dataset.
        The remaining centroids are selected based on the distance
        from the nearest centroid. The probability of selecting a
        point as a centroid is proportional to the squared distance
        from the nearest centroid.
        This helps to spread out the initial centroids.

        Returns:
        --------
            clusters : list
                A list of Cluster objects, each representing a cluster.
        """
        rng = np.random.default_rng(self.random_state)

        centroids = []
        first_centroid_idx = rng.integers(self.n_samples)
        centroids.append(self.X[first_centroid_idx])

        for _ in range(1, self.k):
            # Compute squared distances from nearest centroid
            distances = np.array([
                min(np.sum((x - c)**2) for c in centroids)
                for x in self.X
            ])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = rng.random()

            # Select next centroid based on probability
            next_centroid_idx = np.searchsorted(cumulative_probs, r)
            centroids.append(self.X[next_centroid_idx])

        for i in range(self.k):
            self.clusters.append(Cluster(cluster_id=i, centroid=centroids[i]))
        
        return self.clusters

    def assign_points(self):
        """
        Assign all points to the nearest centroid.

        The function iterates through all the points in the dataset
        and calculates the distance to each centroid. The point is
        assigned to the cluster with the nearest centroid. The
        points are stored in the Cluster object, and the labels
        are also stored in the Cluster object for later use.

        Returns:
        ---------
            None
        """
        for i in range(self.n_samples):
            point = self.X[i]
            label = self.y[i]
            distances = [self.calculate_distance(point, cluster.centroid) for cluster in self.clusters]
            closest_cluster = np.argmin(distances)
            self.clusters[closest_cluster].points.append(point)
            self.clusters[closest_cluster].labels.append(label)
        
    def update_centroids(self):
        """
        Update the centroids based on the assigned points.

        The function iterates through all the clusters and updates
        the centroid of each cluster based on the mean of the
        assigned points.

        Returns:
        --------
            None
        """
        self.old_centroids = []
        for cluster in self.clusters:
            self.old_centroids.append(cluster.centroid)
            cluster.update_centroid()

    def calculate_distance(self, point1, point2):
        """
        Calculate the distance between two points.

        This function uses the Euclidean distance formula.
        However, the squared distance is used to avoid the computational cost of
        taking the square root.

        Parameters:
        ----------
            point1 : list
                The first point.
            point2 : list
                The second point.
        Returns:
        --------
            distance : float
                The squared distance between the two points.
        """
        return np.sum((point1 - point2) ** 2)
    
    def check_convergence(self):
        """
        Check if the centroids have converged.

        The function checks if the centroids have changed
        significantly. If they have not changed, the function
        returns True, indicating convergence.

        Returns:
        --------
            converged : bool
                True if the centroids have converged, False otherwise.
        """
        new_centroids = [cluster.centroid for cluster in self.clusters]
        centroids_moved = [
            np.linalg.norm(np.array(new) - np.array(old))
            for new, old in zip(new_centroids, self.old_centroids)
        ]

        # Check if all movements are below a small threshold
        if all(move < self.threshold for move in centroids_moved):
            return True
            
        return False

if __name__ == "__main__":
    # Example Usage
    # X, y = loader.load_dataset()

    # X_train, X_test, y_train, y_test =  loader.split_dataset(X, y)

    # dim_red = pca.PCA(X_train, var_threshold=0.8)
    # out = dim_red()
    # test_out = dim_red.map_test_data(X_test)

    # clusterer = kmeans.KMeans(out, y_train, k=60, max_iter=100, threshold=1e-10, random_state=42)
    # clusterer()

    # clusterd_out = clusterer.test(test_out)
    pass