class Cluster:
    def __init__(self, cluster_id, centroid):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.points = []
        self.labels = []
        self.cluster_class = None

    def add_point(self, point):
        self.points.append(point)

    def clear_points(self):
        self.points = []
        self.labels = []

    def update_centroid(self):
        if len(self.points) == 0:
            new_centroid = self.centroid
        else:
            new_centroid = [sum(coord) / len(self.points) for coord in zip(*self.points)]
        self.centroid = new_centroid

    def get_cluster_label(self):
        """
        Get the label of the cluster.

        The label is determined by the most common label
        among the points in the cluster.
        i.e majority voting

        Returns:
        --------
            label : int
                The label of the cluster.
        """
        if not self.points:
            return None
        label_counts = {}
        for label in self.labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        self.cluster_class = max(label_counts, key=label_counts.get)
        return self.cluster_class