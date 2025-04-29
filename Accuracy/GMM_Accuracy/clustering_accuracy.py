import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment

class clustering_accuracy:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
    
    def align_labels(self):
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        aligned_labels = np.zeros_like(self.predicted_labels)
        for cluster, true_label in zip(row_ind, col_ind):
            aligned_labels[self.predicted_labels == cluster] = true_label
        return aligned_labels
    
    def compute_accuracy(self):
        aligned_pred = self.align_labels()
        return accuracy_score(self.true_labels, aligned_pred)

        
