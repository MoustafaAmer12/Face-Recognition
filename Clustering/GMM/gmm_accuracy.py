import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment

class gmm_accuracy:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels.astype(int)
        self.predicted_labels = predicted_labels.astype(int)
    
    def align_labels(self):
        # Get unique sorted true labels (1-40)
        unique_true = np.unique(self.true_labels)
        # Create confusion matrix
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        aligned_labels = np.zeros_like(self.predicted_labels)        
        for cluster, col_idx in zip(row_ind, col_ind):
            true_label = unique_true[min(col_idx,len(unique_true)-1)]
            aligned_labels[self.predicted_labels == cluster] = true_label
        
        return aligned_labels
    
    def compute_accuracy(self):
        aligned_pred = self.align_labels()
        return accuracy_score(self.true_labels, aligned_pred)


    def compute_clustering_accuracy(self,y_true, y_pred):
        D = max(y_pred.max(), y_true.max()) + 1
        confusion = np.zeros((D, D), dtype=int)
        for i in range(len(y_true)):
            confusion[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_sum_assignment(-confusion)
        correct = confusion[row_ind, col_ind].sum()
        return correct / len(y_true)

        
