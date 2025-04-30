import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment

class gmm_accuracy:
    
    def compute_clustering_accuracy(self,y_true, y_pred):
        D = max(y_pred.max(), y_true.max()) + 1
        confusion = np.zeros((D, D), dtype=int)
        for i in range(len(y_true)):
            confusion[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_sum_assignment(-confusion)
        correct = confusion[row_ind, col_ind].sum()
        return correct / len(y_true)

        
