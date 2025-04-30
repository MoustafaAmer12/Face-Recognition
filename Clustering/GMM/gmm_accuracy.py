import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment

class gmm_accuracy:
    
    def compute_clustering_accuracy(self,y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(y_true == y_pred)

        
