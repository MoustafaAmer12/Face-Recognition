import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import f1_score as sk_f1_score, accuracy_score as sk_accuracy_score

def accuracy_score(y_true, y_pred):
    """Compute classification accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def f1_score(y_true, y_pred, average='macro'):
    """Compute F1-score (macro or other sklearn-compatible average types)."""
    return sk_f1_score(y_true, y_pred, average=average)

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(15, 12), save_path=None):
    """Compute and plot a confusion matrix for multi-class classification."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[label_to_index[t], label_to_index[p]] += 1

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=6)
    plt.yticks(tick_marks, labels, fontsize=6)

    # Only draw text if classes are fewer than 20 (optional)
    if len(labels) <= 20:
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=6)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()