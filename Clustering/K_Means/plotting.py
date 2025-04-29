import matplotlib.pyplot as plt

def plot_accuracy_for_k(k_index, k_values, alpha_values, accuracy_matrix):
    """
    Plot accuracy vs alpha for a specific K value.

    Parameters:
    - k_index: index of the desired K value in k_values list
    - k_values: list of K values
    - alpha_values: list of alpha values
    - accuracy_matrix: 2D list where rows = K values, columns = alpha values
    """
    accuracies = accuracy_matrix[k_index]
    k_value = k_values[k_index]

    plt.plot(alpha_values, accuracies, marker='o', linestyle='-')
    plt.title(f'Accuracy vs Alpha (K = {k_value})')
    plt.xlabel('Alpha Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()


def plot_accuracy_for_alpha(alpha_index, k_values, alpha_values, accuracy_matrix):
    """
    Plot accuracy vs K for a specific alpha value.

    Parameters:
    - alpha_index: index of the desired alpha value in alpha_values list
    - k_values: list of K values
    - alpha_values: list of alpha values
    - accuracy_matrix: 2D list where rows = K values, columns = alpha values
    """
    accuracies = [row[alpha_index] for row in accuracy_matrix]
    alpha_value = alpha_values[alpha_index]

    plt.plot(k_values, accuracies, marker='o', linestyle='-')
    plt.title(f'Accuracy vs K (Alpha = {alpha_value})')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()