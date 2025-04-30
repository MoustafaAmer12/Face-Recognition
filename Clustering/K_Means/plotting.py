import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

def plot_accuracy_vs_k_for_all_alphas(k_values, alpha_values, accuracy_matrix):
    """
    Plot accuracy vs K with multiple lines, one for each alpha value.

    Parameters:
    - k_values: list of K values (x-axis)
    - alpha_values: list of alpha values
    - accuracy_matrix: 2D list, rows = K values, columns = alpha values
    """
    for alpha_index, alpha in enumerate(alpha_values):
        accuracies = [row[alpha_index] for row in accuracy_matrix]
        plt.plot(k_values, accuracies, marker='o', label=f'α = {alpha}')

    plt.title('Accuracy vs K for Different Alpha Values')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    filename = 'plots/accuracy_vs_K_all_alphas.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")
