import matplotlib.pyplot as plt

class gmm_plotter:
    
    def plot_accuracy_vs_k(self,results):
        plt.figure(figsize=(10, 6))
        for alpha in results:
            ks = list(results[alpha].keys())
            accs = list(results[alpha].values())
            plt.plot(ks, accs, marker='o', label=f"alpha={alpha}")
        
        plt.title("Accuracy vs. K for Different α Values")
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
        