import numpy as np

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4, reg_covar=1e-3):
        """
        Gaussian Mixture Model (GMM) implemented via Expectation-Maximization (EM).
        
        Parameters:
            n_components (int): Number of Gaussian components (clusters).
            max_iter (int): Maximum number of EM iterations.
            tol (float): Convergence threshold for log-likelihood change.
            reg_covar (float): Regularization term for covariance matrices.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.weights_ = None    # Mixing weights (pi_k)
        self.means_ = None      # Means of Gaussians (mu_k)
        self.covariances_ = None  # Covariance matrices (Sigma_k)
        self.log_likelihood_ = []  # Log-likelihood history

    def _initialize(self, X):
        """Initialize GMM parameters using K-means or random assignment."""
        n_samples, n_features = X.shape #200, 10304
        #Initializing equal weights = 1/k => vector of weights = <1/k, 1/k, ...>
        self.weights_ = np.ones(self.n_components) / self.n_components
        # Initialize means using random samples
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        # Initialize covariances as identity matrices
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])

    def _compute_log_responsibilities(self, X):
        """E-step: Compute log responsibilities using current parameters."""
        n_samples, n_features = X.shape
        
        log_resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Add regularization to covariance matrix
            sigma = self.covariances_[k] + self.reg_covar * np.eye(n_features)
            inv_sigma = np.linalg.inv(sigma)
            det_sigma = np.linalg.det(sigma)
            
            # Compute Mahalanobis distance: (x - mu)^T Sigma^{-1} (x - mu)
            diff = X - self.means_[k]
            quadratic = np.sum((diff @ inv_sigma) * diff, axis=1)
            
            # Log of Gaussian PDF
            log_pdf = -0.5 * (n_features * np.log(2 * np.pi) + np.log(det_sigma) + quadratic)
            log_resp[:, k] = np.log(self.weights_[k]) + log_pdf
        
        return log_resp

    def fit(self, X):
        """Fit GMM to data using EM algorithm."""
        self._initialize(X)
        n_samples, n_features = X.shape
        prev_log_likelihood = -np.inf
        
        for _ in range(self.max_iter):
            # E-step: Compute responsibilities
            log_resp = self._compute_log_responsibilities(X)
            
            # Compute log-likelihood and check convergence
            max_log = np.max(log_resp, axis=1, keepdims=True)
            log_sum = max_log + np.log(np.sum(np.exp(log_resp - max_log), axis=1, keepdims=True))
            log_likelihood = np.sum(log_sum)
            self.log_likelihood_.append(log_likelihood)
            
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood
            
            # M-step: Update parameters
            resp = np.exp(log_resp - log_sum)  # Normalized responsibilities
            
            # Update weights
            Nk = np.sum(resp, axis=0)
            self.weights_ = Nk / n_samples
            
            # Update means
            self.means_ = (resp.T @ X) / Nk[:, None]
            
            # Update covariances
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff = resp[:, k, None] * diff
                self.covariances_[k] = (weighted_diff.T @ diff) / Nk[k] + self.reg_covar * np.eye(n_features)
        
        return self

    def predict(self, X):
        """Predict cluster assignments (hard labels)."""
        log_resp = self._compute_log_responsibilities(X)
        #For each sample, we find by majority voting the cluster to which this point belongs
        return np.argmax(log_resp, axis=1)
