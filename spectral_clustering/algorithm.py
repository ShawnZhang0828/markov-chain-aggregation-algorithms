import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import eig

# Ensure the parent directory is accessible or use absolute imports based on your setup
from core.base_aggregator import BaseMarkovAggregator

class SpectralAggregator(BaseMarkovAggregator):
    """
    Concrete implementation of Markov Chain aggregation using spectral clustering.
    """
    
    def __init__(self, k_macro_states: int):
        super().__init__(k_macro_states)

    def _partition_states(self, P: np.ndarray) -> np.ndarray:
        """
        Implements the abstract method using eigenvector embedding and k-means.
        """
        # Compute eigenvalues and right eigenvectors
        eigenvalues, eigenvectors = eig(P)
        
        # Sort eigenvalues by magnitude in descending order
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Extract the top k eigenvectors corresponding to metastable states
        X = np.real(eigenvectors[:, :self.k])
        
        # Row-normalize the embeddings
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = X / (norms + 1e-12)
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_normalized)
        
        return labels