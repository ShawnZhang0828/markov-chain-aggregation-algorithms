import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import eig


def spectral_partition(P: np.ndarray, k: int) -> np.ndarray:
    """
    Partitions a Markov Chain state space using spectral clustering on the transition matrix.
    """
    n = P.shape[0]

    # Compute eigenvalues and right eigenvectors
    eigenvalues, eigenvectors = eig(P)

    # Sort eigenvalues by absolute value in descending order
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Extract the top k eigenvectors
    X = np.real(eigenvectors[:, :k])

    # Row-normalize the extracted eigenvectors (standard practice in spectral clustering)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Add a small value to prevent division by zero for transient states
    X_normalized = X / (norms + 1e-12)

    # Apply k-means clustering in the embedded space
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_normalized)

    return labels


def compute_aggregated_matrix(P: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the aggregated transition matrix P_hat using a simple uniform weighting
    (or empirical averaging) within clusters.
    """
    n = P.shape[0]
    V = np.zeros((n, k))
    for i, label in enumerate(labels):
        V[i, label] = 1.0

    # Lift the transition matrix to the macro space.
    # P_hat_IJ = (1 / |I|) * sum_{i in I, j in J} P_ij
    P_hat = np.zeros((k, k))
    for i in range(k):
        states_in_i = np.where(labels == i)[0]
        if len(states_in_i) == 0:
            continue
        for j in range(k):
            states_in_j = np.where(labels == j)[0]
            # Sum transition probabilities from block I to block J, averaged over states in I
            block_sum = np.sum(P[np.ix_(states_in_i, states_in_j)])
            P_hat[i, j] = block_sum / len(states_in_i)

    return P_hat, V
