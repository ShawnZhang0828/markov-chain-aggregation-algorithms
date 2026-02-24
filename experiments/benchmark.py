import numpy as np
import time
from scipy.linalg import eig

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spectral_clustering.algorithm import spectral_partition, compute_aggregated_matrix


def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Computes the stationary distribution of a transition matrix P.
    """
    n = P.shape[0]

    # Construct the base linear system A = P^T - I
    A = P.T - np.eye(n)

    # Enforce the probability mass constraint: sum(pi) = 1
    # We replace the last row of the singular matrix A with a row of ones
    A[-1, :] = 1.0

    # Construct the right-hand side vector b
    # The last element is 1.0 corresponding to the sum(pi) = 1 constraint
    b = np.zeros(n)
    b[-1] = 1.0

    # Solve the linear system
    pi = np.linalg.solve(A, b)

    return pi


def calculate_mutual_information(P: np.ndarray, pi: np.ndarray) -> float:
    """
    Calculates the mutual information I(X_t; X_{t+1}) for a Markov chain.

    Note:
    P_ij * log(P_ij / pi_j) requires the information theory convention 0 * log(0) = 0.
    """
    n = P.shape[0]
    mi = 0.0
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-12 and pi[j] > 1e-12:
                mi += pi[i] * P[i, j] * np.log2(P[i, j] / pi[j])
    return mi


def evaluate_information_preservation(P: np.ndarray, P_hat: np.ndarray) -> dict:
    """
    Evaluates the information preservation metrics strictly defining
    mutual information and information loss (KL divergence rate).
    """
    # Compute stationary distributions
    pi = compute_stationary_distribution(P)
    pi_hat = compute_stationary_distribution(P_hat)

    # Compute mutual information for both original and aggregated chains
    mi_original = calculate_mutual_information(P, pi)
    mi_aggregated = calculate_mutual_information(P_hat, pi_hat)

    # Compute information loss (KL divergence rate)
    information_loss = mi_original - mi_aggregated

    return {
        "mi_original": mi_original,
        "mi_aggregated": mi_aggregated,
        "information_loss": information_loss,
    }


def generate_metastable_markov_chain(n: int, k: int, noise: float = 0.1) -> np.ndarray:
    """
    Generates a block-diagonal transition matrix with added structural noise
    to simulate local metastable basins.
    """
    P = np.random.rand(n, n) * noise
    block_size = n // k

    for i in range(k):
        start = i * block_size
        end = (i + 1) * block_size if i < k - 1 else n
        P[start:end, start:end] += np.random.rand(end - start, end - start) * 10.0

    row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums
    return P


if __name__ == "__main__":
    N_STATES = 50
    K_MACRO = 5

    print("Generating strictly structured metastable transition matrix...")
    P = generate_metastable_markov_chain(n=N_STATES, k=K_MACRO, noise=0.05)

    print("Executing spectral clustering partition and aggregation...")

    # Runtime measurement using the highest resolution performance counter
    start_time = time.perf_counter()

    labels = spectral_partition(P, k=K_MACRO)
    P_hat, V = compute_aggregated_matrix(P, labels, k=K_MACRO)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print("Evaluating information preservation metrics...")
    metrics = evaluate_information_preservation(P, P_hat)

    print("\n--- Benchmark Results ---")
    print(f"Algorithm Runtime:          {execution_time:.6f} seconds")
    print(f"Original Mutual Info:       {metrics['mi_original']:.6f} bits")
    print(f"Aggregated Mutual Info:     {metrics['mi_aggregated']:.6f} bits")
    print(f"Information Loss (KL Rate): {metrics['information_loss']:.6f} bits")
