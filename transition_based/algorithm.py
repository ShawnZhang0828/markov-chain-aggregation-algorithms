import numpy as np
from sklearn.cluster import AgglomerativeClustering

from core.base_aggregator import BaseMarkovAggregator


class TransitionBasedAggregator(BaseMarkovAggregator):
    """
    Markov chain aggregation by clustering transition distributions.

    Each micro-state i is represented by its outgoing transition distribution
    P[i, :]. States with similar one-step transition behavior are grouped
    together using agglomerative clustering with Jensen-Shannon distance.

    After the partition is found, the aggregated transition matrix is built as
    Q = U_mu P V, consistent with the other aggregators.
    """

    def __init__(self, k_macro_states: int):
        super().__init__(k_macro_states)

    @staticmethod
    def _validate_transition_matrix(P: np.ndarray) -> None:
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be a square 2D numpy array.")
        if np.any(P < -1e-12):
            raise ValueError("P must have nonnegative entries.")
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-10):
            raise ValueError("Each row of P must sum to 1.")

    def _compute_stationary_distribution(self, P: np.ndarray) -> np.ndarray:
        """
        Compute the stationary distribution mu satisfying mu^T P = mu^T.
        """
        n = P.shape[0]
        A = P.T - np.eye(n)
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0

        try:
            mu = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            mu, *_ = np.linalg.lstsq(A, b, rcond=None)

        mu = np.clip(mu, a_min=0.0, a_max=None)
        total = mu.sum()
        if total <= 0:
            raise ValueError("Failed to compute a valid stationary distribution.")
        mu /= total
        return mu

    def _compute_weighted_aggregation(
        self,
        P: np.ndarray,
        mu: np.ndarray,
        labels: np.ndarray,
        num_clusters: int,
    ) -> np.ndarray:
        """
        Compute the aggregated chain Q = U_mu P V.
        """
        Q = np.zeros((num_clusters, num_clusters), dtype=float)
        mu_hat = np.zeros(num_clusters, dtype=float)

        for c in range(num_clusters):
            mu_hat[c] = np.sum(mu[labels == c])

        for u in range(num_clusters):
            states_u = np.where(labels == u)[0]
            if states_u.size == 0 or mu_hat[u] <= 0:
                continue

            weighted_rows = mu[states_u, None] * P[states_u, :]
            for v in range(num_clusters):
                states_v = np.where(labels == v)[0]
                if states_v.size == 0:
                    continue
                joint_prob = np.sum(weighted_rows[:, states_v])
                Q[u, v] = joint_prob / mu_hat[u]

        Q = np.clip(Q, a_min=0.0, a_max=None)
        row_sums = Q.sum(axis=1, keepdims=True)
        nonzero_rows = row_sums[:, 0] > 0
        Q[nonzero_rows] /= row_sums[nonzero_rows]
        return Q

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL(p || q) with the convention 0 log(0/q) = 0.
        """
        eps = 1e-15
        p_safe = np.clip(p, eps, None)
        q_safe = np.clip(q, eps, None)
        return float(np.sum(p * np.log2(p_safe / q_safe)))

    def _jensen_shannon_distance_matrix(self, P: np.ndarray) -> np.ndarray:
        """
        Compute the pairwise Jensen-Shannon distance matrix between rows of P.
        """
        n = P.shape[0]
        D = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                p = P[i, :]
                q = P[j, :]
                m = 0.5 * (p + q)

                js_div = 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)
                js_dist = np.sqrt(max(js_div, 0.0))

                D[i, j] = js_dist
                D[j, i] = js_dist

        return D

    def _partition_states(self, P: np.ndarray) -> np.ndarray:
        """
        Partition states by clustering transition distributions P[i, :].
        """
        self._validate_transition_matrix(P)

        n = P.shape[0]
        if not (1 < self.k <= n):
            raise ValueError("k_macro_states must satisfy 1 < k_macro_states <= number of states.")

        distance_matrix = self._jensen_shannon_distance_matrix(P)

        clustering = AgglomerativeClustering(
            n_clusters=self.k,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)
        return labels

    def aggregate(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the aggregated transition matrix Q and the binary assignment matrix V.
        """
        self._validate_transition_matrix(P)

        labels = self._partition_states(P)
        n = P.shape[0]

        V = np.zeros((n, self.k), dtype=float)
        for i, label in enumerate(labels):
            V[i, label] = 1.0

        mu = self._compute_stationary_distribution(P)
        Q = self._compute_weighted_aggregation(P, mu, labels, self.k)
        return Q, V