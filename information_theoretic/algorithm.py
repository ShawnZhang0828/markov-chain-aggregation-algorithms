import numpy as np

from core.base_aggregator import BaseMarkovAggregator


class InformationTheoreticAggregator(BaseMarkovAggregator):
    """
    Markov chain aggregation using the Agglomerative Information Bottleneck
    relaxation described by Geiger et al.

    The clustering objective follows Section VII of the paper:
        g_IB = arg min_g L_{X_n}(X_{n-1} -> Y_{g,n-1})
    which is equivalent to maximizing I(X_n ; Y_{g,n-1}) for deterministic
    partitions g, because I(X_n ; X_{n-1}) is constant with respect to g.

    After the partition is found, the aggregated transition matrix is built as
    Q = U_mu P V (Lemma 3 / Eq. (21)-(22) in the paper).
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
        Compute the optimal Markov approximation Q = U_mu P V.
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

    def _cluster_masses(self, mu: np.ndarray, labels: np.ndarray, num_clusters: int) -> np.ndarray:
        masses = np.zeros(num_clusters, dtype=float)
        for c in range(num_clusters):
            masses[c] = np.sum(mu[labels == c])
        return masses

    def _cluster_next_state_distributions(
        self,
        P: np.ndarray,
        mu: np.ndarray,
        labels: np.ndarray,
        num_clusters: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For each cluster y, compute p(X_n | Y_{g,n-1}=y).

        Since Y_{g,n-1} = g(X_{n-1}), this is the mu-weighted average of the
        transition rows P[i, :] over all states i assigned to cluster y.
        """
        masses = self._cluster_masses(mu, labels, num_clusters)
        cond = np.zeros((num_clusters, P.shape[1]), dtype=float)

        for c in range(num_clusters):
            states = np.where(labels == c)[0]
            if states.size == 0 or masses[c] <= 0:
                continue
            cond[c, :] = np.sum(mu[states, None] * P[states, :], axis=0) / masses[c]

        cond = np.clip(cond, a_min=0.0, a_max=None)
        row_sums = cond.sum(axis=1, keepdims=True)
        valid_rows = row_sums[:, 0] > 0
        cond[valid_rows] /= row_sums[valid_rows]
        return masses, cond

    def _calculate_ib_mutual_information(
        self,
        P: np.ndarray,
        mu: np.ndarray,
        labels: np.ndarray,
        num_clusters: int,
    ) -> float:
        """
        Compute I(X_n ; Y_{g,n-1}).

        Here:
          - p(X_{n-1}=i) = mu[i]
          - p(X_n=j | X_{n-1}=i) = P[i, j]
          - p(X_n=j) = mu[j] by stationarity
          - Y_{g,n-1} = g(X_{n-1})
        """
        masses, cond = self._cluster_next_state_distributions(P, mu, labels, num_clusters)

        eps = 1e-15
        mu_safe = np.clip(mu, eps, None)
        cond_safe = np.clip(cond, eps, None)

        mi = 0.0
        for c in range(num_clusters):
            if masses[c] <= 0:
                continue
            mi += masses[c] * np.sum(cond[c] * np.log2(cond_safe[c] / mu_safe))
        return float(mi)

    def _partition_states(self, P: np.ndarray) -> np.ndarray:
        """
        Greedy agglomerative IB partitioning.

        At each step, merge the pair of clusters that causes the smallest loss in
        I(X_n ; Y_{g,n-1}), equivalently the largest value after merging.
        """
        self._validate_transition_matrix(P)

        n = P.shape[0]
        if not (1 < self.k <= n):
            raise ValueError("k_macro_states must satisfy 1 < k_macro_states <= number of states.")

        mu = self._compute_stationary_distribution(P)

        labels = np.arange(n)
        current_clusters = n

        while current_clusters > self.k:
            best_merge = None
            best_objective = -np.inf

            unique_labels = np.unique(labels)
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    l1, l2 = unique_labels[i], unique_labels[j]

                    tentative_labels = labels.copy()
                    tentative_labels[tentative_labels == l2] = l1
                    _, contiguous_labels = np.unique(tentative_labels, return_inverse=True)

                    objective = self._calculate_ib_mutual_information(
                        P=P,
                        mu=mu,
                        labels=contiguous_labels,
                        num_clusters=current_clusters - 1,
                    )

                    if objective > best_objective:
                        best_objective = objective
                        best_merge = (l1, l2)

            if best_merge is None:
                raise RuntimeError("Failed to find a valid merge during agglomeration.")

            labels[labels == best_merge[1]] = best_merge[0]
            current_clusters -= 1

        _, final_labels = np.unique(labels, return_inverse=True)
        return final_labels

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
