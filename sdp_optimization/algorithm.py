from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None

from core.base_aggregator import BaseMarkovAggregator

"""
Hierarchical Markov chain aggregation based on the user's discretized SDP idea.

This module follows the interface of BaseMarkovAggregator:
    - The public benchmarking entry point is `aggregate(P)` inherited from the base class.
    - The core mathematical work is implemented in `_partition_states(P)`.

Design summary
--------------
1. For binary aggregation, solve a family of discretized SDP subproblems.
2. Pick the relaxed subproblem with the best convex part A(x).
3. Apply Gaussian hyperplane rounding to the common PSD matrix D.
4. Recover an integral two-way partition.
5. For k > 2 macro-states, repeat binary splitting hierarchically.

Important note
--------------
This implementation assumes the approximation logic discussed by the user exists,
but it does not prove the approximation factor. The code only implements the
workflow implied by that assumption.
"""


@dataclass(frozen=True)
class MeshBin:
    """A single discretization interval for one scalar variable in (0, 1)."""

    side: str
    lower: float
    upper: float
    anchor: float
    index: int


@dataclass(frozen=True)
class SumBin:
    """A derived interval for a sum such as S1+S4 or S2+S3."""

    lower: float
    upper: float


class DiscretizedSDPAggregator(BaseMarkovAggregator):
    """
    Binary Markov aggregation using the discretized SDP pipeline.

    This class implements only the 2-state case. Multi-state aggregation is
    built on top of it by recursive binary splitting in the subclass below.
    """

    def __init__(
        self,
        k_macro_states: int = 2,
        approximation_factor: float = 1.5,
        left_eps: float = 1e-2,
        right_eps: float = 1e-2,
        max_bins_per_side: int | None = None,
        gaussian_rounds: int = 128,
        solver: str = "SCS",
        solver_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        if k_macro_states != 2:
            raise ValueError("DiscretizedSDPAggregator only supports two macro-states.")
        super().__init__(k_macro_states)
        if approximation_factor <= 1.0:
            raise ValueError("approximation_factor must be greater than 1.")
        if gaussian_rounds <= 0:
            raise ValueError("gaussian_rounds must be positive.")

        self.c = float(approximation_factor)
        self.left_eps = float(left_eps)
        self.right_eps = float(right_eps)
        self.max_bins_per_side = max_bins_per_side
        self.gaussian_rounds = int(gaussian_rounds)
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {"eps": 1e-6, "max_iters": 30_000}
        self.rng = np.random.default_rng(random_state)
        self.n_jobs = max(1, int(n_jobs))

    # ---------------------------------------------------------------------
    # Basic validation and Markov-chain utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _validate_transition_matrix(P: np.ndarray) -> None:
        """Validate that P is a square row-stochastic transition matrix."""
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be a square matrix.")
        if np.any(P < -1e-12):
            raise ValueError("P must have nonnegative entries.")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("Every row of P must sum to 1.")

    def _compute_stationary_distribution(self, P: np.ndarray) -> np.ndarray:
        """Compute one stationary distribution by solving a linear system."""
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
        if total <= 0.0:
            raise ValueError("Failed to compute a valid stationary distribution.")
        return mu / total

    # ---------------------------------------------------------------------
    # Discretization helpers
    # ---------------------------------------------------------------------
    def _build_single_variable_bins(self) -> list[MeshBin]:
        """
        Build the hybrid geometric mesh on both sides of 1/e.

        Left bins approach 1/e from below by multiplication with c.
        Right bins approach 1/e from above by multiplying the distance to 1.
        """
        seam = 1.0 / np.e
        bins: list[MeshBin] = []

        x = self.left_eps
        left_index = 0
        while x < seam:
            x_next = min(self.c * x, seam)
            anchor = x if x_next < seam else seam / self.c
            if x_next <= anchor:
                anchor = max(self.left_eps, min(anchor, x_next * (1.0 - 1e-12)))
            bins.append(
                MeshBin("left", float(x), float(x_next), float(anchor), left_index)
            )
            left_index += 1
            x = x_next
            if (
                self.max_bins_per_side is not None
                and left_index >= self.max_bins_per_side
            ):
                break
            if np.isclose(x, seam):
                break

        y = 1.0 - self.right_eps
        right_index = 0
        while y > seam:
            distance = 1.0 - y
            y_next = max(1.0 - self.c * distance, seam)
            anchor = y if y_next > seam else 1.0 - (1.0 - seam) / self.c
            anchor = min(max(anchor, y_next + 1e-12), y - 1e-12)
            bins.append(
                MeshBin("right", float(y_next), float(y), float(anchor), right_index)
            )
            right_index += 1
            y = y_next
            if (
                self.max_bins_per_side is not None
                and right_index >= self.max_bins_per_side
            ):
                break
            if np.isclose(y, seam):
                break

        bins.sort(key=lambda b: (b.lower, b.upper))
        return bins

    @staticmethod
    def _derived_sum_bin(bin_a: MeshBin, bin_b: MeshBin) -> SumBin:
        """Derive the interval of a sum from two scalar bins."""
        return SumBin(lower=bin_a.lower + bin_b.lower, upper=bin_a.upper + bin_b.upper)

    def _enumerate_subproblem_bins(self) -> list[dict[str, Any]]:
        """
        Enumerate all discretized subproblems with early pruning.

        Speed-up strategy
        -----------------
        The naive implementation loops over all quadruples (b1, b2, b3, b4),
        which costs O(m^4) when there are m scalar bins. Here we use the simplex
        relation S1 + S2 + S3 + S4 = 1 to avoid a full fourth loop:

        1. Enumerate only (b1, b2, b3).
        2. Infer the feasible interval of S4 from the first three bins.
        3. Keep only those b4 bins that intersect the induced S4 interval.

        This does not change the set of generated subproblems, but it removes a
        large number of impossible combinations before they are materialized.
        """
        scalar_bins = self._build_single_variable_bins()
        subproblems: list[dict[str, Any]] = []
        tol = 1e-9

        m = len(scalar_bins)
        lowers = np.array([b.lower for b in scalar_bins], dtype=float)
        uppers = np.array([b.upper for b in scalar_bins], dtype=float)

        # Precompute all pairwise sum bins once. This avoids repeated Python
        # object construction inside the nested loops.
        pair_sum_bins = [
            [self._derived_sum_bin(scalar_bins[i], scalar_bins[j]) for j in range(m)]
            for i in range(m)
        ]

        for i, b1 in enumerate(scalar_bins):
            for j, b2 in enumerate(scalar_bins):
                # Even before choosing S3 and S4, the lower bounds of S1 and S2
                # cannot already exceed the simplex budget.
                if b1.lower + b2.lower >= 1.0 - tol:
                    continue

                for k, b3 in enumerate(scalar_bins):
                    low123 = b1.lower + b2.lower + b3.lower
                    up123 = b1.upper + b2.upper + b3.upper

                    # If the first three bins already force the sum to be at
                    # least one, there is no room left for a strictly positive S4.
                    if low123 >= 1.0 - tol:
                        continue

                    # Feasible interval induced for S4 by S1+S2+S3+S4 = 1.
                    s4_low = max(0.0, 1.0 - up123)
                    s4_up = min(1.0, 1.0 - low123)
                    if s4_low > s4_up + tol:
                        continue

                    # T1 depends only on (b2, b3), so compute it once here.
                    t1_bin = pair_sum_bins[j][k]
                    if t1_bin.lower >= 1.0 - tol or t1_bin.upper <= tol:
                        continue

                    # Keep only those S4 bins that overlap the induced feasible
                    # interval [s4_low, s4_up].
                    overlap_mask = (uppers >= s4_low - tol) & (lowers <= s4_up + tol)
                    candidate_l = np.flatnonzero(overlap_mask)
                    if candidate_l.size == 0:
                        continue

                    for l in candidate_l:
                        b4 = scalar_bins[l]
                        t0_bin = pair_sum_bins[i][l]

                        # Basic feasibility screening for the derived sums.
                        if t0_bin.lower >= 1.0 - tol or t0_bin.upper <= tol:
                            continue
                        if t0_bin.lower + t1_bin.lower > 1.0 + tol:
                            continue
                        if t0_bin.upper + t1_bin.upper < 1.0 - tol:
                            continue

                        subproblems.append(
                            {
                                "S_bins": (b1, b2, b3, b4),
                                "T0_bin": t0_bin,
                                "T1_bin": t1_bin,
                            }
                        )

        return subproblems

    # ---------------------------------------------------------------------
    # SDP model construction
    # ---------------------------------------------------------------------
    @staticmethod
    def _linearized_delta_expressions(
        D: cp.Expression,
        flow: np.ndarray,
    ) -> tuple[list[cp.Expression], list[cp.Expression]]:
        """
        Build vectorized affine expressions for delta_1,...,delta_4 and S_1,...,S_4.

        The previous scalar-by-scalar construction created a very large number of
        CVXPY subexpressions. This vectorized version keeps the mathematical model
        the same, but substantially reduces compilation overhead.
        """
        d_col = D[:, [0]]
        d_row = D[[0], :]
        ones = np.ones_like(flow)

        delta1 = 0.25 * cp.multiply(flow, ones - d_col - d_row + D)
        delta2 = 0.25 * cp.multiply(flow, ones + d_col + d_row + D)
        delta3 = 0.25 * cp.multiply(flow, ones + d_col - d_row - D)
        delta4 = 0.25 * cp.multiply(flow, ones - d_col + d_row - D)

        delta_exprs = [delta1, delta2, delta3, delta4]
        S_exprs = [cp.sum(delta) for delta in delta_exprs]
        return S_exprs, delta_exprs

    def _solve_relaxed_subproblem(
        self,
        P: np.ndarray,
        mu: np.ndarray,
        subproblem_spec: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Solve one relaxed SDP subproblem.

        The objective optimizes only the convex part A(x), exactly as assumed in
        the user's proposed framework.
        """
        if cp is None:
            raise ImportError("cvxpy is required to solve the SDP subproblems.")

        n = P.shape[0]
        if n < 2:
            return None

        # print(
        #     f"Solving relaxed subproblem with S_bins {[b.index for b in subproblem_spec['S_bins']]} and T_bins {[subproblem_spec['T0_bin'], subproblem_spec['T1_bin']]}..."
        # )

        flow = mu[:, None] * P
        D = cp.Variable((n, n), symmetric=True)
        S_exprs, delta_exprs = self._linearized_delta_expressions(D, flow)
        T0 = S_exprs[0] + S_exprs[3]
        T1 = S_exprs[1] + S_exprs[2]

        constraints: list[cp.Constraint] = [D >> 0, cp.diag(D) == 1.0]

        for delta in delta_exprs:
            constraints.append(delta >= -1e-10)

        s_bins: tuple[MeshBin, MeshBin, MeshBin, MeshBin] = subproblem_spec["S_bins"]
        for k, b in enumerate(s_bins):
            constraints.append(S_exprs[k] >= b.lower)
            constraints.append(S_exprs[k] <= b.upper)

        t0_bin: SumBin = subproblem_spec["T0_bin"]
        t1_bin: SumBin = subproblem_spec["T1_bin"]
        constraints.append(T0 >= t0_bin.lower)
        constraints.append(T0 <= t0_bin.upper)
        constraints.append(T1 >= t1_bin.lower)
        constraints.append(T1 <= t1_bin.upper)

        # cvxpy's entr(x) = -x log x, so maximizing the sum below matches the
        # convex part 2*T0*log(T0) + 2*T1*log(T1) up to the expected sign.
        eps = 1e-9
        objective = cp.Maximize(2.0 * cp.entr(T0 + eps) + 2.0 * cp.entr(T1 + eps))
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.solver, **self.solver_kwargs)
        except Exception:
            return None

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or D.value is None:
            return None

        D_val = np.asarray(D.value, dtype=float)
        D_val = 0.5 * (D_val + D_val.T)
        S_val = np.array([float(expr.value) for expr in S_exprs], dtype=float)

        return {
            "D": D_val,
            "S": S_val,
            "T0": float(T0.value),
            "T1": float(T1.value),
            "relaxed_A": float(problem.value),
            "subproblem_spec": subproblem_spec,
        }

    def _solve_best_relaxed_subproblem(
        self, P: np.ndarray, mu: np.ndarray
    ) -> dict[str, Any]:
        """Enumerate all subproblems and keep the relaxed one with best A-value."""
        specs = self._enumerate_subproblem_bins()
        best_relaxed: dict[str, Any] | None = None

        if self.n_jobs <= 1:
            for spec in specs:
                candidate = self._solve_relaxed_subproblem(
                    P=P, mu=mu, subproblem_spec=spec
                )
                if candidate is None:
                    continue
                if (
                    best_relaxed is None
                    or candidate["relaxed_A"] > best_relaxed["relaxed_A"]
                ):
                    best_relaxed = candidate
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futures = [
                    ex.submit(
                        self._solve_relaxed_subproblem, P=P, mu=mu, subproblem_spec=spec
                    )
                    for spec in specs
                ]
                for fut in as_completed(futures):
                    candidate = fut.result()
                    if candidate is None:
                        continue
                    if (
                        best_relaxed is None
                        or candidate["relaxed_A"] > best_relaxed["relaxed_A"]
                    ):
                        best_relaxed = candidate

        if best_relaxed is None:
            raise RuntimeError("No feasible relaxed subproblem was found.")
        return best_relaxed

    # ---------------------------------------------------------------------
    # Rounding and objective evaluation
    # ---------------------------------------------------------------------
    @staticmethod
    def _nearest_psd_gram_factor(D: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """Convert a PSD matrix D into a Gram factor V with D ≈ V V^T."""
        D_sym = 0.5 * (D + D.T)
        eigvals, eigvecs = np.linalg.eigh(D_sym)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        keep = eigvals > tol
        if not np.any(keep):
            raise ValueError("D has no positive eigenvalues after PSD projection.")
        return eigvecs[:, keep] @ np.diag(np.sqrt(eigvals[keep]))

    def _gaussian_round_labels(self, D: np.ndarray) -> np.ndarray:
        """
        Perform one Gaussian hyperplane rounding step.

        The rounded sign pattern defines a two-way partition and therefore an
        integral ±1 rank-one matrix D_int = s s^T.
        """
        V = self._nearest_psd_gram_factor(D)
        g = self.rng.standard_normal(V.shape[1])
        scores = V @ g
        labels = (scores >= 0.0).astype(int)

        # Keep the anchor pair separated to avoid trivial collapse.
        # if labels.size >= 2 and labels[0] == labels[1]:
        #     labels[1] = 1 - labels[0]
        return labels

    @staticmethod
    def _labels_to_D(labels: np.ndarray) -> np.ndarray:
        """Convert binary labels into an integral ±1 matrix D = s s^T."""
        spins = np.where(labels == 0, -1.0, 1.0)
        return np.outer(spins, spins)

    @staticmethod
    def _compute_S_values_from_labels(
        P: np.ndarray, mu: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Recompute S1,S2,S3,S4 from an integral partition."""
        n = P.shape[0]
        flow = mu[:, None] * P
        S = np.zeros(4, dtype=float)
        for i in range(n):
            for j in range(n):
                w = flow[i, j]
                a = labels[i]
                b = labels[j]
                if a == 0 and b == 0:
                    S[0] += w
                elif a == 1 and b == 1:
                    S[1] += w
                elif a == 1 and b == 0:
                    S[2] += w
                else:
                    S[3] += w
        return S

    @staticmethod
    def _full_objective_from_S(S: np.ndarray) -> float:
        """Evaluate the original nonlinear objective from S-values."""
        eps = 1e-15
        S_safe = np.clip(S, eps, None)
        T0 = S_safe[0] + S_safe[3]
        T1 = S_safe[1] + S_safe[2]
        return float(
            2.0 * T1 * np.log(T1)
            + 2.0 * T0 * np.log(T0)
            - np.sum(S_safe * np.log(S_safe))
        )

    def _round_and_select_best(
        self, P: np.ndarray, mu: np.ndarray, D: np.ndarray
    ) -> dict[str, Any]:
        """Run several Gaussian rounds and keep the best integral partition."""
        best: dict[str, Any] | None = None
        for _ in range(self.gaussian_rounds):
            labels = self._gaussian_round_labels(D)
            S = self._compute_S_values_from_labels(P, mu, labels)
            objective = self._full_objective_from_S(S)
            if best is None or objective < best["objective"]:
                best = {
                    "labels": labels.copy(),
                    "S": S.copy(),
                    "objective": float(objective),
                    "D_integral": self._labels_to_D(labels),
                }

        if best is None:
            raise RuntimeError(
                "Gaussian rounding failed to produce any candidate solution."
            )
        return best

    def _solve_binary_partition(
        self, P: np.ndarray, mu: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Enumerate subproblems, round each relaxed solution, and keep the best candidate."""
        self._validate_transition_matrix(P)
        if cp is None:
            raise ImportError("cvxpy is required for DiscretizedSDPAggregator.")

        if mu is None:
            mu = self._compute_stationary_distribution(P)

        best_result: dict[str, Any] | None = None
        for spec in self._enumerate_subproblem_bins():
            relaxed = self._solve_relaxed_subproblem(P, mu, spec)
            if relaxed is None:
                continue

            rounded = self._round_and_select_best(P, mu, relaxed["D"])
            candidate = {
                "labels": rounded["labels"],
                "rounded_S": rounded["S"],
                "rounded_objective": rounded["objective"],
                "relaxed_D": relaxed["D"],
                "relaxed_S": relaxed["S"],
                "relaxed_T0": relaxed["T0"],
                "relaxed_T1": relaxed["T1"],
                "relaxed_A": relaxed["relaxed_A"],
                "subproblem_spec": spec,
            }
            if (
                best_result is None
                or candidate["rounded_objective"] < best_result["rounded_objective"]
            ):
                best_result = candidate

        if best_result is None:
            raise RuntimeError(
                "No feasible discretized subproblem produced an integral candidate."
            )
        return best_result

    # ---------------------------------------------------------------------
    # Interface required by BaseMarkovAggregator
    # ---------------------------------------------------------------------
    def _partition_states(self, P: np.ndarray) -> np.ndarray:
        """
        Partition the state space into two macro-states.

        This is the required method from BaseMarkovAggregator. The base class
        takes care of turning these labels into (P_hat, V).
        """
        mu = self._compute_stationary_distribution(P)
        result = self._solve_binary_partition(P, mu)
        return np.asarray(result["labels"], dtype=int)

    # ---------------------------------------------------------------------
    # Optional diagnostics helpers
    # ---------------------------------------------------------------------
    def aggregate_with_diagnostics(self, P: np.ndarray) -> dict[str, Any]:
        """Run the binary method and return detailed intermediate information."""
        self._validate_transition_matrix(P)
        mu = self._compute_stationary_distribution(P)
        result = self._solve_binary_partition(P, mu)
        P_hat, V = super().aggregate(P)
        return {
            "P_hat": P_hat,
            "V": V,
            "labels": result["labels"],
            **result,
        }


class SDPAggregator(DiscretizedSDPAggregator):
    """
    Extend the 2-state method to k > 2 via recursive binary splitting.

    At each stage, the algorithm tries splitting each current cluster into two
    child clusters. It keeps the split that looks best under a chosen scoring
    rule, then repeats until k clusters are obtained.
    """

    def __init__(
        self,
        k_macro_states: int,
        approximation_factor: float = 1.5,
        left_eps: float = 1e-2,
        right_eps: float = 1e-2,
        max_bins_per_side: int | None = None,
        gaussian_rounds: int = 128,
        solver: str = "SCS",
        solver_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
        min_split_cluster_size: int = 2,
        local_split_restarts: int = 1,
        split_selection_metric: str = "block_average_objective",
    ):
        if k_macro_states < 2:
            raise ValueError("k_macro_states must be at least 2.")
        if min_split_cluster_size < 2:
            raise ValueError("min_split_cluster_size must be at least 2.")
        if local_split_restarts < 1:
            raise ValueError("local_split_restarts must be at least 1.")
        if split_selection_metric not in {
            "block_average_objective",
            "local_two_state_objective",
        }:
            raise ValueError("Unsupported split_selection_metric.")

        # Reuse all binary-method parameters, but initialize the base interface
        # with the requested final number of macro-states.
        BaseMarkovAggregator.__init__(self, k_macro_states)
        self.c = float(approximation_factor)
        self.left_eps = float(left_eps)
        self.right_eps = float(right_eps)
        self.max_bins_per_side = max_bins_per_side
        self.gaussian_rounds = int(gaussian_rounds)
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {"eps": 1e-6, "max_iters": 30_000}
        self.rng = np.random.default_rng(random_state)
        self.n_jobs = max(1, int(n_jobs))

        self.min_split_cluster_size = int(min_split_cluster_size)
        self.local_split_restarts = int(local_split_restarts)
        self.split_selection_metric = split_selection_metric

    @staticmethod
    def _make_labels_contiguous(labels: np.ndarray) -> np.ndarray:
        """Relabel clusters so that labels are exactly 0,1,...,k-1."""
        _, inv = np.unique(labels, return_inverse=True)
        return inv.astype(int)

    def _build_local_subchain(self, P: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Restrict the global chain to one cluster and renormalize its rows.

        This produces a local Markov chain on that cluster so the binary method
        can be reused as a splitting primitive.
        """
        P_sub = np.asarray(P[np.ix_(states, states)], dtype=float)
        row_sums = P_sub.sum(axis=1, keepdims=True)
        local = np.zeros_like(P_sub)
        positive = row_sums[:, 0] > 1e-15
        local[positive] = P_sub[positive] / row_sums[positive]
        local[~positive, :] = 0.0
        if np.any(~positive):
            idx = np.where(~positive)[0]
            local[idx, idx] = 1.0
        local = np.clip(local, a_min=0.0, a_max=None)
        local /= local.sum(axis=1, keepdims=True)
        return local

    @staticmethod
    def _block_average_objective(P: np.ndarray, labels: np.ndarray, k: int) -> float:
        """
        Score a global partition using the same uniform block averaging rule as
        BaseMarkovAggregator.aggregate.
        """
        P_hat = np.zeros((k, k), dtype=float)
        for i in range(k):
            states_i = np.where(labels == i)[0]
            if states_i.size == 0:
                continue
            for j in range(k):
                states_j = np.where(labels == j)[0]
                block_sum = np.sum(P[np.ix_(states_i, states_j)])
                P_hat[i, j] = block_sum / states_i.size

        lifted = np.zeros_like(P)
        for i in range(P.shape[0]):
            ci = labels[i]
            members = np.where(labels == ci)[0]
            lifted[i, :] = 0.0
            for j in range(k):
                targets = np.where(labels == j)[0]
                if targets.size > 0:
                    lifted[i, targets] = P_hat[ci, j] / targets.size

        eps = 1e-15
        P_safe = np.clip(P, eps, None)
        lifted_safe = np.clip(lifted, eps, None)
        return float(np.sum(P_safe * np.log(P_safe / lifted_safe)))

    def _split_score(
        self, P: np.ndarray, labels: np.ndarray, local_result: dict[str, Any]
    ) -> float:
        """Compute the score used to choose which current cluster to split."""
        k = int(np.max(labels)) + 1
        if self.split_selection_metric == "local_two_state_objective":
            return -float(local_result["rounded_objective"])
        return -self._block_average_objective(P, labels, k)

    def _candidate_split_for_cluster(
        self,
        P: np.ndarray,
        global_labels: np.ndarray,
        cluster_id: int,
    ) -> dict[str, Any] | None:
        """Try splitting one current cluster and return the best local attempt."""
        states = np.where(global_labels == cluster_id)[0]
        if states.size < self.min_split_cluster_size:
            return None

        P_local = self._build_local_subchain(P, states)
        best_candidate: dict[str, Any] | None = None
        best_score = -np.inf

        for _ in range(self.local_split_restarts):
            print(f"Trying to split cluster {cluster_id} with {states.size} states...")
            try:
                local_result = self._solve_binary_partition(P_local)
            except Exception:
                continue

            local_labels = np.asarray(local_result["labels"], dtype=int)
            if np.all(local_labels == local_labels[0]):
                continue

            new_global_labels = global_labels.copy()
            new_cluster_id = int(np.max(global_labels)) + 1
            for local_idx, state in enumerate(states):
                new_global_labels[state] = (
                    cluster_id if local_labels[local_idx] == 0 else new_cluster_id
                )
            new_global_labels = self._make_labels_contiguous(new_global_labels)

            score = self._split_score(P, new_global_labels, local_result)
            if score > best_score:
                best_score = score
                best_candidate = {
                    "global_labels": new_global_labels,
                    "score": float(score),
                    "split_cluster": int(cluster_id),
                    "split_states": states.copy(),
                    "local_labels": local_labels.copy(),
                    "local_result": local_result,
                }

        return best_candidate

    def _hierarchical_partition(
        self, P: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Repeatedly split clusters until the target number of macro-states is reached."""
        self._validate_transition_matrix(P)
        n = P.shape[0]
        if not (2 <= self.k <= n):
            raise ValueError("k_macro_states must satisfy 2 <= k_macro_states <= n.")

        labels = np.zeros(n, dtype=int)
        history: list[dict[str, Any]] = []

        while int(np.max(labels)) + 1 < self.k:
            best_candidate: dict[str, Any] | None = None
            for cluster_id in np.unique(labels):
                candidate = self._candidate_split_for_cluster(
                    P, labels, int(cluster_id)
                )
                if candidate is None:
                    continue
                if (
                    best_candidate is None
                    or candidate["score"] > best_candidate["score"]
                ):
                    best_candidate = candidate

            if best_candidate is None:
                raise RuntimeError("No cluster could be split further.")

            labels = best_candidate["global_labels"]
            history.append(best_candidate)

        return self._make_labels_contiguous(labels), history

    def _partition_states(self, P: np.ndarray) -> np.ndarray:
        """Partition the state space into k macro-states by recursive splitting."""
        self._validate_transition_matrix(P)
        if self.k == 2:
            return super()._partition_states(P)
        labels, _ = self._hierarchical_partition(P)
        return labels

    def aggregate_with_diagnostics(self, P: np.ndarray) -> dict[str, Any]:
        """Return the final aggregation together with the hierarchical split history."""
        self._validate_transition_matrix(P)
        if self.k == 2:
            result = super().aggregate_with_diagnostics(P)
            result["split_history"] = [
                {"type": "single_binary_split", "labels": result["labels"]}
            ]
            return result

        labels, history = self._hierarchical_partition(P)
        P_hat, V = super().aggregate(P)
        return {
            "P_hat": P_hat,
            "V": V,
            "labels": labels,
            "split_history": history,
        }
