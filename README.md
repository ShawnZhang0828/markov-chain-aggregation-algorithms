# Markov Chain Aggregation Algorithms

## Purpose
This repository provides a rigorous computational framework for evaluating and benchmarking state space aggregation algorithms for Markov Chains. The primary objective is to compare established heuristic approaches (e.g., Spectral Clustering, Information Bottleneck, and Deep Reinforcement Learning) against optimization-based aggregation methodologies. 

By state space aggregation, this test suite focuses on algorithms that project high-dimensional transition matrices onto lower-dimension ones. To compare the performances of the algorithms/heuristics, this suite analyzes the trade-offs between computational tractability, spectral preservation, and information loss (Kullback-Leibler divergence rate).

## Mathematical Assumptions
To ensure the mathematical validity and software engineering rigor of the aggregation algorithms, the following constraints are assumed for all input data:

1. **Stochasticity**: Every input transition matrix $P \in \mathbb{R}^{n \times n}$ must be strictly row-stochastic. 
   $$\forall i \in \{1, \dots, n\}, \sum_{j=1}^{n} P_{ij} = 1 \quad \text{and} \quad P_{ij} \geq 0$$
2. **Ergodicity**: The underlying Markov Chain is assumed to be irreducible and positive recurrent (ergodic). This guarantees the existence of a unique, strictly positive stationary distribution $\pi \in \mathbb{R}^{1 \times n}$, which is a prerequisite for calculating information-theoretic metrics.
3. **Time-Homogeneity**: The transition probabilities $P_{ij}$ are independent of time $t$.
4. **Information Theory Limits**: To maintain numerical stability and strict adherence to Shannon entropy definitions, the calculation of mutual information enforces the limit convention where zero-probability transitions yield zero information content:
   $$\lim_{p \to 0} p \log p = 0$$

## Repository Structure
The repository is modularized into isolated algorithm environments and a centralized benchmarking suite:

```text
├── spectral_clustering/        # Spectral partition algorithms using eigenvector embedding (e.g., Deng et al.)
├── information_theoretic/      # Agglomerative KL-divergence minimization (e.g., Geiger et al.)
├── reinforcement_learning/     # Deep RL state abstraction methods
├── sdp_optimization/           # Optimization and deterministic rounding methodologies
└── experiment/                 # Centralized evaluation pipelines and synthetic data generation
    ├── benchmark.py            # Main execution script for empirical testing
    └── metrics.py              # Mathematical definitions of evaluation criteria
```

## Evaluation Metrics

The algorithms in this suite are benchmarked across three strictly defined dimensions to quantify the trade-offs between computational efficiency, spectral preservation, and information loss.

### 1. Computational Runtime

To objectively evaluate the algorithmic complexity, the wall-clock execution time of the aggregation phase is recorded.

* **Measurement Standard:** Utilizes high-resolution, monotonic performance counters (e.g., Python's `time.perf_counter()`) to eliminate OS-level time synchronization artifacts.
* **Exclusions:** Data generation, matrix initialization, and metric evaluations are excluded from this measurement.

### 2. Information Preservation

Based on Information Bottleneck principles, the aggregation process is evaluated by how much predictive information the macro-state chain retains about its future dynamics.

* **Mutual Information ($\mathcal{I}$):** Measures the information the current state contains about the next state. For the original ergodic transition matrix $P$ with stationary distribution $\pi$:

$$ I(X_t;X_{t+1}) = \sum_{i=1}^n \sum_{j=1}^n \pi_i P_{ij} \log(\frac{P_{ij}}{\pi_j}) $$


The aggregated chain's mutual information, $I(\hat{X}_t; \hat{X}_{t+1})$, is computed identically using $\hat{P}$ and $\hat{\pi}$.

* **Information Loss (KLDR):** Quantifies the exact information lost due to the dimensionality reduction. Lower values indicate higher fidelity to the original system's dynamics.

$$ \mathcal{D}_{KL}(P || \hat{P}) = I(X_t;X_{t+1}) - I(\hat{X}_t; \hat{X}_{t+1}) $$

---

## Usage

### 1. Installation and Requirements

This benchmarking suite requires a Python 3.8+ environment. The numerical computations rely heavily on optimized linear algebra libraries.

Install the required dependencies using `pip`:

```bash
pip install numpy scipy scikit-learn
```

### 2. Running the Benchmark Pipeline

The centralized experiment script generates a synthetic  metastable transition matrix, applies the designated aggregation algorithm, and outputs the strictly defined metrics.

Execute the main experiment script from the repository root:

```bash
python experiment/benchmark.py
```

**Expected Output Structure:**

```text
Generating strictly structured metastable transition matrix...
Executing spectral clustering partition and aggregation...
Evaluating information preservation metrics...

--- Benchmark Results ---
Algorithm Runtime:          0.041230 seconds
Original Mutual Info:       4.123456 bits
Aggregated Mutual Info:     3.987654 bits
Information Loss (KL Rate): 0.135802 bits
```
