import csv
import math
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil

from data_handler.alanine_dipeptide_loader import load_alanine_dipeptide_markov_chain
from data_handler.wikimedia_loader import load_wikimedia_clickstream_markov_chain
from data_handler.geolife_loader import load_geolife_markov_chain
from spectral_clustering.algorithm import SpectralAggregator
from information_theoretic.algorithm import InformationTheoreticAggregator
from transition_based.algorithm import TransitionBasedAggregator
from sdp_optimization.algorithm import SDPAggregator


# =========================
# User-facing benchmark knobs
# =========================
K_MACRO = 4
K_MACRO_SMALL = 2
TIMEOUT_SECONDS = 18 * 60
SMALL_N_STATES = 30
LARGE_N_STATES = 100
RANDOM_N_STATES = 10
RANDOM_SEED = 0
RESULTS_DIR = "./benchmark_results"
CACHE_DIR = os.path.join(RESULTS_DIR, "matrix_cache")
MEMORY_POLL_INTERVAL_SECONDS = 0.2


# =========================
# Dataset-specific paths / options
# =========================
ALANINE_WORKDIR = "./data_handler/datasets"
WIKIMEDIA_CLICKSTREAM_PATH = "./data_handler/datasets/clickstream-enwiki-2026-02.tsv.gz"
GEOLIFE_ROOT_DIR = (
    "./data_handler/datasets/Geolife Trajectories 1.3/Geolife Trajectories 1.3/Data"
)

ALANINE_KWARGS = {
    "lag": 1,
    "seed": 0,
    "smoothing": 1e-8,
    "working_directory": ALANINE_WORKDIR,
}

WIKIMEDIA_KWARGS = {
    "clickstream_tsv_gz": WIKIMEDIA_CLICKSTREAM_PATH,
    "link_only": True,
    "min_count": 50,
    "smoothing": 1e-8,
    "top_by": "source_flow",
}

GEOLIFE_KWARGS = {
    "root_dir": GEOLIFE_ROOT_DIR,
    "max_users": 50,
    "max_files_per_user": 50,
    "random_state": 0,
}


@dataclass
class BenchmarkCase:
    dataset_name: str
    variant: str
    n_states: int
    extra: Optional[dict] = None

    @property
    def case_id(self) -> str:
        return f"{self.dataset_name}__{self.variant}__n{self.n_states}"


@dataclass
class BenchmarkRecord:
    dataset_name: str
    variant: str
    n_states: int
    algorithm: str
    status: str
    runtime_seconds: float
    kldr: Optional[float]
    memory_consumption_mb: Optional[float]
    matrix_path: str = ""
    error_message: str = ""


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, a_min=0.0, a_max=None)
    s = pi.sum()
    if s <= 0:
        raise ValueError("Failed to compute a valid stationary distribution.")
    return pi / s


def build_lifted_chain(
    P: np.ndarray, P_hat: np.ndarray, labels: np.ndarray, pi: np.ndarray
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or labels.shape[0] != P.shape[0]:
        raise ValueError(
            "labels must be a 1D array with length equal to number of states."
        )

    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    remapped = np.array([label_to_idx[label] for label in labels], dtype=int)
    k = len(unique_labels)

    if P_hat.shape != (k, k):
        raise ValueError(
            f"P_hat shape {P_hat.shape} is inconsistent with {k} aggregated states implied by labels."
        )

    cluster_mass = np.zeros(k, dtype=float)
    for c in range(k):
        cluster_mass[c] = pi[remapped == c].sum()

    Q = np.zeros_like(P, dtype=float)
    for i in range(P.shape[0]):
        a = remapped[i]
        for j in range(P.shape[1]):
            b = remapped[j]
            if cluster_mass[b] > 0:
                Q[i, j] = P_hat[a, b] * pi[j] / cluster_mass[b]

    row_sums = Q.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    Q[valid] /= row_sums[valid]
    return Q


def calculate_kldr(
    P: np.ndarray, Q: np.ndarray, pi: np.ndarray, eps: float = 1e-15
) -> float:
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} and {Q.shape}."
        )

    total = 0.0
    n = P.shape[0]
    for i in range(n):
        for j in range(n):
            pij = float(P[i, j])
            if pij <= eps:
                continue
            qij = max(float(Q[i, j]), eps)
            total += float(pi[i]) * pij * np.log(pij / qij)
    return float(total)


def extract_labels_from_aggregate_output(
    P: np.ndarray, aggregate_output: Tuple[np.ndarray, object]
) -> np.ndarray:
    if not isinstance(aggregate_output, tuple) or len(aggregate_output) < 2:
        raise ValueError(
            "aggregate(P) must return at least two objects, e.g. (P_hat, labels_or_membership)."
        )

    second = np.asarray(aggregate_output[1])
    n = P.shape[0]

    if second.ndim == 1 and second.shape[0] == n:
        return second.astype(int)

    if second.ndim == 2:
        if second.shape[0] == n:
            return np.argmax(second, axis=1).astype(int)
        if second.shape[1] == n:
            return np.argmax(second, axis=0).astype(int)

    raise ValueError(
        f"Could not extract labels from aggregate() second output with shape {second.shape}."
    )


def evaluate_information_loss_kldr(
    P: np.ndarray, P_hat: np.ndarray, labels: np.ndarray
) -> dict:
    pi = compute_stationary_distribution(P)
    Q = build_lifted_chain(P, P_hat, labels, pi)
    return {"kldr": calculate_kldr(P, Q, pi)}


def generate_metastable_markov_chain(
    n: int, k: int, noise: float = 0.1, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = rng.random((n, n)) * noise
    block_size = n // k
    for i in range(k):
        start = i * block_size
        end = (i + 1) * block_size if i < k - 1 else n
        P[start:end, start:end] += rng.random((end - start, end - start)) * 10.0
    P /= P.sum(axis=1, keepdims=True)
    return P


def load_dataset(case: BenchmarkCase) -> np.ndarray:
    if case.dataset_name == "alanine_dipeptide":
        kwargs = dict(ALANINE_KWARGS)
        kwargs["n_states"] = case.n_states
        return load_alanine_dipeptide_markov_chain(**kwargs)

    if case.dataset_name == "wikimedia":
        kwargs = dict(WIKIMEDIA_KWARGS)
        kwargs["n_states"] = case.n_states
        return load_wikimedia_clickstream_markov_chain(**kwargs)

    if case.dataset_name == "geolife":
        kwargs = dict(GEOLIFE_KWARGS)
        kwargs["n_states"] = case.n_states
        return load_geolife_markov_chain(**kwargs)

    if case.dataset_name == "random_generated_markov_chain":
        return generate_metastable_markov_chain(
            n=case.n_states,
            k=K_MACRO_SMALL,
            noise=0.05,
            seed=RANDOM_SEED,
        )

    raise ValueError(f"Unknown dataset: {case.dataset_name}")


def build_cases() -> List[BenchmarkCase]:
    return [
        BenchmarkCase("alanine_dipeptide", "small", SMALL_N_STATES),
        BenchmarkCase("alanine_dipeptide", "large", LARGE_N_STATES),
        BenchmarkCase("wikimedia", "small", SMALL_N_STATES),
        BenchmarkCase("wikimedia", "large", LARGE_N_STATES),
        BenchmarkCase("geolife", "small", SMALL_N_STATES),
        BenchmarkCase("geolife", "large", LARGE_N_STATES),
        BenchmarkCase("random_generated_markov_chain", "single", RANDOM_N_STATES),
    ]


def algorithm_factories(k_macro: int) -> Dict[str, Callable[[], object]]:
    return {
        "spectral": lambda: SpectralAggregator(k_macro_states=k_macro),
        "sdp": lambda: SDPAggregator(k_macro_states=k_macro),
        "information_theoretic": lambda: InformationTheoreticAggregator(
            k_macro_states=k_macro
        ),
        "transition_based": lambda: TransitionBasedAggregator(k_macro_states=k_macro),
    }


def cache_path_for_case(case: BenchmarkCase) -> str:
    ensure_dir(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{case.case_id}.npy")


def materialize_case_matrix(case: BenchmarkCase, overwrite: bool = False) -> str:
    path = cache_path_for_case(case)
    if os.path.exists(path) and not overwrite:
        print(f"[Cache] Reusing cached matrix for {case.case_id}: {path}")
        return path

    print(f"[Cache] Loading matrix once for {case.case_id} ...")
    P = load_dataset(case)
    np.save(path, P)
    print(f"[Cache] Saved matrix for {case.case_id} with shape={P.shape} to {path}")
    return path


def materialize_all_case_matrices(
    cases: List[BenchmarkCase], overwrite: bool = False
) -> Dict[str, str]:
    matrix_paths: Dict[str, str] = {}
    for case in cases:
        matrix_paths[case.case_id] = materialize_case_matrix(case, overwrite=overwrite)
    return matrix_paths


def _worker_run(
    queue: mp.Queue, matrix_path: str, algorithm_name: str, k_macro: int
) -> None:
    if "random" not in matrix_path and algorithm_name == "sdp":
        queue.put(
            {
                "status": "success",
                "runtime_seconds": 0.0,
                "kldr": None,
                "error_message": "SDP algorithm is not expected to run on large chains due to scalability issues.",
            }
        )
        return
    
    start_time = time.perf_counter()
    try:
        factories = algorithm_factories(k_macro)
        aggregator = factories[algorithm_name]()

        print(f"[Worker] Loading cached transition matrix from {matrix_path}")
        P = np.load(matrix_path)
        print(
            f"[Worker] Cached matrix loaded with shape={P.shape}. Running algorithm={algorithm_name} ..."
        )

        aggregate_output = aggregator.aggregate(P)
        P_hat = np.asarray(aggregate_output[0], dtype=float)
        labels = extract_labels_from_aggregate_output(P, aggregate_output)
        metrics = evaluate_information_loss_kldr(P, P_hat, labels)
        runtime = time.perf_counter() - start_time

        queue.put(
            {
                "status": "success",
                "runtime_seconds": runtime,
                "kldr": metrics["kldr"],
                "error_message": "",
            }
        )
    except Exception:
        runtime = time.perf_counter() - start_time
        queue.put(
            {
                "status": "error",
                "runtime_seconds": runtime,
                "kldr": None,
                "error_message": traceback.format_exc(),
            }
        )


def get_process_tree_rss_mb(pid: int) -> float:
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0

    rss_bytes = 0
    try:
        rss_bytes += proc.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0

    try:
        for child in proc.children(recursive=True):
            try:
                rss_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return float(rss_bytes) / (1024.0 * 1024.0)


def run_case_with_timeout(
    case: BenchmarkCase,
    matrix_path: str,
    algorithm_name: str,
    timeout_seconds: int,
    k_macro: int,
) -> BenchmarkRecord:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_worker_run, args=(queue, matrix_path, algorithm_name, k_macro)
    )

    print(
        f"\n=== Running dataset={case.dataset_name} | variant={case.variant} | "
        f"n_states={case.n_states} | algorithm={algorithm_name} ==="
    )
    print(f"[Main] Using cached matrix: {matrix_path}")

    wall_start = time.perf_counter()
    process.start()

    peak_memory_mb = 0.0
    while True:
        if process.pid is not None:
            current_memory_mb = get_process_tree_rss_mb(process.pid)
            if current_memory_mb > peak_memory_mb:
                peak_memory_mb = current_memory_mb

        if not process.is_alive():
            break

        runtime = time.perf_counter() - wall_start
        if runtime > timeout_seconds:
            print(
                f"[Timeout] dataset={case.dataset_name}, variant={case.variant}, "
                f"algorithm={algorithm_name} exceeded {timeout_seconds} seconds."
            )
            process.terminate()
            process.join()
            return BenchmarkRecord(
                dataset_name=case.dataset_name,
                variant=case.variant,
                n_states=case.n_states,
                algorithm=algorithm_name,
                status="timeout",
                runtime_seconds=runtime,
                kldr=None,
                memory_consumption_mb=peak_memory_mb if peak_memory_mb > 0 else None,
                matrix_path=matrix_path,
                error_message=f"Timed out after {timeout_seconds} seconds.",
            )

        time.sleep(MEMORY_POLL_INTERVAL_SECONDS)

    process.join()
    runtime = time.perf_counter() - wall_start

    if queue.empty():
        return BenchmarkRecord(
            dataset_name=case.dataset_name,
            variant=case.variant,
            n_states=case.n_states,
            algorithm=algorithm_name,
            status="error",
            runtime_seconds=runtime,
            kldr=None,
            memory_consumption_mb=peak_memory_mb if peak_memory_mb > 0 else None,
            matrix_path=matrix_path,
            error_message="Worker exited without returning a result.",
        )

    result = queue.get()
    return BenchmarkRecord(
        dataset_name=case.dataset_name,
        variant=case.variant,
        n_states=case.n_states,
        algorithm=algorithm_name,
        status=result["status"],
        runtime_seconds=result["runtime_seconds"],
        kldr=result["kldr"],
        memory_consumption_mb=peak_memory_mb if peak_memory_mb > 0 else None,
        matrix_path=matrix_path,
        error_message=result["error_message"],
    )


def save_results_csv(records: List[BenchmarkRecord], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _unique_case_labels(records: List[BenchmarkRecord]) -> List[str]:
    labels = []
    seen = set()
    for r in records:
        key = f"{r.dataset_name}\n{r.variant}\n(n={r.n_states})"
        if key not in seen:
            seen.add(key)
            labels.append(key)
    return labels


def plot_runtime_chart(records: List[BenchmarkRecord], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    case_labels = _unique_case_labels(records)
    algorithms = sorted({r.algorithm for r in records})
    x = np.arange(len(case_labels))
    width = 0.18

    plt.figure(figsize=(14, 6))
    for idx, algorithm in enumerate(algorithms):
        heights = []
        for label in case_labels:
            match = next(
                (
                    r
                    for r in records
                    if r.algorithm == algorithm
                    and f"{r.dataset_name}\n{r.variant}\n(n={r.n_states})" == label
                ),
                None,
            )
            if match is None:
                heights.append(np.nan)
            elif match.status == "success":
                heights.append(match.runtime_seconds)
            elif match.status == "timeout":
                heights.append(TIMEOUT_SECONDS)
            else:
                heights.append(np.nan)
        plt.bar(
            x + (idx - (len(algorithms) - 1) / 2) * width,
            heights,
            width=width,
            label=algorithm,
        )

    plt.xticks(x, case_labels, rotation=20, ha="right")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime by Dataset and Algorithm")
    plt.axhline(TIMEOUT_SECONDS, linestyle="--", linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_kldr_chart(records: List[BenchmarkRecord], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    success_records = [
        r for r in records if r.status == "success" and r.kldr is not None
    ]
    case_labels = _unique_case_labels(success_records)
    algorithms = sorted({r.algorithm for r in success_records})
    x = np.arange(len(case_labels))
    width = 0.18

    plt.figure(figsize=(14, 6))
    for idx, algorithm in enumerate(algorithms):
        heights = []
        for label in case_labels:
            match = next(
                (
                    r
                    for r in success_records
                    if r.algorithm == algorithm
                    and f"{r.dataset_name}\n{r.variant}\n(n={r.n_states})" == label
                ),
                None,
            )
            heights.append(np.nan if match is None else match.kldr)
        plt.bar(
            x + (idx - (len(algorithms) - 1) / 2) * width,
            heights,
            width=width,
            label=algorithm,
        )

    plt.xticks(x, case_labels, rotation=20, ha="right")
    plt.ylabel("KLDR")
    plt.title("KLDR by Dataset and Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_memory_chart(records: List[BenchmarkRecord], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    success_records = [
        r
        for r in records
        if r.status == "success" and r.memory_consumption_mb is not None
    ]
    case_labels = _unique_case_labels(success_records)
    algorithms = sorted({r.algorithm for r in success_records})
    x = np.arange(len(case_labels))
    width = 0.18

    plt.figure(figsize=(14, 6))
    for idx, algorithm in enumerate(algorithms):
        heights = []
        for label in case_labels:
            match = next(
                (
                    r
                    for r in success_records
                    if r.algorithm == algorithm
                    and f"{r.dataset_name}\n{r.variant}\n(n={r.n_states})" == label
                ),
                None,
            )
            heights.append(np.nan if match is None else match.memory_consumption_mb)
        plt.bar(
            x + (idx - (len(algorithms) - 1) / 2) * width,
            heights,
            width=width,
            label=algorithm,
        )

    plt.xticks(x, case_labels, rotation=20, ha="right")
    plt.ylabel("Peak memory (MB)")
    plt.title("Peak Memory Consumption by Dataset and Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_summary_md(records: List[BenchmarkRecord], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(f"- Timeout per run: {TIMEOUT_SECONDS} seconds\n")
        f.write(f"- Number of runs: {len(records)}\n\n")
        f.write(
            "| Dataset | Variant | n_states | Algorithm | Status | Runtime (s) | KLDR | Peak Memory (MB) | Matrix Cache |\n"
        )
        f.write("|---|---:|---:|---|---|---:|---:|---:|---|\n")
        for r in records:
            loss = "" if r.kldr is None else f"{r.kldr:.6f}"
            memory = (
                ""
                if r.memory_consumption_mb is None
                else f"{r.memory_consumption_mb:.3f}"
            )
            f.write(
                f"| {r.dataset_name} | {r.variant} | {r.n_states} | {r.algorithm} | "
                f"{r.status} | {r.runtime_seconds:.3f} | {loss} | {memory} | {r.matrix_path} |\n"
            )


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(CACHE_DIR)

    cases = build_cases()
    matrix_paths = materialize_all_case_matrices(cases)
    algorithms = list(algorithm_factories(K_MACRO).keys())

    records: List[BenchmarkRecord] = []
    for case in cases:
        matrix_path = matrix_paths[case.case_id]
        print(
            f"\n[Main] All algorithms for case={case.case_id} will use the same matrix: {matrix_path}"
        )
        for algorithm_name in algorithms:
            record = run_case_with_timeout(
                case=case,
                matrix_path=matrix_path,
                algorithm_name=algorithm_name,
                timeout_seconds=TIMEOUT_SECONDS,
                k_macro=K_MACRO if 'random' not in case.dataset_name else K_MACRO_SMALL,
            )
            records.append(record)
            print(
                f"[Main] Finished algorithm={algorithm_name} for case={case.case_id} "
                f"with status={record.status}, runtime={record.runtime_seconds:.3f}s"
            )

    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    runtime_chart_path = os.path.join(RESULTS_DIR, "runtime_chart.png")
    kldr_chart_path = os.path.join(RESULTS_DIR, "kldr_chart.png")
    memory_chart_path = os.path.join(RESULTS_DIR, "memory_chart.png")
    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.md")

    save_results_csv(records, csv_path)
    plot_runtime_chart(records, runtime_chart_path)
    plot_kldr_chart(records, kldr_chart_path)
    plot_memory_chart(records, memory_chart_path)
    save_summary_md(records, summary_path)

    print("\nBenchmark complete.")
    print(f"- CSV: {csv_path}")
    print(f"- Runtime chart: {runtime_chart_path}")
    print(f"- KLDR chart: {kldr_chart_path}")
    print(f"- Memory chart: {memory_chart_path}")
    print(f"- Summary: {summary_path}")


if __name__ == "__main__":
    main()
