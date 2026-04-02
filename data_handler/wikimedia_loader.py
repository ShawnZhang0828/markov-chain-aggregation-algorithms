import gzip
import numpy as np
from collections import Counter, defaultdict


def load_wikimedia_clickstream_markov_chain(
    clickstream_tsv_gz: str,
    n_states: int = 30,
    link_only: bool = True,
    min_count: int = 0,
    smoothing: float = 1e-8,
    top_by: str = "source_flow",
):
    # Read aggregated clickstream edges from the Wikimedia dump
    # Expected columns: source, target, type, count
    source_flow = Counter()
    target_flow = Counter()
    edge_counts = defaultdict(float)

    with gzip.open(clickstream_tsv_gz, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue

            src, dst, edge_type, count_str = parts

            try:
                count = float(count_str)
            except ValueError:
                continue

            if link_only and edge_type != "link":
                continue

            if count < min_count:
                continue

            source_flow[src] += count
            target_flow[dst] += count
            edge_counts[(src, dst)] += count

    if len(edge_counts) == 0:
        raise ValueError("No valid clickstream edges were loaded.")

    # Choose the pages to keep
    if top_by == "source_flow":
        top_pages = [p for p, _ in source_flow.most_common(n_states)]
    elif top_by == "target_flow":
        top_pages = [p for p, _ in target_flow.most_common(n_states)]
    elif top_by == "total_flow":
        total_flow = Counter()
        for p, c in source_flow.items():
            total_flow[p] += c
        for p, c in target_flow.items():
            total_flow[p] += c
        top_pages = [p for p, _ in total_flow.most_common(n_states)]
    else:
        raise ValueError("top_by must be one of: 'source_flow', 'target_flow', 'total_flow'.")

    if len(top_pages) < n_states:
        raise ValueError(
            f"Only found {len(top_pages)} pages, fewer than n_states={n_states}."
        )

    page_to_idx = {p: i for i, p in enumerate(top_pages)}
    top_page_set = set(top_pages)

    # Build the transition count matrix on the selected pages
    C = np.zeros((n_states, n_states), dtype=float)
    for (src, dst), w in edge_counts.items():
        if src in top_page_set and dst in top_page_set:
            i = page_to_idx[src]
            j = page_to_idx[dst]
            C[i, j] += w

    # Fix zero rows to avoid invalid normalization
    for i in range(n_states):
        if C[i].sum() == 0:
            C[i, i] = 1.0

    # Add tiny smoothing and normalize row-wise
    C += smoothing
    P = C / C.sum(axis=1, keepdims=True)

    return P