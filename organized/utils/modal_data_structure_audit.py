"""CPU-only audit of response spectra and Qwen item-neighbour structure.

This script describes the filtered training data. It does not train a model,
evaluate a submission, or use an accelerator.

Run:
    MODAL_PROFILE=agrover112 modal run --detach modal_data_structure_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "aims-neurips-data-structure-audit"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
QWEN_STORE = Path("/data/datasets/full/allbench-embed-qwen3_8b-v2")
OUT = Path("/data/results/data-analysis/data-structure-audit")
MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
K_NEIGHBOURS = 20
N_SPECTRAL_VALUES = 32

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scipy>=1.13,<2", "faiss-cpu>=1.9,<2"
)


def _filtered_observations(pd):
    """The current CNP data contract, without filling missing responses."""
    obs = pd.read_parquet(
        PROCESSED / "observations.parquet",
        columns=["subject_uid", "item_uid", "benchmark_id", "label"],
    )
    raw_rows = len(obs)
    obs = obs.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)["label"].mean()
    obs["label"] = (obs["label"] >= 0.5).astype("int8")
    grouped_rows = len(obs)

    items = pd.read_parquet(PROCESSED / "items.parquet", columns=["item_uid", "content_hash"])
    obs = obs.merge(items, on="item_uid", how="left", validate="many_to_one")
    obs = obs.drop_duplicates(["content_hash", "subject_uid"])
    dedup_rows = len(obs)

    variable = obs.groupby("item_uid")["label"].mean()
    obs = obs[obs["item_uid"].isin(variable[(variable > 0) & (variable < 1)].index)]
    n_subjects = obs.groupby("item_uid")["subject_uid"].nunique()
    obs = obs[obs["item_uid"].isin(n_subjects[n_subjects >= MIN_SUBJECTS].index)]
    n_items = obs.groupby("subject_uid")["item_uid"].nunique()
    obs = obs[obs["subject_uid"].isin(n_items[n_items >= MIN_ITEMS_PER_TASK].index)].copy()
    audit = {
        "raw_rows": int(raw_rows),
        "deduplicated_subject_item_rows": int(grouped_rows),
        "after_content_duplicate_rows": int(dedup_rows),
        "final_rows": int(len(obs)),
        "min_subjects_per_item": MIN_SUBJECTS,
        "min_responses_per_subject_task": MIN_ITEMS_PER_TASK,
    }
    return obs, audit


def _summary(frame, benchmark_id, np, pd, sparse, connected_components):
    subject_codes, subjects = pd.factorize(frame["subject_uid"], sort=True)
    item_codes, items = pd.factorize(frame["item_uid"], sort=True)
    coverage = sparse.csr_matrix(
        (np.ones(len(frame), dtype=np.int8), (subject_codes, item_codes)),
        shape=(len(subjects), len(items)),
    )
    graph = sparse.bmat([[None, coverage], [coverage.T, None]], format="csr")
    n_components, _ = connected_components(graph, directed=False, return_labels=True)
    subject_degree = np.asarray(coverage.sum(axis=1)).ravel()
    item_degree = np.asarray(coverage.sum(axis=0)).ravel()
    summary = {
        "subjects": int(len(subjects)), "items": int(len(items)), "responses": int(len(frame)),
        "density": float(coverage.nnz / (len(subjects) * len(items))),
        "pass_rate": float(frame["label"].mean()), "components": int(n_components),
        "subject_degree_median": float(np.median(subject_degree)),
        "item_degree_median": float(np.median(item_degree)),
    }
    degrees = pd.DataFrame({
        "benchmark_id": benchmark_id,
        "node_type": ["subject"] * len(subject_degree) + ["item"] * len(item_degree),
        "degree": np.concatenate([subject_degree, item_degree]),
    })
    return summary, degrees


def _response_spectrum(frame, scope, np, pd, sparse, svds):
    """Leading singular values of the observed, degree-normalised residual operator."""
    rows, task_ids = pd.factorize(frame["task_uid"], sort=True)
    cols, item_ids = pd.factorize(frame["item_uid"], sort=True)
    item_mean = frame.groupby("item_uid")["label"].transform("mean").to_numpy(float)
    row_degree = np.bincount(rows, minlength=len(task_ids)).astype(float)
    col_degree = np.bincount(cols, minlength=len(item_ids)).astype(float)
    values = (frame["label"].to_numpy(float) - item_mean) / np.sqrt(row_degree[rows] * col_degree[cols])
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(task_ids), len(item_ids)))
    max_rank = min(N_SPECTRAL_VALUES, min(matrix.shape) - 1)
    total_energy = float(np.dot(values, values))
    if max_rank < 1 or total_energy == 0:
        return []
    singular = np.sort(svds(matrix, k=max_rank, which="LM", return_singular_vectors=False))[::-1]
    return [
        {"scope": scope, "rank": rank, "singular_value": float(value),
         "eigenvalue": float(value * value), "energy_fraction": float(value * value / total_energy)}
        for rank, value in enumerate(singular, start=1)
    ]


def _qwen_knn(obs, np, pd, faiss, sparse, connected_components, eigsh):
    """Build a content-only kNN graph from frozen Qwen item embeddings."""
    qwen_items = pd.read_parquet(QWEN_STORE / "items.parquet", columns=["item_uid"])
    qwen_items["embedding_index"] = np.arange(len(qwen_items), dtype=np.int64)
    item_info = (
        obs.groupby(["item_uid", "benchmark_id"], as_index=False)["label"].mean()
        .rename(columns={"label": "item_pass_rate"})
        .merge(qwen_items, on="item_uid", how="left", validate="one_to_one")
    )
    if item_info["embedding_index"].isna().any():
        raise RuntimeError("Filtered item missing from Qwen3-8B v2 embedding store")

    embeddings = np.load(QWEN_STORE / "item_embeddings.npy", mmap_mode="r")
    vectors = np.asarray(embeddings[item_info["embedding_index"].to_numpy(np.int64)], dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    faiss.omp_set_num_threads(16)
    index.add(vectors)
    score, neighbour = index.search(vectors, K_NEIGHBOURS + 1)
    score, neighbour = score[:, 1:], neighbour[:, 1:]
    source_index = np.repeat(np.arange(len(item_info)), K_NEIGHBOURS)
    target_index = neighbour.ravel()
    directed = sparse.csr_matrix((score.ravel(), (source_index, target_index)),
                                 shape=(len(item_info), len(item_info)))
    topology = directed.astype(bool).astype(np.int8)
    inbound = np.asarray(topology.sum(axis=0)).ravel()
    reciprocal_fraction = float(topology.multiply(topology.T).nnz / topology.nnz)
    weak_components, weak_labels = connected_components(topology, directed=True, connection="weak")
    strong_components, _ = connected_components(topology, directed=True, connection="strong")
    # Cosine ranks the neighbours. The Laplacian uses the resulting undirected
    # kNN topology, which is the stable sparse-graph form for this audit.
    undirected = topology.maximum(topology.T).astype(float).tocsr()
    graph_strength = np.asarray(undirected.sum(axis=1)).ravel()
    laplacian = sparse.eye(len(item_info), format="csr") - sparse.diags(1 / np.sqrt(graph_strength)) @ undirected @ sparse.diags(1 / np.sqrt(graph_strength))
    component_sizes = np.bincount(weak_labels)
    largest_component = int(component_sizes.argmax())
    largest_index = np.flatnonzero(weak_labels == largest_component)
    largest_laplacian = laplacian[largest_index][:, largest_index]
    largest_values_full, largest_vectors_full = eigsh(largest_laplacian, k=32, which="SM", tol=1e-4)
    largest_order_full = np.argsort(largest_values_full)
    largest_values = largest_values_full[largest_order_full[:8]]
    largest_vectors = largest_vectors_full[:, largest_order_full[:8]]
    spectrum_parts = [largest_values_full]
    for component in range(weak_components):
        if component == largest_component:
            continue
        indices = np.flatnonzero(weak_labels == component)
        spectrum_parts.append(np.linalg.eigvalsh(laplacian[indices][:, indices].toarray()))
    laplacian_values = np.sort(np.concatenate(spectrum_parts))[:32]
    fiedler = np.full(len(item_info), np.nan, dtype=np.float32)
    fiedler[largest_index] = largest_vectors[:, 1].astype(np.float32)

    source_benchmark = item_info["benchmark_id"].to_numpy()
    source_rate = item_info["item_pass_rate"].to_numpy(float)
    target_benchmark = source_benchmark[neighbour]
    target_rate = source_rate[neighbour]
    same = target_benchmark == source_benchmark[:, None]
    abs_gap = np.abs(target_rate - source_rate[:, None])

    item_graph = item_info[["item_uid", "benchmark_id", "item_pass_rate"]].copy()
    item_graph["mean_knn_cosine"] = score.mean(axis=1)
    item_graph["out_degree"] = K_NEIGHBOURS
    item_graph["in_degree"] = inbound
    item_graph["weak_component"] = weak_labels
    item_graph["cross_benchmark_neighbour_fraction"] = 1.0 - same.mean(axis=1)
    item_graph["mean_knn_pass_rate_gap"] = abs_gap.mean(axis=1)

    pair_rows, rng = [], np.random.default_rng(20261027)
    benchmarks = sorted(item_info["benchmark_id"].unique())
    for source in benchmarks:
        source_index = np.flatnonzero(source_benchmark == source)
        for target in benchmarks:
            edge_mask = (source_benchmark[:, None] == source) & (target_benchmark == target)
            edge_values = abs_gap[edge_mask]
            edge_scores = score[edge_mask]
            if len(edge_values) == 0:
                continue
            target_index = np.flatnonzero(source_benchmark == target)
            random_target = rng.choice(target_index, size=len(source_index) * K_NEIGHBOURS, replace=True)
            random_source = np.repeat(source_index, K_NEIGHBOURS)
            pair_rows.append({
                "source_benchmark": source, "target_benchmark": target, "edge_count": int(len(edge_values)),
                "neighbour_probability": float(len(edge_values) / (len(source_index) * K_NEIGHBOURS)),
                "mean_cosine": float(edge_scores.mean()), "mean_pass_rate_gap": float(edge_values.mean()),
                "random_pass_rate_gap": float(np.abs(source_rate[random_source] - source_rate[random_target]).mean()),
            })
    laplacian_spectrum = pd.DataFrame({"rank": np.arange(1, len(laplacian_values) + 1), "eigenvalue": laplacian_values})
    fiedler_frame = item_graph[["item_uid", "benchmark_id", "item_pass_rate", "weak_component"]].copy()
    fiedler_frame["fiedler_value"] = fiedler
    return item_graph, pd.DataFrame(pair_rows), laplacian_spectrum, {
        "embedding_model": "Qwen/Qwen3-Embedding-8B", "native_dimension": 4096,
        "graph_dimension": int(vectors.shape[1]), "mrl_truncated": True, "l2_normalized": True,
        "neighbours_per_item": K_NEIGHBOURS, "filtered_items_with_embeddings": int(len(item_info)),
        "directed_edges": int(directed.nnz), "reciprocal_edge_fraction": reciprocal_fraction,
        "weak_components": int(weak_components), "strong_components": int(strong_components),
        "edge_weight": "binary symmetrised kNN adjacency; cosine selects the neighbours",
        "largest_weak_component_items": int(len(largest_index)),
        "largest_component_laplacian_eigenvalues": [float(x) for x in largest_values],
    }, fiedler_frame


@app.function(image=image, cpu=16, memory=131_072, volumes={"/data": volume}, timeout=4 * 60 * 60)
def run() -> dict:
    import faiss
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh, svds

    obs, audit = _filtered_observations(pd)
    obs["task_uid"] = obs["benchmark_id"].astype(str) + "::" + obs["subject_uid"].astype(str)
    summaries, degree_frames = [], []
    spectra = _response_spectrum(obs, "all benchmarks", np, pd, sparse, svds)
    for benchmark_id, frame in obs.groupby("benchmark_id", sort=True):
        summary, degrees = _summary(frame, str(benchmark_id), np, pd, sparse, connected_components)
        summaries.append({"benchmark_id": str(benchmark_id), **summary})
        degree_frames.append(degrees)
        spectra.extend(_response_spectrum(frame, str(benchmark_id), np, pd, sparse, svds))
    item_graph, graph_pairs, laplacian_spectrum, graph_meta, fiedler_frame = _qwen_knn(
        obs, np, pd, faiss, sparse, connected_components, eigsh
    )

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).sort_values("responses", ascending=False).to_parquet(OUT / "benchmarks.parquet", index=False)
    pd.DataFrame(spectra).to_parquet(OUT / "response_spectra.parquet", index=False)
    pd.concat(degree_frames, ignore_index=True).to_parquet(OUT / "response_degrees.parquet", index=False)
    item_graph.to_parquet(OUT / "qwen_item_knn.parquet", index=False)
    graph_pairs.to_parquet(OUT / "qwen_knn_by_benchmark_pair.parquet", index=False)
    laplacian_spectrum.to_parquet(OUT / "qwen_laplacian_spectrum.parquet", index=False)
    fiedler_frame.to_parquet(OUT / "qwen_fiedler.parquet", index=False)
    (OUT / "audit.json").write_text(json.dumps({**audit, "qwen_graph": graph_meta}, indent=2), encoding="utf-8")
    volume.commit()
    return {**audit, "benchmarks": len(summaries), "qwen_graph": graph_meta, "output": str(OUT)}


@app.local_entrypoint()
def main() -> None:
    print(run.remote())
