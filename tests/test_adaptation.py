"""Tests for the training-task ability prior."""

from __future__ import annotations

import numpy as np

from aims_starter.adaptation import fit_ability_prior


def _synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    n_benchmarks, per_benchmark, dimension = 20, 3, 16
    direction = rng.normal(size=dimension)
    direction /= np.linalg.norm(direction)
    benchmark_effect = rng.normal(scale=0.6, size=n_benchmarks)
    embeddings = rng.normal(size=(n_benchmarks * per_benchmark, dimension))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    abilities, benchmarks, subjects = [], [], []
    index = {}
    for b in range(n_benchmarks):
        for t in range(per_benchmark):
            subject = f"subject_{b}_{t}"
            index[subject] = len(index)
            abilities.append(
                benchmark_effect[b] + 0.8 * float(embeddings[index[subject]] @ direction)
            )
            benchmarks.append(f"benchmark_{b}")
            subjects.append(subject)
    return (
        np.asarray(abilities),
        benchmarks,
        subjects,
        embeddings,
        index,
        benchmark_effect,
        direction,
    )


def test_subject_deviation_improves_held_out_tasks() -> None:
    abilities, benchmarks, subjects, embeddings, index, _, _ = _synthetic()
    train = [i for i, subject in enumerate(subjects) if not subject.endswith("_2")]
    test = [i for i, subject in enumerate(subjects) if subject.endswith("_2")]
    prior = fit_ability_prior(
        abilities[train],
        [benchmarks[i] for i in train],
        [subjects[i] for i in train],
        embeddings,
        index,
    )
    assert prior.n_fit > 0
    assert prior.weights.size == embeddings.shape[1]
    benchmark_only = []
    with_subject = []
    for i in test:
        key = benchmarks[i]
        embedding = embeddings[index[subjects[i]]]
        benchmark_only.append(abs(prior.predict(key, embedding, subject=False) - abilities[i]))
        with_subject.append(abs(prior.predict(key, embedding, subject=True) - abilities[i]))
    assert np.mean(with_subject) < np.mean(benchmark_only)


def test_singleton_benchmarks_carry_no_subject_rows() -> None:
    abilities = np.array([0.0, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, -1.0])
    benchmarks = ["pair0", "pair0", "pair1", "pair1", "pair2", "pair2", "pair3", "pair3", "single"]
    subjects = [f"s{i}" for i in range(len(abilities))]
    embeddings = np.eye(len(subjects))
    prior = fit_ability_prior(
        abilities, benchmarks, subjects, embeddings, {name: i for i, name in enumerate(subjects)}
    )
    # The singleton benchmark contributes no row; four pair benchmarks contribute two each.
    assert prior.n_fit == 8
    assert prior.n_benchmarks == 5


def test_unseen_benchmark_falls_back_to_global_mean() -> None:
    abilities, benchmarks, subjects, embeddings, index, _, _ = _synthetic()
    prior = fit_ability_prior(abilities, benchmarks, subjects, embeddings, index)
    value = prior.predict("unseen", embeddings[0], subject=False)
    assert value == np.clip(prior.global_mean, -4.0, 4.0)


def test_prior_mean_is_clipped_to_map_range() -> None:
    abilities, benchmarks, subjects, embeddings, index, _, _ = _synthetic()
    prior = fit_ability_prior(abilities, benchmarks, subjects, embeddings, index)
    extreme = 1000.0 * embeddings[0]
    assert -4.0 <= prior.predict(benchmarks[0], extreme, subject=True) <= 4.0


def test_numpy_ability_map_matches_the_torch_anchor() -> None:
    import torch

    from aims_starter.adaptation import map_ability
    from aims_starter.faithful_np import CNPAIRT

    torch.manual_seed(0)
    model = CNPAIRT(12, d_model=8, subject_feature_slice=(4, 6))
    x = torch.randn(9, 12)
    x[:, 4:6] = 1.0
    y = (torch.rand(9) < 0.5).float()
    with torch.no_grad():
        encoded = model.encode_item_x(x)
        torch_theta = float(model.map_theta(encoded, y, prior_mean=0.3))
        raw = model.irt_parameter_net(encoded)
        a = (torch.nn.functional.softplus(raw[:, 0]) + 0.05).numpy()
        b = raw[:, 1].numpy()
    numpy_theta = map_ability(a, b, y.numpy(), prior_mean=0.3)
    assert abs(numpy_theta - torch_theta) < 1e-4
