from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from peffects import inference, ols, simulate, twfe
from tests.conftest import panel_from


def brute_force_p(p, weighted: bool) -> float:  # type: ignore[no-untyped-def]
    """WCR p-value by re-running the full regression for every sign pattern."""
    d = p.treated.astype(float)
    w = p.weight[p.unit_index] if weighted else None
    groups = [p.unit_index, p.time_index]

    def t_of(y: np.ndarray) -> float:
        fit = ols.regress(
            y, d, ["d"], absorb=groups, weights=w, cluster=p.unit_index, nested=[True, False]
        )
        return fit["d"] / fit.se("d")

    u_r = ols.demean(p.y, groups, w)
    fitted = p.y - u_r
    t_obs = t_of(p.y)
    draws = [
        t_of(fitted + np.array(v)[p.unit_index] * u_r)
        for v in itertools.product([-1.0, 1.0], repeat=p.n_units)
    ]
    return float(np.mean(np.abs(draws) >= abs(t_obs) - 1e-12))


@pytest.mark.parametrize("weighted", [False, True])
def test_wild_bootstrap_matches_brute_force_enumeration(weighted: bool) -> None:
    frame = simulate.staggered(n_units=9, cohorts={2003.0: 0.34, 2006.0: 0.34}, seed=5)
    frame["w"] = 1.0 + frame["unit"] % 3
    p = panel_from(frame, weight="w")
    result = inference.wild_cluster_bootstrap(p, weighted=weighted, interval=False)
    assert result.enumerated and result.reps == 2**9
    assert result.p_value == pytest.approx(brute_force_p(p, weighted), abs=1e-12)
    fit = twfe.twfe(p, weighted=weighted)
    assert result.t == pytest.approx(fit["treated"] / fit.se("treated"), rel=1e-10)


def test_interval_inverts_the_test(noisy_frame: pd.DataFrame) -> None:
    p = panel_from(noisy_frame)
    r = inference.wild_cluster_bootstrap(p, reps=999)
    assert r.low < r.estimate < r.high
    assert not r.enumerated and r.clusters == p.n_units
    for kind in ("webb", "mammen"):
        other = inference.wild_cluster_bootstrap(p, reps=499, weights=kind, interval=False)  # type: ignore[arg-type]
        assert other.p_value < 0.05
    with pytest.raises(ValueError):
        inference.wild_cluster_bootstrap(p, reps=10, weights="bad", interval=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        inference.wild_cluster_bootstrap(p, test="x", interval=False)


def test_bootstrap_rejects_at_the_nominal_rate_under_the_null() -> None:
    rejections = 0
    for seed in range(60):
        frame = simulate.staggered(
            n_units=14, cohorts={2004.0: 0.3, 2007.0: 0.3}, effect=simulate.constant(0.0), seed=seed
        )
        r = inference.wild_cluster_bootstrap(panel_from(frame), reps=399, seed=seed, interval=False)
        rejections += r.p_value < 0.1
    assert rejections / 60 <= 0.2


def test_coarser_clusters_and_covariates() -> None:
    frame = simulate.staggered(n_units=40, cohorts={2004.0: 0.5}, seed=8)
    frame["x"] = np.random.default_rng(0).normal(size=len(frame))
    from peffects.panel import Panel

    p = Panel.from_frame(
        frame, unit="unit", time="time", outcome="y", cohort="cohort", covariates=["x"]
    )
    states = (np.arange(40) // 4)[p.unit_index]
    r = inference.wild_cluster_bootstrap(
        p, regressors=("treated", "x"), cluster=states, reps=999, interval=False
    )
    assert r.clusters == 10 and r.enumerated


def test_randomization_inference(noisy_frame: pd.DataFrame) -> None:
    p = panel_from(noisy_frame)

    def stat(q):  # type: ignore[no-untyped-def]
        return twfe.twfe(q)["treated"]

    result = inference.randomization_inference(p, stat, reps=99, seed=1)
    assert result.p_value == pytest.approx(0.01)
    null = panel_from(noisy_frame.assign(y=noisy_frame.y0))
    assert inference.randomization_inference(null, stat, reps=99, seed=1).p_value > 0.05
