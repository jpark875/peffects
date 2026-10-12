from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peffects import ols, simulate, twfe
from tests.conftest import panel_from


def dummies(codes: np.ndarray) -> np.ndarray:
    return (codes[:, None] == np.unique(codes)[None, 1:]).astype(float)


def test_absorbed_regression_matches_explicit_dummies(noisy_frame: pd.DataFrame) -> None:
    p = panel_from(noisy_frame)
    d = p.treated.astype(float)
    w = 1 + (p.unit_index % 3).astype(float)
    fit = ols.regress(p.y, d, ["d"], absorb=[p.unit_index, p.time_index], weights=w)
    design = np.column_stack([np.ones(p.n_obs), d, dummies(p.unit_index), dummies(p.time_index)])
    root = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * root[:, None], p.y * root, rcond=None)
    assert fit["d"] == pytest.approx(coef[1], abs=1e-9)


def test_cluster_se_matches_the_sandwich_formula(noisy_frame: pd.DataFrame) -> None:
    p = panel_from(noisy_frame)
    d = p.treated.astype(float)
    fit = ols.regress(
        p.y,
        d,
        ["d"],
        absorb=[p.unit_index, p.time_index],
        cluster=p.unit_index,
        nested=[True, False],
    )
    dt = ols.demean(d, [p.unit_index, p.time_index])
    u = fit.resid
    scores = np.bincount(p.unit_index, weights=dt * u)
    g, n, k = p.n_units, p.n_obs, 1 + p.n_times
    manual = np.sqrt(g / (g - 1) * (n - 1) / (n - k) * np.sum(scores**2)) / np.sum(dt**2)
    assert fit.se("d") == pytest.approx(manual, rel=1e-10)
    assert fit.df == g - 1
    table = fit.table()
    assert table.loc["d", "low"] < fit["d"] < table.loc["d", "high"]
    with pytest.raises(KeyError):
        fit.se("nope")


def test_collinear_regressor_is_reported(noisy_frame: pd.DataFrame) -> None:
    p = panel_from(noisy_frame)
    with pytest.raises(np.linalg.LinAlgError, match="collinear"):
        ols.regress(p.y, p.time.copy(), ["t"], absorb=[p.unit_index, p.time_index])


def test_against_pyfixest(noisy_frame: pd.DataFrame) -> None:
    pf = pytest.importorskip("pyfixest")
    frame = noisy_frame.assign(d=(noisy_frame.time >= noisy_frame.cohort).astype(float))
    frame["w"] = 1.0 + frame["unit"] % 4
    p = panel_from(frame, weight="w")
    for cluster in ("unit", "time"):
        fit = twfe.twfe(p, cluster=cluster)
        ref = pf.feols("y ~ d | unit + time", data=frame, vcov={"CRV1": cluster}, weights="w")
        assert fit["treated"] == pytest.approx(float(ref.coef().iloc[0]), abs=1e-10)
        assert fit.se("treated") == pytest.approx(float(ref.se().iloc[0]), rel=1e-8)


def test_twfe_recovers_a_constant_effect(noisy_frame: pd.DataFrame) -> None:
    fit = twfe.twfe(panel_from(noisy_frame))
    assert abs(fit["treated"] - 1.0) < 3 * fit.se("treated")


def test_twfe_is_biased_when_effects_grow(dynamic_frame: pd.DataFrame) -> None:
    p = panel_from(dynamic_frame)
    true_att = dynamic_frame.loc[dynamic_frame.time >= dynamic_frame.cohort, "effect"].mean()
    assert abs(twfe.twfe(p)["treated"] - true_att) > 0.3


def test_event_study_recovers_dynamic_effects_without_cohort_differences() -> None:
    frame = simulate.staggered(
        n_units=300, cohorts={2003.0: 0.3, 2006.0: 0.3}, effect=simulate.dynamic(0.5), noise=0.0
    )
    es = twfe.event_study(panel_from(frame))
    post = es.post().set_index("e")["coef"]
    for e, value in post.items():
        assert value == pytest.approx(0.5 * (e + 1), abs=1e-8)
    assert np.allclose(es.pre()["coef"], 0, atol=1e-8)
    ref = es.estimates.set_index("e").loc[-1]
    assert ref["coef"] == 0 and np.isnan(ref["se"])


def test_event_study_binning_and_identification() -> None:
    frame = simulate.staggered(n_units=120, cohorts={2003.0: 0.5, 2006.0: 0.5}, seed=4)
    es = twfe.event_study(panel_from(frame), leads=2, lags=2)
    assert es.estimates["e"].tolist() == [-1, 0, 1, 2]  # no never-treated: farthest lead dropped
    with pytest.raises(ValueError):
        twfe.event_study(panel_from(frame), reference=5, lags=2)
    with pytest.raises(ValueError):
        twfe.twfe(panel_from(frame.assign(cohort=np.inf)))
