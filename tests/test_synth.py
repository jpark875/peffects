from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peffects import synth


def test_simplex_weights_recover_an_exact_convex_combination() -> None:
    rng = np.random.default_rng(0)
    donors = rng.normal(size=(12, 8))
    true = np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0])
    w = synth.simplex_weights(donors @ true, donors, np.ones(12))
    assert np.allclose(w, true, atol=1e-6)
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()


def test_simplex_weights_satisfy_kkt_when_target_is_outside_the_hull() -> None:
    rng = np.random.default_rng(1)
    donors = rng.normal(size=(6, 10))
    target = donors.max(axis=1) + 1.0
    v = rng.uniform(0.5, 2, 6)
    w = synth.simplex_weights(target, donors, v)
    grad = donors.T @ (v * (donors @ w - target))
    active = w > 1e-8
    # At the optimum every used donor has the same gradient, and unused ones no lower.
    assert np.ptp(grad[active]) < 1e-5 * max(1.0, np.abs(grad).max())
    assert (grad[~active] >= grad[active].min() - 1e-6).all()


def panel(effect: float = -10.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.arange(1980, 2001)
    rows = []
    factors = rng.normal(size=(len(years), 2)).cumsum(axis=0)
    loadings = rng.uniform(0.5, 1.5, size=(10, 2))
    for j in range(10):
        y = 50 + factors @ loadings[j] + rng.normal(0, 0.2, len(years))
        rows += [
            {"unit": f"u{j}", "year": t, "y": v, "x": loadings[j, 0]}
            for t, v in zip(years, y, strict=True)
        ]
    frame = pd.DataFrame(rows)
    blend = frame[frame.unit.isin(["u1", "u2", "u3"])].groupby("year")["y"].mean().to_numpy()
    treated = blend + np.where(years >= 1995, effect, 0.0)
    extra = pd.DataFrame({"unit": "T", "year": years, "y": treated, "x": loadings[1:4, 0].mean()})
    return pd.concat([frame, extra], ignore_index=True)


def test_synthetic_control_finds_the_effect_and_placebos_rank_it_first() -> None:
    frame = panel()
    fit = synth.synth(frame, unit="unit", time="year", outcome="y", treated="T", adoption=1995)
    assert fit.rmspe(False) < 0.5
    assert fit.gap[~fit.pre].mean() == pytest.approx(-10.0, abs=1.0)
    assert set(fit.weight_table(0.05)["donor"]) <= {"u1", "u2", "u3"}
    test = synth.placebo_test(frame, fit, unit="unit", time="year", outcome="y")
    assert test.ratios().index[0] == "T"
    assert test.p_value == pytest.approx(1 / 11)
    assert test.gaps().shape == (21, 11)
    assert test.gaps(max_pre_ratio=1e-9).shape[1] == 1


def test_predictors_with_optimized_importance() -> None:
    frame = panel(seed=3)
    preds = [
        synth.Predictor("x", 1980, 1994),
        synth.Predictor("y", 1990),
        synth.Predictor("y", 1985),
    ]
    fit = synth.synth(
        frame, unit="unit", time="year", outcome="y", treated="T", adoption=1995, predictors=preds
    )
    assert fit.v.sum() == pytest.approx(1.0)
    assert list(fit.predictors.index) == ["x (1980-1994)", "y (1990)", "y (1985)"]
    assert fit.ratio > 3


def test_synth_input_errors() -> None:
    frame = panel()
    kw = {"unit": "unit", "time": "year", "outcome": "y"}
    with pytest.raises(ValueError, match="not a unit"):
        synth.synth(frame, treated="nope", adoption=1995, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two donors"):
        synth.synth(frame, treated="T", adoption=1995, donors=["u1"], **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pre-treatment"):
        synth.synth(frame, treated="T", adoption=1980, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing"):
        synth.synth(
            frame,
            treated="T",
            adoption=1995,
            predictors=[synth.Predictor("y", 1970)],
            **kw,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="observed"):
        synth.synth(frame.iloc[1:], treated="T", adoption=1995, **kw)  # type: ignore[arg-type]
