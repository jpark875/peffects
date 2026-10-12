from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from peffects import cs, simulate, sunab
from tests.conftest import panel_from


def test_recovers_every_group_time_effect_without_noise(dynamic_frame: pd.DataFrame) -> None:
    gt = cs.att_gt(panel_from(dynamic_frame), reps=0)
    truth = simulate.true_att(dynamic_frame).set_index(["cohort", "time"])["att"]
    post = gt.table[gt.table["post"]].set_index(["group", "time"])["att"]
    assert len(post) == len(truth)
    assert np.allclose(post.sort_index(), truth.sort_index(), atol=1e-9)
    assert np.allclose(gt.table.loc[~gt.table["post"], "att"], 0, atol=1e-9)


@pytest.mark.parametrize("control", ["never", "notyet"])
def test_aggregates_match_the_truth(dynamic_frame: pd.DataFrame, control: cs.Control) -> None:
    gt = cs.att_gt(panel_from(dynamic_frame), control=control, reps=0)
    treated = dynamic_frame[dynamic_frame.time >= dynamic_frame.cohort]
    simple = cs.aggregate(gt, "simple", reps=0).overall.estimate
    assert simple == pytest.approx(treated["effect"].mean(), abs=1e-9)
    dynamic = cs.aggregate(gt, "dynamic", reps=0).table.set_index("key")["att"]
    by_e = treated.assign(e=treated.time - treated.cohort).groupby("e")["effect"].mean()
    assert np.allclose(dynamic.loc[by_e.index], by_e, atol=1e-9)
    group = cs.aggregate(gt, "group", reps=0)
    assert set(group.table["key"]) == {2003.0, 2006.0}
    calendar = cs.aggregate(gt, "calendar", reps=0)
    assert calendar.table["key"].min() == 2003.0


def test_standard_errors_have_nominal_coverage() -> None:
    hits, n = 0, 150
    for seed in range(n):
        frame = simulate.staggered(
            n_units=120,
            cohorts={2004.0: 0.3, 2007.0: 0.3},
            effect=simulate.constant(1.0),
            seed=seed,
        )
        est = cs.aggregate(cs.att_gt(panel_from(frame), reps=0), "simple", reps=0).overall
        hits += est.low <= 1.0 <= est.high
    assert 0.89 <= hits / n <= 0.99


def test_uniform_bands_are_wider_than_pointwise(noisy_frame: pd.DataFrame) -> None:
    gt = cs.att_gt(panel_from(noisy_frame), reps=500)
    crit = (gt.table["high"] - gt.table["att"]) / gt.table["se"]
    assert (crit > 1.96).all()


def test_covariates_change_little_when_irrelevant(noisy_frame: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    frame = noisy_frame.assign(x=rng.normal(size=len(noisy_frame)))
    from peffects.panel import Panel

    p = Panel.from_frame(
        frame, unit="unit", time="time", outcome="y", cohort="cohort", covariates=["x"]
    )
    plain = cs.aggregate(cs.att_gt(p, reps=0), reps=0).overall
    adjusted = cs.aggregate(cs.att_gt(p, covariates=True, reps=0), reps=0).overall
    assert adjusted.estimate == pytest.approx(plain.estimate, abs=0.1)
    with pytest.raises(ValueError, match="no covariates"):
        cs.att_gt(panel_from(noisy_frame), covariates=True)


def test_covariate_adjustment_removes_confounded_trends() -> None:
    frame = simulate.staggered(n_units=400, cohorts={2005.0: 0.4}, noise=0.2, seed=3)
    rng = np.random.default_rng(1)
    treated = np.isfinite(frame.groupby("unit")["cohort"].first().to_numpy())
    unit_x = rng.normal(size=400) + 1.5 * treated
    frame["x"] = unit_x[frame["unit"].to_numpy()]
    frame["y"] += 0.3 * frame["x"] * (frame["time"] - 2000)  # trend depends on x
    from peffects.panel import Panel

    p = Panel.from_frame(
        frame, unit="unit", time="time", outcome="y", cohort="cohort", covariates=["x"]
    )
    plain = cs.aggregate(cs.att_gt(p, reps=0), reps=0).overall.estimate
    adjusted = cs.aggregate(cs.att_gt(p, covariates=True, reps=0), reps=0).overall.estimate
    assert abs(adjusted - 1.0) < 0.1 < abs(plain - 1.0)


def test_input_errors(noisy_frame: pd.DataFrame) -> None:
    p = panel_from(noisy_frame)
    with pytest.raises(ValueError):
        cs.att_gt(p, control="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cs.att_gt(p, base="bad")  # type: ignore[arg-type]
    everyone = panel_from(noisy_frame.assign(cohort=noisy_frame.cohort.replace(np.inf, 2009.0)))
    with pytest.raises(ValueError, match="never-treated"):
        cs.att_gt(everyone)
    assert cs.att_gt(everyone, control="notyet", reps=0).table["att"].notna().all()
    with pytest.raises(ValueError):
        cs.aggregate(cs.att_gt(p, reps=0), "bad", reps=0)  # type: ignore[arg-type]


def test_unbalanced_units_and_always_treated_are_dropped(noisy_frame: pd.DataFrame) -> None:
    frame = noisy_frame.iloc[3:].copy()
    frame.loc[frame.unit == 5, "cohort"] = 1990.0
    gt = cs.att_gt(panel_from(frame), reps=0)
    assert gt.dropped_units >= 2


def test_sun_abraham_equals_cs_with_a_fixed_base(noisy_frame: pd.DataFrame) -> None:
    for weighted in (False, True):
        frame = noisy_frame.assign(w=1.0 + noisy_frame.unit % 3)
        p = panel_from(frame, weight="w")
        if not weighted:
            p = replace(p, weight=np.ones(p.n_units))
        sa = sunab.sun_abraham(p)
        gt = cs.att_gt(p, base="universal", reps=0)
        ours = sa.cohort_effects.set_index(["group", "e"])["catt"]
        ref = gt.table.set_index(["group", "e"])["att"].reindex(ours.index)
        assert np.allclose(ours, ref, atol=1e-10)
        simple = cs.aggregate(gt, "simple", reps=0).overall.estimate
        assert sa.overall.estimate == pytest.approx(simple, abs=1e-10)


def test_sun_abraham_uses_the_last_cohort_without_never_treated() -> None:
    frame = simulate.staggered(
        n_units=150, cohorts={2003.0: 0.3, 2005.0: 0.3, 2008.0: 0.4}, noise=0.0, seed=5
    )
    sa = sunab.sun_abraham(panel_from(frame))
    assert "last cohort" in sa.control
    post = sa.event_study.post()
    assert np.allclose(post["coef"], 1.0, atol=1e-9)
    assert set(sa.cohort_effects["group"]) == {2003.0, 2005.0}
