from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peffects.panel import Panel, PanelError


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "year": [1, 2, 3] * 3,
            "y": np.arange(9.0),
            "d": [0, 1, 1, 0, 0, 0, 0, 0, 1],
            "g": [2, 2, 2, 0, 0, 0, 3, 3, 3],
            "w": [2.0] * 3 + [1.0] * 6,
        }
    )


def test_cohorts_from_indicator_and_column_agree() -> None:
    a = Panel.from_frame(frame(), unit="id", time="year", outcome="y", treat="d")
    b = Panel.from_frame(frame(), unit="id", time="year", outcome="y", cohort="g")
    assert a.cohort.tolist() == [2.0, np.inf, 3.0]
    assert np.array_equal(a.cohort, b.cohort)
    assert a.treated.tolist() == [False, True, True, False, False, False, False, False, True]
    assert a.balanced and a.cohorts.tolist() == [2.0, 3.0]


def test_rows_are_sorted_and_wide_matches() -> None:
    shuffled = frame().sample(frac=1, random_state=0)
    p = Panel.from_frame(shuffled, unit="id", time="year", outcome="y", cohort="g")
    assert np.array_equal(p.wide(), np.arange(9.0).reshape(3, 3))


def test_validation_errors() -> None:
    bad = frame()
    bad.loc[2, "d"] = 0  # treated then untreated
    with pytest.raises(PanelError, match="absorbing"):
        Panel.from_frame(bad, unit="id", time="year", outcome="y", treat="d")
    with pytest.raises(PanelError, match="more than one row"):
        Panel.from_frame(
            pd.concat([frame(), frame().head(1)]), unit="id", time="year", outcome="y", cohort="g"
        )
    varying = frame()
    varying.loc[0, "w"] = 5.0
    with pytest.raises(PanelError, match="constant"):
        Panel.from_frame(varying, unit="id", time="year", outcome="y", cohort="g", weight="w")
    with pytest.raises(PanelError, match="exactly one"):
        Panel.from_frame(frame(), unit="id", time="year", outcome="y")
    with pytest.raises(PanelError, match="not found"):
        Panel.from_frame(frame(), unit="id", time="year", outcome="nope", cohort="g")
    missing = frame()
    missing.loc[0, "y"] = np.nan
    with pytest.raises(PanelError, match="missing"):
        Panel.from_frame(missing, unit="id", time="year", outcome="y", cohort="g")


def test_late_adopters_are_never_treated_in_sample() -> None:
    f = frame()
    f.loc[f["id"] == "c", "g"] = 9
    p = Panel.from_frame(f, unit="id", time="year", outcome="y", cohort="g")
    assert np.isinf(p.cohort[2])


def test_subsetting_reindexes() -> None:
    p = Panel.from_frame(frame(), unit="id", time="year", outcome="y", cohort="g", weight="w")
    q = p.keep_units(np.array([False, True, True]))
    assert q.units.tolist() == ["b", "c"] and q.n_obs == 6
    assert q.unit_index.max() == 1 and q.weight.tolist() == [1.0, 1.0]
    r = p.keep_times(np.array([True, True, False]))
    assert r.n_times == 2 and r.time_index.max() == 1
    assert r.to_frame()["time"].max() == 2
    with pytest.raises(PanelError):
        p.keep_units(np.array([True]))
    with pytest.raises(PanelError):
        p.with_cohorts(np.array([1.0]))
    assert p.always_treated.tolist() == [False, False, False]
