from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peffects import bacon, simulate, twfe
from tests.conftest import panel_from


@pytest.mark.parametrize(
    "cohorts",
    [
        {2003.0: 0.3, 2006.0: 0.3},
        {2002.0: 0.2, 2004.0: 0.2, 2007.0: 0.3},
        {1990.0: 0.2, 2003.0: 0.3, 2006.0: 0.2, 2008.0: 0.3},  # always treated, no never
    ],
)
def test_weights_sum_to_one_and_reproduce_twfe(cohorts: dict[float, float]) -> None:
    frame = simulate.staggered(
        n_units=90, cohorts=cohorts, effect=simulate.dynamic(0.3, 0.2), seed=7
    )
    p = panel_from(frame)
    d = bacon.decompose(p)
    assert d.comparisons["weight"].sum() == pytest.approx(1.0, abs=1e-12)
    assert (d.comparisons["weight"] >= 0).all()
    assert d.estimate == pytest.approx(d.twfe, abs=1e-12)
    assert d.twfe == pytest.approx(twfe.twfe(p)["treated"], abs=1e-12)
    summary = d.summary()
    assert summary["weight"].sum() == pytest.approx(1.0)


def test_growing_effects_make_later_vs_earlier_comparisons_wrong(
    dynamic_frame: pd.DataFrame,
) -> None:
    summary = bacon.decompose(panel_from(dynamic_frame)).summary()
    # Already-treated controls keep moving, so this comparison understates the effect.
    clean = summary.loc["treated vs never treated", "estimate"]
    assert summary.loc["later vs earlier treated", "estimate"] < clean - 0.5


def test_unbalanced_panels_are_refused(noisy_frame: pd.DataFrame) -> None:
    p = panel_from(noisy_frame.iloc[1:])
    with pytest.raises(ValueError, match="balanced"):
        bacon.decompose(p)
    with pytest.raises(ValueError):
        bacon.decompose(panel_from(noisy_frame.assign(cohort=np.inf)))
