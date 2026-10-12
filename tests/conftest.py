from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from peffects import data, simulate
from peffects.panel import Panel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def have_data() -> bool:
    return all((DATA_DIR / s.filename).exists() for s in data.SOURCES.values())


needs_data = pytest.mark.skipif(not have_data(), reason="run `peffects fetch` first")


def panel_from(frame: pd.DataFrame, **kwargs: object) -> Panel:
    return Panel.from_frame(frame, unit="unit", time="time", outcome="y", cohort="cohort", **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def dynamic_frame() -> pd.DataFrame:
    """Effects that grow with exposure and differ by cohort, with no noise."""
    return simulate.staggered(
        n_units=300,
        cohorts={2003.0: 0.3, 2006.0: 0.3},
        effect=simulate.dynamic(0.5, cohort_shift=0.1),
        noise=0.0,
        seed=1,
    )


@pytest.fixture(scope="session")
def noisy_frame() -> pd.DataFrame:
    return simulate.staggered(
        n_units=200, cohorts={2003.0: 0.25, 2005.0: 0.25, 2007.0: 0.2}, seed=2
    )
