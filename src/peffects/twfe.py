"""Two-way fixed effects: the static DiD regression and the dynamic event-study version."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from peffects.ols import Fit, regress
from peffects.panel import Panel


def _cluster(panel: Panel, cluster: str) -> tuple[np.ndarray, list[bool]]:
    if cluster == "unit":
        return panel.unit_index, [True, False]
    if cluster == "time":
        return panel.time_index, [False, True]
    raise ValueError("cluster must be 'unit' or 'time'")


def twfe(
    panel: Panel, *, cluster: str = "unit", weighted: bool = True, covariates: bool = False
) -> Fit:
    """Regress the outcome on the treatment dummy plus unit and period fixed effects, and the
    panel's time-varying covariates if ``covariates``.
    """
    d = panel.treated.astype(np.float64)
    if np.min(d) == np.max(d):
        raise ValueError("treatment does not vary in this panel")
    codes, nested = _cluster(panel, cluster)
    x = panel.x if covariates else panel.x[:, :0]
    return regress(
        panel.y,
        np.column_stack([d, x]),
        ["treated", *(panel.x_names if covariates else ())],
        absorb=[panel.unit_index, panel.time_index],
        weights=panel.weight[panel.unit_index] if weighted else None,
        cluster=codes,
        nested=nested,
    )


@dataclass(frozen=True)
class EventStudy:
    estimates: pd.DataFrame  # columns e, coef, se, low, high, cells; e = periods since adoption
    reference: int
    method: str

    def post(self) -> pd.DataFrame:
        return self.estimates[self.estimates["e"] >= 0]

    def pre(self) -> pd.DataFrame:
        return self.estimates[(self.estimates["e"] < 0) & (self.estimates["e"] != self.reference)]


def event_study(
    panel: Panel,
    *,
    leads: int | None = None,
    lags: int | None = None,
    reference: int = -1,
    cluster: str = "unit",
    weighted: bool = True,
    covariates: bool = False,
) -> EventStudy:
    """Dynamic TWFE with one dummy per period relative to adoption."""
    rel = panel.time - panel.unit_cohort
    ever = np.isfinite(rel)
    if not ever.any():
        raise ValueError("no treated units")
    lo = int(np.min(rel[ever])) if leads is None else -int(leads)
    hi = int(np.max(rel[ever])) if lags is None else int(lags)
    if not lo <= reference < hi + 1:
        raise ValueError("the reference period must lie inside the window")
    binned = np.where(ever, np.clip(rel, lo, hi), np.nan)
    periods = [e for e in range(lo, hi + 1) if e != reference]
    if not panel.never_treated.any() and periods and periods[0] == lo:
        periods = periods[1:]
    columns, names, kept = [], [], []
    for e in periods:
        col = (binned == e).astype(np.float64)
        if col.any():
            columns.append(col)
            names.append(f"e{e}")
            kept.append(e)
    codes, nested = _cluster(panel, cluster)
    x = panel.x if covariates else panel.x[:, :0]
    fit = regress(
        panel.y,
        np.column_stack([*columns, x]),
        [*names, *(panel.x_names if covariates else ())],
        absorb=[panel.unit_index, panel.time_index],
        weights=panel.weight[panel.unit_index] if weighted else None,
        cluster=codes,
        nested=nested,
    )
    table = fit.table().loc[names]
    cells = [int((binned == e).sum()) for e in kept]
    frame = pd.DataFrame(
        {
            "e": kept,
            "coef": table["coef"].to_numpy(),
            "se": table["se"].to_numpy(),
            "low": table["low"].to_numpy(),
            "high": table["high"].to_numpy(),
            "cells": cells,
        }
    )
    ref = pd.DataFrame(
        {"e": [reference], "coef": [0.0], "se": [np.nan], "low": [np.nan], "high": [np.nan]}
    )
    ref["cells"] = int((binned == reference).sum())
    out = pd.concat([frame, ref]).sort_values("e").reset_index(drop=True)
    return EventStudy(out, reference, "twfe")
