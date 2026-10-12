"""Sun and Abraham (2021): the interaction-weighted event study.

A dynamic TWFE regression with one dummy per period since adoption mixes cohorts, and the
coefficient for one period can be contaminated by effects from others. Sun and Abraham
saturate the regression in cohort x relative period, so each coefficient CATT(g, e) uses a
clean control (never-treated units, or the last cohort before it adopts), then average
across cohorts with weights equal to each cohort's share of the units observed at e.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from peffects.cs import Estimate
from peffects.ols import Fit, regress
from peffects.panel import Panel
from peffects.twfe import EventStudy


@dataclass(frozen=True)
class SunAbraham:
    event_study: EventStudy
    cohort_effects: pd.DataFrame  # group, e, catt, se
    overall: Estimate  # treated-cell weighted average of post-period CATT
    control: str
    fit: Fit


def sun_abraham(
    panel: Panel,
    *,
    reference: int = -1,
    cluster: str = "unit",
    weighted: bool = True,
    alpha: float = 0.05,
) -> SunAbraham:
    """Estimate CATT(g, e) by saturated TWFE and aggregate them into an event study."""
    panel = panel.keep_units(~panel.always_treated)
    if panel.never_treated.any():
        control = "never treated"
    else:
        last = float(panel.cohorts.max())
        panel = panel.keep_times(panel.times < last)
        panel = panel.with_cohorts(np.where(panel.cohort == last, np.inf, panel.cohort))
        control = f"last cohort ({last:g}) before it adopts"
    rel = panel.time - panel.unit_cohort
    cohorts = [g for g in panel.cohorts if g > panel.times[0]]
    if not cohorts:
        raise ValueError("no cohorts adopt inside the window")

    columns, names, keys = [], [], []
    unit_cohort = panel.unit_cohort
    for g in cohorts:
        mine = unit_cohort == g
        for e in np.unique(rel[mine]):
            if e == reference:
                continue
            columns.append((mine & (rel == e)).astype(np.float64))
            names.append(f"g{g:g}_e{e:g}")
            keys.append((float(g), float(e)))
    if not columns:
        raise ValueError("nothing to estimate")
    codes = panel.unit_index if cluster == "unit" else panel.time_index
    nested = [True, False] if cluster == "unit" else [False, True]
    w_obs = panel.weight[panel.unit_index] if weighted else np.ones(panel.n_obs)
    fit = regress(
        panel.y,
        np.column_stack(columns),
        names,
        absorb=[panel.unit_index, panel.time_index],
        weights=w_obs,
        cluster=codes,
        nested=nested,
    )

    cells = pd.DataFrame(keys, columns=["group", "e"])
    cells["catt"] = fit.coef
    cells["se"] = np.sqrt(np.diag(fit.vcov))
    unit_w = panel.weight if weighted else np.ones(panel.n_units)
    size = {g: float(unit_w[panel.cohort == g].sum()) for g in cohorts}
    cells["share"] = cells["group"].map(size)
    cells["treated_cells"] = [float(w_obs[(unit_cohort == g) & (rel == e)].sum()) for g, e in keys]

    crit = float(stats.t.ppf(1 - alpha / 2, fit.df))
    rows = []
    for e in np.sort(cells["e"].unique()):
        block = cells[cells["e"] == e]
        weights = np.zeros(len(cells))
        weights[block.index] = block["share"] / block["share"].sum()
        est = float(weights @ fit.coef)
        se = float(np.sqrt(weights @ fit.vcov @ weights))
        rows.append(
            {
                "e": int(e),
                "coef": est,
                "se": se,
                "low": est - crit * se,
                "high": est + crit * se,
                "cells": int(sum((rel == e) & np.isfinite(rel))),
            }
        )
    rows.append(
        {
            "e": reference,
            "coef": 0.0,
            "se": np.nan,
            "low": np.nan,
            "high": np.nan,
            "cells": int((rel == reference).sum()),
        }
    )
    profile = pd.DataFrame(rows).sort_values("e").reset_index(drop=True)

    post = (cells["e"] >= 0).to_numpy()
    weights = np.where(post, cells["treated_cells"], 0.0)
    weights = weights / weights.sum()
    est = float(weights @ fit.coef)
    se = float(np.sqrt(weights @ fit.vcov @ weights))
    overall = Estimate(est, se, est - crit * se, est + crit * se)
    return SunAbraham(EventStudy(profile, reference, "sun-abraham"), cells, overall, control, fit)
