"""Goodman-Bacon (2021): the TWFE coefficient as a weighted average of 2x2 comparisons.

With staggered adoption, the static TWFE estimate averages every pairwise DiD between
timing groups. Some of those comparisons use already-treated units as controls ("later vs
earlier"); when effects grow over time, those comparisons are biased and can even flip sign.
The decomposition shows how much weight they carry. It is exact: the weights sum to one and
the weighted average reproduces the unweighted TWFE coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from peffects.ols import demean
from peffects.panel import Panel

KINDS = (
    "treated vs never treated",
    "earlier vs later treated",
    "later vs earlier treated",
    "later vs always treated",
)


@dataclass(frozen=True)
class Decomposition:
    comparisons: pd.DataFrame  # kind, treated, control, estimate, weight
    twfe: float

    @property
    def estimate(self) -> float:
        frame = self.comparisons
        return float((frame["estimate"] * frame["weight"]).sum())

    def summary(self) -> pd.DataFrame:
        """Total weight and weighted mean estimate per kind of comparison."""
        frame = self.comparisons.assign(product=lambda d: d["estimate"] * d["weight"])
        out = frame.groupby("kind", sort=False).agg(
            weight=("weight", "sum"), product=("product", "sum"), comparisons=("weight", "size")
        )
        out["estimate"] = out["product"] / out["weight"]
        return out.drop(columns="product").reindex([k for k in KINDS if k in out.index])


def _mean(wide: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    return float(wide[np.ix_(rows, cols)].mean())


def _did(
    wide: np.ndarray, treat: np.ndarray, control: np.ndarray, pre: np.ndarray, post: np.ndarray
) -> float:
    return (_mean(wide, treat, post) - _mean(wide, treat, pre)) - (
        _mean(wide, control, post) - _mean(wide, control, pre)
    )


def decompose(panel: Panel) -> Decomposition:
    """Decompose the unweighted static TWFE coefficient. Needs a balanced panel."""
    if not panel.balanced:
        raise ValueError("the decomposition needs a balanced panel")
    wide = panel.wide()
    times = panel.times
    cohort = panel.cohort
    n = panel.n_units

    # Timing groups: never treated (share of periods treated 0), always treated (1), and each
    # cohort adopting inside the window.
    groups: list[tuple[str, float, np.ndarray]] = []
    never = ~np.isfinite(cohort)
    if never.any():
        groups.append(("never", np.inf, np.flatnonzero(never)))
    always = cohort <= times[0]
    if always.any():
        groups.append(("always", -np.inf, np.flatnonzero(always)))
    for g in panel.cohorts:
        groups.append((f"{g:g}", float(g), np.flatnonzero(cohort == g)))

    d = panel.treated.astype(np.float64)
    d_tilde = demean(d, [panel.unit_index, panel.time_index])
    variance = float(np.mean(d_tilde**2))
    if variance <= 0:
        raise ValueError("treatment does not vary after removing unit and period effects")

    def share_treated(g: float) -> float:
        return float(np.mean(times >= g))

    rows = []
    for i, (name_a, g_a, units_a) in enumerate(groups):
        for name_b, g_b, units_b in groups[i + 1 :]:
            # Order the pair so that k adopts earlier than l.
            (name_k, g_k, units_k), (name_l, g_l, units_l) = sorted(
                [(name_a, g_a, units_a), (name_b, g_b, units_b)], key=lambda item: item[1]
            )
            n_k, n_l = len(units_k) / n, len(units_l) / n
            d_k, d_l = share_treated(g_k), share_treated(g_l)
            if d_k == d_l:
                continue
            n_kl = n_k / (n_k + n_l)
            mix = n_kl * (1 - n_kl)
            if name_l == "never":
                weight = (n_k + n_l) ** 2 * mix * d_k * (1 - d_k) / variance
                post = times >= g_k
                estimate = _did(wide, units_k, units_l, np.flatnonzero(~post), np.flatnonzero(post))
                rows.append((KINDS[0], name_k, name_l, estimate, weight))
                continue
            # k treated, l not yet treated: periods before l adopts.
            if d_k < 1:
                weight = (
                    ((n_k + n_l) * (1 - d_l)) ** 2
                    * mix
                    * ((d_k - d_l) / (1 - d_l))
                    * ((1 - d_k) / (1 - d_l))
                    / variance
                )
                window = times < g_l
                pre, post = window & (times < g_k), window & (times >= g_k)
                estimate = _did(wide, units_k, units_l, np.flatnonzero(pre), np.flatnonzero(post))
                rows.append((KINDS[1], name_k, name_l, estimate, weight))
            # l treated, k already treated: periods after k adopts.
            if d_l > 0:
                weight = (
                    ((n_k + n_l) * d_k) ** 2 * mix * (d_l / d_k) * ((d_k - d_l) / d_k) / variance
                )
                window = times >= g_k
                pre, post = window & (times < g_l), window & (times >= g_l)
                estimate = _did(wide, units_l, units_k, np.flatnonzero(pre), np.flatnonzero(post))
                kind = KINDS[3] if name_k == "always" else KINDS[2]
                rows.append((kind, name_l, name_k, estimate, weight))

    frame = pd.DataFrame(rows, columns=["kind", "treated", "control", "estimate", "weight"])
    y_tilde = demean(panel.y, [panel.unit_index, panel.time_index])
    beta = float(np.sum(d_tilde * y_tilde) / np.sum(d_tilde**2))
    return Decomposition(frame, beta)
