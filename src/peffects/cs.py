"""Callaway and Sant'Anna (2021): group-time average treatment effects and their summaries.

Each ATT(g, t) compares cohort g's outcome change from a base period to t against the same
change among units not (yet) treated, so no already-treated unit is ever used as a control.
Inference follows the paper: every estimate carries an influence function, standard errors
come from it, and a multiplier bootstrap of the same functions gives uniform bands.

With covariates, the comparison is outcome regression (Heckman, Ichimura and Todd, 1997):
the control group's change is predicted from base-period covariates. Without them it is the
plain difference in mean changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from peffects.panel import Panel

FloatArray = NDArray[np.float64]
Control = Literal["never", "notyet"]
Base = Literal["varying", "universal"]
Kind = Literal["simple", "dynamic", "group", "calendar"]


@dataclass(frozen=True)
class Estimate:
    estimate: float
    se: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "se": self.se, "low": self.low, "high": self.high}


@dataclass(frozen=True)
class GroupTime:
    """ATT(g, t) for every cohort and period, with the pieces aggregation needs."""

    table: pd.DataFrame  # group, time, e, att, se, low, high (uniform band), post
    influence: FloatArray  # units x cells
    unit_weight: FloatArray  # normalized to mean one
    unit_cohort: FloatArray
    clusters: NDArray[np.int64]
    control: Control
    base: Base
    alpha: float
    dropped_units: int

    @property
    def n(self) -> int:
        return int(self.influence.shape[0])

    def group_share(self) -> FloatArray:
        """P(G = g) for each cell's cohort, weighted, over all units."""
        groups = self.table["group"].to_numpy()
        return np.array([np.mean(self.unit_weight * (self.unit_cohort == g)) for g in groups])


@dataclass(frozen=True)
class Aggregate:
    kind: Kind
    overall: Estimate
    table: pd.DataFrame  # one row per event time, group or period: key, att, se, low, high


def _se(influence: FloatArray, clusters: NDArray[np.int64]) -> FloatArray:
    """Standard errors from influence functions, summing within clusters first."""
    if influence.ndim == 1:
        influence = influence[:, None]
    n = influence.shape[0]
    summed = np.zeros((int(clusters.max()) + 1, influence.shape[1]))
    np.add.at(summed, clusters, influence)
    se = np.sqrt(np.sum(summed**2, axis=0)) / n
    return np.where(se > np.sqrt(np.finfo(float).eps) * 10, se, np.nan)


def _uniform_crit(
    influence: FloatArray,
    clusters: NDArray[np.int64],
    se: FloatArray,
    alpha: float,
    reps: int,
    seed: int,
) -> float:
    """Critical value for simultaneous bands: the (1 - alpha) quantile of the largest
    standardized bootstrap deviation across estimates (Mammen multiplier weights)."""
    pointwise = float(stats.norm.ppf(1 - alpha / 2))
    ok = np.isfinite(se)
    if reps <= 0 or ok.sum() <= 1:
        return pointwise
    n = influence.shape[0]
    g = int(clusters.max()) + 1
    summed = np.zeros((g, influence.shape[1]))
    np.add.at(summed, clusters, influence)
    rng = np.random.default_rng(seed)
    root5 = np.sqrt(5.0)
    high = rng.uniform(size=(reps, g)) < (root5 + 1) / (2 * root5)
    v = np.where(high, (1 - root5) / 2, (1 + root5) / 2)
    draws = v @ summed[:, ok] / n
    worst = np.max(np.abs(draws) / se[ok], axis=1)
    return max(float(np.quantile(worst, 1 - alpha)), pointwise)


def _period_index(times: FloatArray, value: float) -> int:
    """Index of the first period at or after ``value``."""
    return int(np.searchsorted(times, value, side="left").item())


def _cell(
    dy: FloatArray,
    x: FloatArray,
    treat: NDArray[np.bool_],
    control: NDArray[np.bool_],
    w: FloatArray,
) -> tuple[float, FloatArray]:
    """Outcome-regression DiD on the units in ``treat | control``; returns the ATT and an
    influence function over all units (zero outside the comparison)."""
    n = len(dy)
    sub = treat | control
    n1 = int(sub.sum())
    d = treat[sub].astype(np.float64)
    ws = w[sub]
    ys, xs = dy[sub], x[sub]
    w_treat = ws * d
    w_cont = ws * (1 - d)
    xpx = (xs * w_cont[:, None]).T @ xs / n1
    try:
        xpx_inv = np.linalg.inv(xpx)
    except np.linalg.LinAlgError:
        xpx_inv = np.linalg.pinv(xpx)
    beta = xpx_inv @ ((xs * w_cont[:, None]).T @ ys / n1)
    fitted = xs @ beta
    mean_treat = float(np.mean(w_treat))
    eta_treat = float(np.mean(w_treat * ys)) / mean_treat
    eta_cont = float(np.mean(w_treat * fitted)) / mean_treat
    att = eta_treat - eta_cont

    ols_rep = (w_cont * (ys - fitted))[:, None] * xs @ xpx_inv
    inf_treat = (w_treat * ys - w_treat * eta_treat) / mean_treat
    m1 = np.mean(w_treat[:, None] * xs, axis=0)
    inf_cont = (w_treat * fitted - w_treat * eta_cont + ols_rep @ m1) / mean_treat
    influence = np.zeros(n)
    influence[sub] = (n / n1) * (inf_treat - inf_cont)
    return att, influence


def att_gt(
    panel: Panel,
    *,
    control: Control = "never",
    base: Base = "varying",
    covariates: bool = False,
    clusters: NDArray[np.int64] | None = None,
    alpha: float = 0.05,
    reps: int = 1000,
    seed: int = 0,
) -> GroupTime:
    """Estimate ATT(g, t) for every cohort g and period t (except the base period).

    ``base="varying"`` uses period g-1 for post-treatment cells and t-1 for pre-treatment
    cells (short differences, as the reference implementation does by default);
    ``"universal"`` uses g-1 throughout, so pre-period cells read like an event study.
    Units observed in some periods only are dropped. Cohorts treated in the first period have
    no pre-period and are dropped too.
    """
    if control not in ("never", "notyet"):
        raise ValueError("control must be 'never' or 'notyet'")
    if base not in ("varying", "universal"):
        raise ValueError("base must be 'varying' or 'universal'")
    wide = panel.wide()
    complete = ~np.isnan(wide).any(axis=1)
    always = panel.always_treated
    keep = complete & ~always
    dropped = int((~keep).sum())
    if dropped:
        panel = panel.keep_units(keep)
        wide = wide[keep]
        if clusters is not None:
            clusters = np.asarray(clusters)[keep]
    times = panel.times
    n_times = len(times)
    cohort = panel.cohort
    w = panel.weight / panel.weight.mean()
    cluster_codes = (
        np.unique(np.asarray(clusters), return_inverse=True)[1].astype(np.int64)
        if clusters is not None
        else np.arange(panel.n_units, dtype=np.int64)
    )
    if covariates and panel.x.shape[1] == 0:
        raise ValueError("the panel has no covariates")
    if covariates:
        xwide = np.full((panel.n_units, n_times, panel.x.shape[1]), np.nan)
        xwide[panel.unit_index, panel.time_index] = panel.x
    never = ~np.isfinite(cohort)
    if control == "never" and not never.any():
        raise ValueError("no never-treated units; use control='notyet'")

    rows, influences = [], []
    for g in panel.cohorts:
        gi = _period_index(times, g)
        if gi == 0 or gi >= n_times:
            continue
        for ti in range(n_times):
            post = ti >= gi
            bi = gi - 1 if base == "universal" or post else ti - 1
            if bi < 0 or bi == ti:
                continue
            treat = cohort == g
            if control == "never":
                ctrl = never.copy()
            else:
                horizon = times[max(ti, bi)]
                ctrl = never | ((cohort > horizon) & (cohort != g))
            if not treat.any() or not ctrl.any():
                continue
            dy = wide[:, ti] - wide[:, bi]
            x = np.ones((panel.n_units, 1))
            if covariates:
                x = np.column_stack([x, xwide[:, bi, :]])
            att, inf = _cell(dy, x, treat, ctrl, w)
            rows.append(
                {"group": g, "time": times[ti], "e": times[ti] - g, "att": att, "post": post}
            )
            influences.append(inf)
    if not rows:
        raise ValueError("no estimable group-time cells")
    influence = np.column_stack(influences)
    table = pd.DataFrame(rows)
    se = _se(influence, cluster_codes)
    crit = _uniform_crit(influence, cluster_codes, se, alpha, reps, seed)
    table["se"] = se
    table["low"] = table["att"] - crit * se
    table["high"] = table["att"] + crit * se
    return GroupTime(
        table=table[["group", "time", "e", "att", "se", "low", "high", "post"]],
        influence=influence,
        unit_weight=w,
        unit_cohort=cohort,
        clusters=cluster_codes,
        control=control,
        base=base,
        alpha=alpha,
        dropped_units=dropped,
    )


def _weighted(
    gt: GroupTime, cells: NDArray[np.bool_], share: FloatArray
) -> tuple[float, FloatArray]:
    """Share-weighted average of the chosen cells, with the influence function including the
    estimation error in the shares themselves."""
    idx = np.flatnonzero(cells)
    att = gt.table["att"].to_numpy()[idx]
    pg = share[idx]
    total = pg.sum()
    weights = pg / total
    estimate = float(weights @ att)
    groups = gt.table["group"].to_numpy()[idx]
    indicator = np.column_stack(
        [gt.unit_weight * (gt.unit_cohort == g) - p for g, p in zip(groups, pg, strict=True)]
    )
    wif = indicator / total - np.outer(indicator.sum(axis=1), pg / total**2)
    influence = gt.influence[:, idx] @ weights + wif @ att
    return estimate, influence


def aggregate(
    gt: GroupTime, kind: Kind = "simple", *, reps: int = 1000, seed: int = 0
) -> Aggregate:
    """Summarize ATT(g, t) as one number and, except for ``simple``, a profile.

    - ``simple``: every post-treatment cell, weighted by cohort size.
    - ``dynamic``: by periods since adoption (an event study), overall = mean of e >= 0.
    - ``group``: by cohort, averaging its post-treatment periods; overall weights by size.
    - ``calendar``: by period, across cohorts already treated; overall = mean over periods.
    """
    table = gt.table
    share = gt.group_share()
    post = table["post"].to_numpy(bool)
    z = float(stats.norm.ppf(1 - gt.alpha / 2))

    def estimate(value: float, influence: FloatArray) -> Estimate:
        se = float(_se(influence, gt.clusters)[0])
        return Estimate(value, se, value - z * se, value + z * se)

    if kind == "simple":
        value, inf = _weighted(gt, post, share)
        return Aggregate(kind, estimate(value, inf), pd.DataFrame())

    keys: list[float] = []
    values: list[float] = []
    infs: list[FloatArray] = []
    if kind == "dynamic":
        for e in np.sort(table["e"].unique()):
            value, inf = _weighted(gt, table["e"].to_numpy() == e, share)
            keys.append(float(e))
            values.append(value)
            infs.append(inf)
        mask = np.array(keys) >= 0
        overall_value = float(np.mean(np.array(values)[mask]))
        overall_inf = np.mean(np.column_stack(infs)[:, mask], axis=1)
    elif kind == "group":
        groups = np.sort(table["group"].unique())
        for g in groups:
            cells = post & (table["group"].to_numpy() == g)
            keys.append(float(g))
            values.append(float(table["att"].to_numpy()[cells].mean()))
            infs.append(gt.influence[:, cells].mean(axis=1))
        pg = np.array([np.mean(gt.unit_weight * (gt.unit_cohort == g)) for g in groups])
        weights = pg / pg.sum()
        overall_value = float(weights @ np.array(values))
        indicator = np.column_stack(
            [gt.unit_weight * (gt.unit_cohort == g) - p for g, p in zip(groups, pg, strict=True)]
        )
        wif = indicator / pg.sum() - np.outer(indicator.sum(axis=1), pg / pg.sum() ** 2)
        overall_inf = np.column_stack(infs) @ weights + wif @ np.array(values)
    elif kind == "calendar":
        first = table["group"].min()
        for t in np.sort(table["time"].unique()):
            if t < first:
                continue
            cells = post & (table["time"].to_numpy() == t)
            if not cells.any():
                continue
            value, inf = _weighted(gt, cells, share)
            keys.append(float(t))
            values.append(value)
            infs.append(inf)
        overall_value = float(np.mean(values))
        overall_inf = np.mean(np.column_stack(infs), axis=1)
    else:
        raise ValueError("kind must be simple, dynamic, group or calendar")

    matrix = np.column_stack(infs)
    se = _se(matrix, gt.clusters)
    crit = _uniform_crit(matrix, gt.clusters, se, gt.alpha, reps, seed)
    profile = pd.DataFrame({"key": keys, "att": values, "se": se})
    profile["low"] = profile["att"] - crit * profile["se"]
    profile["high"] = profile["att"] + crit * profile["se"]
    return Aggregate(kind, estimate(overall_value, overall_inf), profile)
