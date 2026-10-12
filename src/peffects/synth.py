"""Synthetic control (Abadie, Diamond and Hainmueller 2010) with placebo inference.

The synthetic unit is a convex combination of donors chosen to match the treated unit on
pre-treatment predictors. Predictor importance (the diagonal V matrix) is chosen, as in the
original method, to minimize the pre-treatment fit of the outcome itself. Inference is by
placebo: refit treating each donor as if it were treated and ask how unusual the real
post/pre fit ratio is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Predictor:
    """The mean of ``column`` over periods ``start`` to ``end`` inclusive (NaNs skipped)."""

    column: str
    start: float
    end: float | None = None

    @property
    def label(self) -> str:
        span = (
            f"{self.start:g}" if self.end in (None, self.start) else f"{self.start:g}-{self.end:g}"
        )
        return f"{self.column} ({span})"


def simplex_weights(target: FloatArray, donors: FloatArray, v: FloatArray) -> FloatArray:
    """argmin_w (x1 - X0 w)' diag(v) (x1 - X0 w) subject to w >= 0, sum(w) = 1.

    Solved exactly by non-negative least squares with the adding-up constraint appended as a
    heavily weighted row, then renormalized.
    """
    root = np.sqrt(np.maximum(v, 0))
    a = donors * root[:, None]
    b = target * root
    scale = max(float(np.max(np.abs(a))), 1.0) * 1e4
    a_aug = np.vstack([a, np.full((1, donors.shape[1]), scale)])
    b_aug = np.append(b, scale)
    w, _ = optimize.nnls(a_aug, b_aug, maxiter=50 * donors.shape[1])
    w = np.where(w < 1e-12, 0.0, w)
    total = w.sum()
    if total <= 0:
        raise ValueError("no feasible synthetic control")
    return np.asarray(w / total, dtype=np.float64)


@dataclass(frozen=True)
class SynthFit:
    unit: object
    donors: NDArray[np.object_]
    weights: FloatArray
    v: FloatArray
    predictors: pd.DataFrame  # treated, synthetic, donor mean, importance
    times: FloatArray
    treated_path: FloatArray
    synthetic_path: FloatArray
    adoption: float

    @property
    def gap(self) -> FloatArray:
        return self.treated_path - self.synthetic_path

    @property
    def pre(self) -> NDArray[np.bool_]:
        return self.times < self.adoption

    def rmspe(self, post: bool) -> float:
        mask = ~self.pre if post else self.pre
        return float(np.sqrt(np.mean(self.gap[mask] ** 2)))

    @property
    def ratio(self) -> float:
        pre = self.rmspe(False)
        return self.rmspe(True) / pre if pre > 0 else np.inf

    def weight_table(self, minimum: float = 0.001) -> pd.DataFrame:
        frame = pd.DataFrame({"donor": self.donors, "weight": self.weights})
        frame = frame[frame["weight"] >= minimum]
        return frame.sort_values("weight", ascending=False).reset_index(drop=True)


def _predictor_matrix(
    df: pd.DataFrame, unit: str, time: str, units: Sequence[object], predictors: Sequence[Predictor]
) -> FloatArray:
    out = np.empty((len(predictors), len(units)))
    for i, p in enumerate(predictors):
        end = p.start if p.end is None else p.end
        window = df[(df[time] >= p.start) & (df[time] <= end)]
        means = window.groupby(unit)[p.column].mean()
        out[i] = means.reindex(list(units)).to_numpy(np.float64)
    if np.isnan(out).any():
        bad = [predictors[int(i)].label for i in np.flatnonzero(np.isnan(out).any(axis=1))]
        raise ValueError(f"predictors missing for some units: {bad}")
    return out


def _fit_weights(
    x1: FloatArray,
    x0: FloatArray,
    z1: FloatArray,
    z0: FloatArray,
    *,
    optimize_v: bool,
    starts: int,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Weights and V. Predictors are scaled by their standard deviation across units first."""
    sd = np.concatenate([x1[:, None], x0], axis=1).std(axis=1, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    x1s, x0s = x1 / sd, x0 / sd[:, None]
    k = len(x1)

    def weights_for(v: FloatArray) -> FloatArray:
        return simplex_weights(x1s, x0s, v)

    def loss(theta: FloatArray) -> float:
        v = np.exp(theta - np.max(theta))
        w = weights_for(v / v.sum())
        return float(np.mean((z1 - z0 @ w) ** 2))

    if not optimize_v or k == 1:
        v = np.full(k, 1.0 / k)
        return weights_for(v), v
    rng = np.random.default_rng(seed)
    candidates = [np.zeros(k)]
    # Synth's regression-based start: importance from regressing outcomes on predictors.
    design = np.column_stack(
        [np.ones(x0s.shape[1] + 1), np.concatenate([x1s[:, None], x0s], axis=1).T]
    )
    outcome = np.concatenate([z1[:, None], z0], axis=1).T
    coef, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    reg_v = np.sum(coef[1:] ** 2, axis=1)
    if np.all(reg_v > 0):
        candidates.append(np.log(reg_v / reg_v.sum()))
    candidates += [rng.normal(0, 1.5, k) for _ in range(max(starts - len(candidates), 0))]
    best_theta, best_loss = candidates[0], np.inf
    for theta0 in candidates:
        result = optimize.minimize(
            loss,
            theta0,
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-10},
        )
        if result.fun < best_loss:
            best_theta, best_loss = result.x, float(result.fun)
    v = np.exp(best_theta - np.max(best_theta))
    v = v / v.sum()
    return weights_for(v), v


def synth(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    treated: object,
    adoption: float,
    predictors: Sequence[Predictor] | None = None,
    donors: Sequence[object] | None = None,
    optimize_v: bool = True,
    starts: int = 6,
    seed: int = 0,
) -> SynthFit:
    """Fit a synthetic control for ``treated``, which adopts in period ``adoption``.

    Without ``predictors``, every pre-treatment outcome is a predictor with equal importance.
    """
    units = df[unit].unique()
    if treated not in set(units):
        raise ValueError(f"{treated!r} is not a unit")
    pool = [u for u in (donors if donors is not None else units) if u != treated]
    if len(pool) < 2:
        raise ValueError("need at least two donors")
    wide = df.pivot(index=unit, columns=time, values=outcome)
    if wide.loc[[treated, *pool]].isna().any().any():
        raise ValueError("the outcome must be observed for every unit and period")
    times = wide.columns.to_numpy(np.float64)
    pre = times < adoption
    if pre.sum() < 2 or (~pre).sum() < 1:
        raise ValueError("need at least two pre-treatment periods and one after")
    z1 = wide.loc[[treated]].to_numpy(np.float64)[0]
    z0 = wide.loc[pool].to_numpy(np.float64).T  # periods x donors

    if predictors is None:
        x1, x0 = z1[pre], z0[pre]
        labels = [f"{outcome} ({t:g})" for t in times[pre]]
        weights, v = _fit_weights(x1, x0, z1[pre], z0[pre], optimize_v=False, starts=1, seed=seed)
    else:
        matrix = _predictor_matrix(df, unit, time, [treated, *pool], predictors)
        x1, x0 = matrix[:, 0], matrix[:, 1:]
        labels = [p.label for p in predictors]
        weights, v = _fit_weights(
            x1, x0, z1[pre], z0[pre], optimize_v=optimize_v, starts=starts, seed=seed
        )
    table = pd.DataFrame(
        {
            "treated": x1,
            "synthetic": x0 @ weights,
            "donor mean": x0.mean(axis=1),
            "importance": v,
        },
        index=labels,
    )
    return SynthFit(
        unit=treated,
        donors=np.asarray(pool, dtype=object),
        weights=weights,
        v=v,
        predictors=table,
        times=times,
        treated_path=z1,
        synthetic_path=z0 @ weights,
        adoption=adoption,
    )


@dataclass(frozen=True)
class PlaceboTest:
    fit: SynthFit
    placebos: dict[object, SynthFit]

    def ratios(self) -> pd.Series:
        values = {self.fit.unit: self.fit.ratio} | {u: f.ratio for u, f in self.placebos.items()}
        return pd.Series(values).sort_values(ascending=False)

    @property
    def p_value(self) -> float:
        """Share of units (treated included) whose post/pre RMSPE ratio is at least the
        treated unit's: the permutation p-value of Abadie et al. (2010)."""
        ratios = self.ratios()
        return float(np.mean(ratios >= self.fit.ratio))

    def gaps(self, max_pre_ratio: float | None = None) -> pd.DataFrame:
        """Gap paths (period x unit). ``max_pre_ratio`` drops placebos whose pre-period
        MSPE exceeds that multiple of the treated unit's, as in the paper's figures."""
        base = self.fit.rmspe(False) ** 2
        frame = {self.fit.unit: self.fit.gap}
        for u, f in self.placebos.items():
            if max_pre_ratio is None or f.rmspe(False) ** 2 <= max_pre_ratio * base:
                frame[u] = f.gap
        return pd.DataFrame(frame, index=self.fit.times)


def placebo_test(
    df: pd.DataFrame,
    fit: SynthFit,
    *,
    unit: str,
    time: str,
    outcome: str,
    predictors: Sequence[Predictor] | None = None,
    optimize_v: bool = True,
    starts: int = 3,
    seed: int = 0,
) -> PlaceboTest:
    """Refit with each donor as the treated unit and the remaining donors as its pool."""
    placebos = {}
    donors = list(fit.donors)
    for d in donors:
        pool = [u for u in donors if u != d]
        placebos[d] = synth(
            df,
            unit=unit,
            time=time,
            outcome=outcome,
            treated=d,
            adoption=fit.adoption,
            predictors=predictors,
            donors=pool,
            optimize_v=optimize_v,
            starts=starts,
            seed=seed,
        )
    return PlaceboTest(fit, placebos)
