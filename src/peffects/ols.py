"""Weighted least squares with absorbed fixed effects and cluster-robust inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def demean(
    values: FloatArray,
    groups: Sequence[IntArray],
    weights: FloatArray | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> FloatArray:
    """Sweep out every set of group means (alternating projections).

    One sweep is exact for a single factor or a balanced unweighted two-way panel; other
    designs iterate until the largest change falls below ``tol``.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    flat = out.ndim == 1
    if flat:
        out = out[:, None]
    if not groups:
        return out[:, 0] if flat else out
    w = np.ones(out.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    sizes = [np.bincount(g, weights=w) for g in groups]
    scale = max(float(np.max(np.abs(out))) if out.size else 0.0, 1.0)
    for _ in range(max_iter):
        change = 0.0
        for g, size in zip(groups, sizes, strict=True):
            for j in range(out.shape[1]):
                means = np.bincount(g, weights=w * out[:, j], minlength=len(size))
                with np.errstate(invalid="ignore", divide="ignore"):
                    means = np.where(size > 0, means / size, 0.0)
                step = means[g]
                out[:, j] -= step
                change = max(change, float(np.max(np.abs(step))) if step.size else 0.0)
        if len(groups) == 1 or change < tol * scale:
            break
    return out[:, 0] if flat else out


@dataclass(frozen=True)
class Fit:
    names: tuple[str, ...]
    coef: FloatArray
    vcov: FloatArray
    resid: FloatArray
    n_obs: int
    n_clusters: int
    df: int  # degrees of freedom for t tests: clusters - 1, or n - k without clusters

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no coefficient named {name!r}") from None

    def __getitem__(self, name: str) -> float:
        return float(self.coef[self.index(name)])

    def se(self, name: str) -> float:
        i = self.index(name)
        return float(np.sqrt(self.vcov[i, i]))

    def table(self, alpha: float = 0.05) -> pd.DataFrame:
        se = np.sqrt(np.diag(self.vcov))
        with np.errstate(invalid="ignore", divide="ignore"):
            t = self.coef / se
        crit = stats.t.ppf(1 - alpha / 2, self.df)
        return pd.DataFrame(
            {
                "coef": self.coef,
                "se": se,
                "t": t,
                "p": 2 * stats.t.sf(np.abs(t), self.df),
                "low": self.coef - crit * se,
                "high": self.coef + crit * se,
            },
            index=list(self.names),
        )


def cluster_vcov(
    x: FloatArray,
    resid: FloatArray,
    weights: FloatArray,
    clusters: IntArray,
    bread: FloatArray,
    k: int,
) -> tuple[FloatArray, int]:
    """CR1 sandwich: G/(G-1) * (N-1)/(N-K) small-sample factor, as Stata and fixest use."""
    _, codes = np.unique(clusters, return_inverse=True)
    g = int(codes.max()) + 1
    scores = x * (weights * resid)[:, None]
    per_cluster = np.zeros((g, x.shape[1]))
    np.add.at(per_cluster, codes, scores)
    meat = per_cluster.T @ per_cluster
    n = x.shape[0]
    factor = g / (g - 1) * (n - 1) / (n - k) if g > 1 and n > k else np.nan
    return factor * bread @ meat @ bread, g


def regress(
    y: FloatArray,
    x: FloatArray,
    names: Sequence[str],
    *,
    absorb: Sequence[IntArray] = (),
    weights: FloatArray | None = None,
    cluster: IntArray | None = None,
    nested: Sequence[bool] | None = None,
) -> Fit:
    """OLS of ``y`` on ``x`` after absorbing the fixed effects in ``absorb``.

    ``nested`` marks which absorbed factors are nested within the clusters; those levels are
    not counted in K for the small-sample correction (the reghdfe and fixest convention).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] != len(names):
        raise ValueError("one name per regressor")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    groups = [np.asarray(g, dtype=np.int64) for g in absorb]
    yt = demean(np.asarray(y, dtype=np.float64), groups, w)
    xt = demean(x, groups, w)
    xtwx = xt.T @ (xt * w[:, None])
    rank = np.linalg.matrix_rank(xtwx)
    if rank < x.shape[1]:
        raise np.linalg.LinAlgError(
            f"regressors are collinear with the fixed effects ({rank} of {x.shape[1]} identified)"
        )
    bread = np.asarray(np.linalg.inv(xtwx), dtype=np.float64)
    coef = bread @ (xt.T @ (w * yt))
    resid = yt - xt @ coef

    nested_flags = list(nested) if nested is not None else [False] * len(groups)
    absorbed = sum(len(np.unique(g)) for g, n in zip(groups, nested_flags, strict=True) if not n)
    if not any(nested_flags):
        # Each factor beyond the first repeats the intercept that the others already carry.
        absorbed -= max(len(groups) - 1, 0)
    k = x.shape[1] + absorbed
    if cluster is None:
        n = len(y)
        levels = sum(len(np.unique(g)) for g in groups) - max(len(groups) - 1, 0)
        dof = n - x.shape[1] - levels
        if dof <= 0:
            raise ValueError("no residual degrees of freedom")
        sigma2 = float(np.sum(w * resid**2)) / dof
        return Fit(tuple(names), coef, sigma2 * bread, resid, n, n, dof)
    vcov, g = cluster_vcov(xt, resid, w, np.asarray(cluster), bread, k)
    return Fit(tuple(names), coef, vcov, resid, len(y), g, g - 1)
