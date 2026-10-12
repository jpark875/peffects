"""Inference that stays honest with few treated clusters."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from peffects.ols import demean
from peffects.panel import Panel

FloatArray = NDArray[np.float64]
Weights = Literal["rademacher", "webb", "mammen"]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


@dataclass(frozen=True)
class BootstrapTest:
    estimate: float
    t: float
    p_value: float
    low: float
    high: float
    reps: int
    clusters: int
    enumerated: bool
    weights: str


def _draws(g: int, reps: int, kind: Weights, rng: np.random.Generator) -> tuple[FloatArray, bool]:
    if kind == "rademacher" and 2**g <= max(reps, 1) * 4 and g <= 12:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=g)), dtype=np.float64)
        return signs, True
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=(reps, g)).astype(np.float64), False
    if kind == "webb":
        return rng.choice(WEBB, size=(reps, g)), False
    if kind == "mammen":
        root5 = np.sqrt(5.0)
        high = rng.uniform(size=(reps, g)) < (root5 + 1) / (2 * root5)
        return np.where(high, (1 - root5) / 2, (1 + root5) / 2), False
    raise ValueError("weights must be rademacher, webb or mammen")


class _Design:
    """Within-transformed regression pieces shared by every null value tested."""

    def __init__(
        self,
        y: FloatArray,
        x: FloatArray,
        j: int,
        clusters: NDArray[np.int64],
        w: FloatArray,
        k_total: int,
        absorb: Sequence[NDArray[np.int64]],
    ) -> None:
        self.y, self.x, self.j, self.w, self.absorb = y, x, j, w, list(absorb)
        _, self.codes = np.unique(clusters, return_inverse=True)
        self.g = int(np.max(self.codes)) + 1
        n = len(y)
        self.factor = self.g / (self.g - 1) * (n - 1) / (n - k_total)
        xw = x * w[:, None]
        self.bread = np.linalg.inv(x.T @ xw)
        self.beta = self.bread @ (xw.T @ y)
        k = x.shape[1]
        # Per-cluster X'WX blocks projected on row j of the bread.
        self.h = np.zeros((self.g, k))
        for c in range(self.g):
            rows = self.codes == c
            self.h[c] = self.bread[j] @ (x[rows].T @ xw[rows])
        self.xw = xw

    def _cluster_sum(self, values: FloatArray) -> FloatArray:
        out = np.zeros((self.g,) + values.shape[1:])
        np.add.at(out, self.codes, values)
        return out

    def t_stat(self, beta: FloatArray, resid: FloatArray, null: float) -> float:
        scores = self._cluster_sum(self.xw * resid[:, None]) @ self.bread[self.j]
        se = np.sqrt(self.factor * np.sum(scores**2))
        return float((beta[self.j] - null) / se)

    def bootstrap_t(self, null: float, v: FloatArray) -> tuple[float, FloatArray]:
        """Observed t against ``null`` and the bootstrap t distribution under that null."""
        resid = self.y - self.x @ self.beta
        t_obs = self.t_stat(self.beta, resid, null)
        others = [c for c in range(self.x.shape[1]) if c != self.j]
        target = self.y - null * self.x[:, self.j]
        beta_r = np.zeros(self.x.shape[1])
        beta_r[self.j] = null
        if others:
            xo = self.x[:, others]
            xow = xo * self.w[:, None]
            beta_r[others] = np.linalg.solve(xo.T @ xow, xow.T @ target)
        u_r = self.y - self.x @ beta_r
        t_g = self._cluster_sum(self.xw * u_r[:, None])  # G x k
        s = t_g @ self.bread.T  # G x k: each cluster's push on the coefficients
        delta = v @ s  # B x k
        # Bootstrap residuals are re-projected off the fixed effects: sign-flipping whole
        # clusters does not preserve within-period means. a[g, h] is cluster h's flipped
        # residuals, after that projection, as they load on cluster g's score.
        flipped = np.zeros((len(u_r), self.g))
        flipped[np.arange(len(u_r)), self.codes] = u_r
        projected = demean(flipped, self.absorb, self.w)
        a = self._cluster_sum((self.xw @ self.bread[self.j])[:, None] * projected)  # G x G
        q = v @ a.T - delta @ self.h.T  # B x G
        se = np.sqrt(self.factor * np.sum(q**2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_boot = delta[:, self.j] / se
        return t_obs, t_boot


def wild_cluster_bootstrap(
    panel: Panel,
    *,
    regressors: Sequence[str] = ("treated",),
    test: str = "treated",
    cluster: NDArray[np.int64] | None = None,
    weighted: bool = True,
    reps: int = 9999,
    weights: Weights = "rademacher",
    alpha: float = 0.05,
    seed: int = 0,
    interval: bool = True,
) -> BootstrapTest:
    """WCR bootstrap-t for one coefficient of the TWFE regression of ``panel``."""
    columns, names = [], []
    for name in regressors:
        if name == "treated":
            columns.append(panel.treated.astype(np.float64))
        else:
            columns.append(panel.x[:, panel.x_names.index(name)])
        names.append(name)
    if test not in names:
        raise ValueError(f"{test!r} is not among the regressors")
    x = np.column_stack(columns)
    w = panel.weight[panel.unit_index] if weighted else np.ones(panel.n_obs)
    groups = [panel.unit_index, panel.time_index]
    xt = demean(x, groups, w)
    yt = demean(panel.y, groups, w)
    codes = panel.unit_index if cluster is None else np.asarray(cluster)
    # K as in ``ols.regress``: unit effects nested within clusters are not counted.
    nested = all(len(np.unique(codes[panel.unit_index == u])) == 1 for u in range(panel.n_units))
    k_total = x.shape[1] + (panel.n_times if nested else panel.n_units + panel.n_times - 1)
    design = _Design(yt, xt, names.index(test), codes, w, k_total, groups)
    rng = np.random.default_rng(seed)
    v, enumerated = _draws(design.g, reps, weights, rng)

    def p_value(null: float) -> float:
        t_obs, t_boot = design.bootstrap_t(null, v)
        return float(np.mean(np.abs(t_boot) >= abs(t_obs) - 1e-12))

    t_obs, _ = design.bootstrap_t(0.0, v)
    p0 = p_value(0.0)
    estimate = float(design.beta[design.j])
    low = high = np.nan
    if interval:
        resid = yt - xt @ design.beta
        scores = design._cluster_sum(xt * (w * resid)[:, None]) @ design.bread[design.j]
        se = float(np.sqrt(design.factor * np.sum(scores**2)))
        low = _invert(p_value, estimate, -se, alpha)
        high = _invert(p_value, estimate, se, alpha)
    return BootstrapTest(estimate, t_obs, p0, low, high, len(v), design.g, enumerated, weights)


def _invert(p_value: Callable[[float], float], center: float, step: float, alpha: float) -> float:
    """Walk out from the estimate until the test rejects, then bisect to the boundary."""
    inside, outside = center, center + step
    for _ in range(60):
        if p_value(outside) < alpha:
            break
        inside, outside = outside, outside + step
        step *= 1.5
    else:
        return np.inf if step > 0 else -np.inf
    return float(
        optimize.brentq(lambda r: p_value(r) - alpha - 1e-9, inside, outside, xtol=1e-6 * abs(step))
    )


@dataclass(frozen=True)
class PermutationTest:
    estimate: float
    draws: FloatArray
    p_value: float


def randomization_inference(
    panel: Panel,
    statistic: Callable[[Panel], float],
    *,
    reps: int = 999,
    seed: int = 0,
) -> PermutationTest:
    """Shuffle adoption dates across units, keeping how many units adopt when."""
    rng = np.random.default_rng(seed)
    observed = float(statistic(panel))
    draws = np.empty(reps)
    for b in range(reps):
        draws[b] = statistic(panel.with_cohorts(rng.permutation(panel.cohort)))
    p = (1 + np.sum(np.abs(draws) >= abs(observed) - 1e-12)) / (reps + 1)
    return PermutationTest(observed, draws, float(p))
