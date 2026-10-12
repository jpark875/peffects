"""Simulated staggered-adoption panels with known effects, for tests and demonstrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

Effect = Callable[[float, float], float]  # (cohort, periods since adoption) -> effect


def constant(value: float) -> Effect:
    return lambda g, e: value


def dynamic(slope: float, cohort_shift: float = 0.0) -> Effect:
    """Effects that grow with time since adoption and differ by cohort (TWFE's worst case)."""
    return lambda g, e: slope * (e + 1) + cohort_shift * (g - 2000)


def staggered(
    *,
    n_units: int = 200,
    periods: tuple[int, int] = (2000, 2009),
    cohorts: Mapping[float, float] | None = None,
    effect: Effect | None = None,
    noise: float = 1.0,
    unit_sd: float = 1.0,
    trend_sd: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Long panel with columns unit, time, y, y0, cohort (inf = never) and effect.

    ``cohorts`` maps adoption period to the share of units adopting then; the remainder are
    never treated. Untreated outcomes follow parallel trends: unit effect + period effect +
    noise, plus an optional unit-specific linear trend that breaks parallel trends.
    """
    effect = effect or constant(1.0)
    rng = np.random.default_rng(seed)
    first, last = periods
    times = np.arange(first, last + 1, dtype=np.float64)
    shares = dict(cohorts if cohorts is not None else {2003.0: 0.25, 2006.0: 0.25})
    if sum(shares.values()) > 1:
        raise ValueError("cohort shares must sum to at most 1")
    counts = {g: int(round(s * n_units)) for g, s in shares.items()}
    labels = np.concatenate(
        [np.full(c, g, dtype=np.float64) for g, c in counts.items()]
        + [np.full(n_units - sum(counts.values()), np.inf)]
    )
    rng.shuffle(labels)
    alpha = rng.normal(0, unit_sd, n_units)
    trend = rng.normal(0, trend_sd, n_units)
    gamma = np.cumsum(rng.normal(0.1, 0.3, len(times)))

    unit = np.repeat(np.arange(n_units), len(times))
    t = np.tile(times, n_units)
    g = labels[unit]
    y0 = alpha[unit] + gamma[np.tile(np.arange(len(times)), n_units)] + trend[unit] * (t - first)
    y0 = y0 + rng.normal(0, noise, len(t))
    tau = np.array([effect(gi, ti - gi) if ti >= gi else 0.0 for gi, ti in zip(g, t, strict=True)])
    return pd.DataFrame(
        {"unit": unit, "time": t, "y": y0 + tau, "y0": y0, "cohort": g, "effect": tau}
    )


def true_att(frame: pd.DataFrame) -> pd.DataFrame:
    """Average effect per (cohort, period) among treated cells: the ATT(g, t) estimands."""
    treated = frame[frame["time"] >= frame["cohort"]]
    return treated.groupby(["cohort", "time"])["effect"].mean().rename("att").reset_index()
