"""A validated panel: one outcome per unit and period, with each unit's adoption date.

Treatment is staggered and absorbing: a unit is untreated before its cohort (the first
treated period) and treated from then on. Never-treated units have cohort ``inf``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

NEVER = np.inf
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class PanelError(ValueError):
    pass


@dataclass(frozen=True)
class Panel:
    units: NDArray[np.object_]
    times: FloatArray
    unit_index: IntArray
    time_index: IntArray
    y: FloatArray
    cohort: FloatArray  # per unit: first treated period, inf if never
    weight: FloatArray  # per unit, constant over time
    x: FloatArray  # per observation covariates, shape (n_obs, k)
    x_names: tuple[str, ...] = ()
    outcome: str = "y"

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def n_obs(self) -> int:
        return len(self.y)

    @property
    def balanced(self) -> bool:
        return self.n_obs == self.n_units * self.n_times

    @property
    def time(self) -> FloatArray:
        """Calendar time of each observation."""
        return self.times[self.time_index]

    @property
    def unit_cohort(self) -> FloatArray:
        """Cohort of each observation's unit."""
        return self.cohort[self.unit_index]

    @property
    def treated(self) -> NDArray[np.bool_]:
        return self.time >= self.unit_cohort

    @property
    def cohorts(self) -> FloatArray:
        """Adoption dates of units first treated inside the sample window."""
        finite = self.cohort[np.isfinite(self.cohort) & (self.cohort > self.times[0])]
        return np.unique(finite)

    @property
    def never_treated(self) -> NDArray[np.bool_]:
        return ~np.isfinite(self.cohort)

    @property
    def always_treated(self) -> NDArray[np.bool_]:
        return np.asarray(self.cohort <= self.times[0], dtype=bool)

    def wide(self) -> FloatArray:
        """Outcomes as a units x periods matrix, NaN where unobserved."""
        out = np.full((self.n_units, self.n_times), np.nan)
        out[self.unit_index, self.time_index] = self.y
        return out

    def keep_units(self, mask: NDArray[np.bool_]) -> Panel:
        """A panel with only the units where ``mask`` is true (re-indexed)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_units,):
            raise PanelError("mask must have one entry per unit")
        new_code = np.cumsum(mask) - 1
        rows = mask[self.unit_index]
        return replace(
            self,
            units=self.units[mask],
            unit_index=new_code[self.unit_index[rows]].astype(np.int64),
            time_index=self.time_index[rows],
            y=self.y[rows],
            cohort=self.cohort[mask],
            weight=self.weight[mask],
            x=self.x[rows],
        )

    def keep_times(self, mask: NDArray[np.bool_]) -> Panel:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_times,):
            raise PanelError("mask must have one entry per period")
        new_code = np.cumsum(mask) - 1
        rows = mask[self.time_index]
        return replace(
            self,
            times=self.times[mask],
            unit_index=self.unit_index[rows],
            time_index=new_code[self.time_index[rows]].astype(np.int64),
            y=self.y[rows],
            x=self.x[rows],
        )

    def with_cohorts(self, cohort: FloatArray) -> Panel:
        """The same outcomes with a different adoption schedule (used by placebo tests)."""
        cohort = np.asarray(cohort, dtype=np.float64)
        if cohort.shape != (self.n_units,):
            raise PanelError("need one cohort per unit")
        return replace(self, cohort=cohort)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "unit": self.units[self.unit_index],
                "time": self.time,
                self.outcome: self.y,
                "cohort": self.unit_cohort,
                "treated": self.treated.astype(int),
                "weight": self.weight[self.unit_index],
            }
        )
        for j, name in enumerate(self.x_names):
            frame[name] = self.x[:, j]
        return frame

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        unit: str,
        time: str,
        outcome: str,
        cohort: str | None = None,
        treat: str | None = None,
        weight: str | None = None,
        covariates: Sequence[str] = (),
    ) -> Panel:
        """Build a panel from long data.

        Give either ``cohort`` (first treated period; 0, NaN or inf mean never treated) or
        ``treat`` (a 0/1 indicator, which must be absorbing). Units adopting after the last
        period are recorded as never treated.
        """
        if (cohort is None) == (treat is None):
            raise PanelError("give exactly one of cohort= or treat=")
        columns = [unit, time, outcome, *([cohort] if cohort else []), *([treat] if treat else [])]
        columns += [*([weight] if weight else []), *covariates]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise PanelError(f"columns not found: {missing}")
        data = df[columns].copy()
        if data[[unit, time, outcome, *covariates]].isna().any().any():
            raise PanelError("unit, time, outcome and covariates must not be missing")
        if data.duplicated([unit, time]).any():
            raise PanelError("more than one row per unit and period")

        unit_codes, units = pd.factorize(data[unit], sort=True)
        time_values = data[time].to_numpy(np.float64)
        times = np.unique(time_values)
        time_codes = np.searchsorted(times, time_values)

        if treat is not None:
            per_unit = _cohort_from_indicator(
                data[treat].to_numpy(np.float64), unit_codes, time_values, len(units)
            )
        else:
            assert cohort is not None
            raw = data[cohort].to_numpy(np.float64)
            raw = np.where(np.isnan(raw) | (raw == 0), NEVER, raw)
            per_unit = _constant_per_unit(raw, unit_codes, len(units), "cohort")

        if weight is not None:
            w = _constant_per_unit(
                data[weight].to_numpy(np.float64), unit_codes, len(units), "weight"
            )
            if np.any(~np.isfinite(w)) or np.any(w <= 0):
                raise PanelError("weights must be positive")
        else:
            w = np.ones(len(units))

        # Adoption after the last period is never observed: within the sample it is never.
        per_unit = np.where(per_unit > times[-1], NEVER, per_unit)
        order = np.lexsort((time_codes, unit_codes))
        x = data[list(covariates)].to_numpy(np.float64) if covariates else np.zeros((len(data), 0))
        return cls(
            units=np.asarray(units, dtype=object),
            times=times,
            unit_index=unit_codes[order].astype(np.int64),
            time_index=time_codes[order].astype(np.int64),
            y=data[outcome].to_numpy(np.float64)[order],
            cohort=per_unit,
            weight=w,
            x=x[order],
            x_names=tuple(covariates),
            outcome=outcome,
        )


def _constant_per_unit(values: FloatArray, codes: IntArray, n: int, name: str) -> FloatArray:
    out = np.full(n, np.nan)
    out[codes] = values
    filled = out[codes]
    same = (filled == values) | (np.isnan(filled) & np.isnan(values))
    if not same.all():
        raise PanelError(f"{name} must be constant within each unit")
    return out


def _cohort_from_indicator(d: FloatArray, codes: IntArray, t: FloatArray, n: int) -> FloatArray:
    if not np.isin(d, [0.0, 1.0]).all():
        raise PanelError("the treatment indicator must be 0 or 1")
    first = np.full(n, NEVER)
    np.minimum.at(first, codes[d == 1], t[d == 1])
    implied = t >= first[codes]
    if np.any(implied != (d == 1)):
        raise PanelError("treatment must be absorbing: once treated, a unit stays treated")
    return first
