"""Reproduce published numbers. Each check names its source, the printed value, our value
and the tolerance implied by the printed precision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from peffects import bacon, cs, data, studies, synth

DID_README = "Callaway and Sant'Anna, did R package (att_gt / aggte on mpdta)"
MIXTAPE = "Cunningham, The Mixtape, ch. 9 (bacondecomp on castle)"
ADH = "Abadie, Diamond and Hainmueller (2010)"

# ATT(g, t) on mpdta, never-treated controls, no covariates, varying base period.
MPDTA_ATT_GT = {
    (2004, 2004): -0.0105,
    (2004, 2005): -0.0704,
    (2004, 2006): -0.1373,
    (2004, 2007): -0.1008,
    (2006, 2004): 0.0065,
    (2006, 2005): -0.0028,
    (2006, 2006): -0.0046,
    (2006, 2007): -0.0412,
    (2007, 2004): 0.0305,
    (2007, 2005): -0.0027,
    (2007, 2006): -0.0311,
    (2007, 2007): -0.0261,
}
MPDTA_AGGREGATES: dict[cs.Kind, float] = {
    "simple": -0.0400,
    "dynamic": -0.0772,
    "group": -0.0310,
    "calendar": -0.0417,
}
CASTLE_BACON = {
    "treated vs never treated": (0.899, 0.078),
    "earlier vs later treated": (0.077, -0.029),
    "later vs earlier treated": (0.024, 0.046),
}
PROP99_WEIGHTS = {
    "Colorado": 0.164,
    "Connecticut": 0.069,
    "Montana": 0.199,
    "Nevada": 0.234,
    "Utah": 0.334,
}
# Synthetic California's predictor values (Table 1); age share is printed in percent.
PROP99_PREDICTORS = {
    "lnincome (1980-1988)": 9.86,
    "retprice (1980-1988)": 89.41,
    "age15to24 (1980-1988)": 0.1740,
    "beer (1984-1988)": 24.20,
    "cigsale (1988)": 91.62,
    "cigsale (1980)": 120.43,
    "cigsale (1975)": 126.99,
}


@dataclass(frozen=True)
class Check:
    study: str
    quantity: str
    source: str
    published: float
    ours: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return abs(self.ours - self.published) <= self.tolerance + 1e-12


def mpdta_checks(data_dir: Path) -> list[Check]:
    gt = cs.att_gt(data.mpdta(data_dir), reps=0)
    table = gt.table.set_index(["group", "time"])["att"]
    checks = [
        Check("mpdta", f"ATT({g}, {t})", DID_README, value, float(table.loc[(g, t)]), 5e-5)
        for (g, t), value in MPDTA_ATT_GT.items()
    ]
    for kind, value in MPDTA_AGGREGATES.items():
        ours = cs.aggregate(gt, kind, reps=0).overall.estimate
        checks.append(Check("mpdta", f"{kind} aggregate", DID_README, value, ours, 5e-5))
    return checks


def castle_checks(data_dir: Path) -> list[Check]:
    decomposition = bacon.decompose(data.castle(data_dir))
    summary = decomposition.summary()
    weight_of = summary["weight"].to_dict()
    estimate_of = summary["estimate"].to_dict()
    checks = [Check("castle", "TWFE, unweighted", MIXTAPE, 0.069, decomposition.twfe, 5e-4)]
    for kind, (weight, estimate) in CASTLE_BACON.items():
        checks.append(
            Check("castle", f"{kind}: weight", MIXTAPE, weight, float(weight_of[kind]), 5e-4)
        )
        checks.append(
            Check(
                "castle",
                f"{kind}: estimate",
                MIXTAPE,
                estimate,
                float(estimate_of[kind]),
                5e-4,
            )
        )
    return checks


def prop99_checks(data_dir: Path, *, placebos: bool = True) -> list[Check]:
    frame = data.smoking_frame(data_dir)
    fit = synth.synth(
        frame,
        unit="state",
        time="year",
        outcome="cigsale",
        treated="California",
        adoption=1989,
        predictors=studies.PROP99_PREDICTORS,
    )
    weights = dict(zip(fit.donors, fit.weights, strict=True))
    checks = [
        Check("prop99", f"weight on {state}", f"{ADH}, Table 2", value, float(weights[state]), 0.01)
        for state, value in PROP99_WEIGHTS.items()
    ]
    for label, value in PROP99_PREDICTORS.items():
        ours = float(fit.predictors["synthetic"].to_dict()[label])
        checks.append(
            Check("prop99", f"synthetic {label}", f"{ADH}, Table 1", value, ours, 0.01 * abs(value))
        )
    checks.append(Check("prop99", "gap in 2000 (packs)", ADH, -26.0, float(fit.gap[-1]), 1.5))
    if placebos:
        test = synth.placebo_test(
            frame,
            fit,
            unit="state",
            time="year",
            outcome="cigsale",
            predictors=studies.PROP99_PREDICTORS,
        )
        checks.append(Check("prop99", "placebo p-value (1/39)", ADH, 1 / 39, test.p_value, 1e-9))
    return checks


def run(data_dir: Path, *, placebos: bool = True) -> pd.DataFrame:
    checks = (
        mpdta_checks(data_dir)
        + castle_checks(data_dir)
        + prop99_checks(data_dir, placebos=placebos)
    )
    return pd.DataFrame(
        {
            "study": [c.study for c in checks],
            "quantity": [c.quantity for c in checks],
            "published": [c.published for c in checks],
            "ours": [c.ours for c in checks],
            "tolerance": [c.tolerance for c in checks],
            "passed": [c.passed for c in checks],
            "source": [c.source for c in checks],
        }
    )
