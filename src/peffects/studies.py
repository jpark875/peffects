"""The three applications, each run end to end and returned as plain data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from peffects import bacon, cs, data, inference, sunab, synth, twfe
from peffects.panel import Panel

PROP99_PREDICTORS = (
    synth.Predictor("lnincome", 1980, 1988),
    synth.Predictor("retprice", 1980, 1988),
    synth.Predictor("age15to24", 1980, 1988),
    synth.Predictor("beer", 1984, 1988),
    synth.Predictor("cigsale", 1988),
    synth.Predictor("cigsale", 1980),
    synth.Predictor("cigsale", 1975),
)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (float, np.floating)):
                v = float(value)
                clean[str(key)] = None if not np.isfinite(v) else v
            elif isinstance(value, (np.integer,)):
                clean[str(key)] = int(value)
            elif isinstance(value, (np.bool_,)):
                clean[str(key)] = bool(value)
            else:
                clean[str(key)] = value
        out.append(clean)
    return out


def _fit_dict(fit: Any, name: str = "treated") -> dict[str, float]:
    row = fit.table().loc[name]
    return {k: float(row[k]) for k in ("coef", "se", "p", "low", "high")}


def staggered_study(
    panel: Panel,
    *,
    control: cs.Control = "never",
    reps: int = 999,
    permutations: int = 999,
    seed: int = 0,
) -> dict[str, Any]:
    """Every staggered-adoption estimator on one panel, with inference."""
    static = twfe.twfe(panel)
    decomposition = bacon.decompose(panel)
    es_twfe = twfe.event_study(panel)
    gt = cs.att_gt(panel, control=control, reps=reps, seed=seed)
    simple = cs.aggregate(gt, "simple", reps=reps, seed=seed)
    dynamic = cs.aggregate(gt, "dynamic", reps=reps, seed=seed)
    group = cs.aggregate(gt, "group", reps=reps, seed=seed)
    sa = sunab.sun_abraham(panel)
    wcr = inference.wild_cluster_bootstrap(panel, reps=reps, seed=seed)

    def twfe_stat(p: Panel) -> float:
        return float(twfe.twfe(p)["treated"])

    ri = inference.randomization_inference(panel, twfe_stat, reps=permutations, seed=seed)
    cohorts = pd.Series(panel.cohort).replace(np.inf, np.nan).value_counts().sort_index()
    return {
        "units": panel.n_units,
        "periods": [float(panel.times[0]), float(panel.times[-1])],
        "cohorts": {f"{k:g}": int(v) for k, v in cohorts.items()},
        "never_treated": int(panel.never_treated.sum()),
        "twfe": _fit_dict(static),
        "twfe_unweighted": float(decomposition.twfe),
        "bacon": {
            "comparisons": _records(decomposition.comparisons),
            "summary": _records(decomposition.summary().reset_index()),
            "check": decomposition.estimate,
        },
        "event_study": {
            "twfe": _records(es_twfe.estimates),
            "callaway_santanna": _records(
                dynamic.table.rename(columns={"key": "e", "att": "coef"})
            ),
            "sun_abraham": _records(sa.event_study.estimates),
        },
        "callaway_santanna": {
            "control": control,
            "group_time": _records(gt.table),
            "simple": simple.overall.as_dict(),
            "dynamic": dynamic.overall.as_dict(),
            "group": group.overall.as_dict(),
            "dropped_units": gt.dropped_units,
        },
        "sun_abraham": {"overall": sa.overall.as_dict(), "control": sa.control},
        "inference": {
            "cluster_robust_p": _fit_dict(static)["p"],
            "wild_bootstrap": {
                "p": wcr.p_value,
                "low": wcr.low,
                "high": wcr.high,
                "reps": wcr.reps,
                "clusters": wcr.clusters,
            },
            "randomization": {"p": ri.p_value, "permutations": permutations},
        },
    }


def castle_study(data_dir: Path, **kwargs: Any) -> dict[str, Any]:
    result = staggered_study(data.castle(data_dir), **kwargs)
    result["source"] = data.SOURCES["castle"].citation
    result["outcome"] = "log homicides per 100,000"
    return result


def mpdta_study(data_dir: Path, **kwargs: Any) -> dict[str, Any]:
    result = staggered_study(data.mpdta(data_dir), **kwargs)
    result["source"] = data.SOURCES["mpdta"].citation
    result["outcome"] = "log teen employment"
    return result


def prop99_study(data_dir: Path, *, placebos: bool = True, seed: int = 0) -> dict[str, Any]:
    frame = data.smoking_frame(data_dir)
    kwargs: dict[str, Any] = {"unit": "state", "time": "year", "outcome": "cigsale"}
    fit = synth.synth(
        frame,
        treated="California",
        adoption=1989,
        predictors=PROP99_PREDICTORS,
        seed=seed,
        **kwargs,
    )
    result: dict[str, Any] = {
        "source": data.SOURCES["smoking"].citation,
        "weights": _records(fit.weight_table()),
        "predictors": _records(fit.predictors.reset_index(names="predictor")),
        "path": _records(
            pd.DataFrame(
                {"year": fit.times, "california": fit.treated_path, "synthetic": fit.synthetic_path}
            )
        ),
        "pre_rmspe": fit.rmspe(False),
        "post_rmspe": fit.rmspe(True),
        "gap_last": float(fit.gap[-1]),
        "gap_mean_post": float(fit.gap[~fit.pre].mean()),
    }
    if placebos:
        test = synth.placebo_test(frame, fit, predictors=PROP99_PREDICTORS, seed=seed, **kwargs)
        gaps = test.gaps(max_pre_ratio=5)
        result["placebo"] = {
            "p_value": test.p_value,
            "units": len(test.placebos) + 1,
            "rank": int(list(test.ratios().index).index("California")) + 1,
            "ratios": [{"state": str(k), "ratio": float(v)} for k, v in test.ratios().items()],
            "gaps": {str(c): [float(x) for x in gaps[c]] for c in gaps.columns},
            "kept_at_5x": int(gaps.shape[1]) - 1,
        }
    return result
