"""Run every study, write the numbers to results.json and the page to index.html."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from peffects import cs, data, render, studies, twfe, validate


def sensitivity(data_dir: Path, reps: int, seed: int) -> list[dict[str, Any]]:
    """The castle estimate under the choices a researcher has to make."""
    panel = data.castle(data_dir)
    flat = replace(panel, weight=np.ones(panel.n_units))
    rows: list[dict[str, Any]] = []
    for label, p in (("TWFE, population-weighted", panel), ("TWFE, unweighted", flat)):
        row = twfe.twfe(p).table().to_dict(orient="index")["treated"]
        rows.append(
            {"label": label, "estimate": row["coef"], "low": row["low"], "high": row["high"]}
        )
    specs: tuple[tuple[str, Any, cs.Control], ...] = (
        ("Callaway–Sant'Anna, weighted, never-treated controls", panel, "never"),
        ("Callaway–Sant'Anna, weighted, not-yet-treated controls", panel, "notyet"),
        ("Callaway–Sant'Anna, unweighted, never-treated controls", flat, "never"),
    )
    for label, p, control in specs:
        gt = cs.att_gt(p, control=control, reps=0)
        est = cs.aggregate(gt, "simple", reps=reps, seed=seed).overall
        rows.append({"label": label, **est.as_dict()})
    return rows


def collect(data_dir: Path, *, reps: int = 999, seed: int = 0) -> dict[str, Any]:
    checks = validate.run(data_dir)
    return {
        "generated": dt.date.today().isoformat(),
        "castle": studies.castle_study(data_dir, reps=reps, permutations=reps, seed=seed),
        "sensitivity": sensitivity(data_dir, reps, seed),
        "prop99": studies.prop99_study(data_dir, seed=seed),
        "validation": [
            {k: (bool(v) if k == "passed" else v) for k, v in row.items()}
            for row in checks.to_dict(orient="records")
        ],
    }


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_page(results: dict[str, Any], out: Path) -> Path:
    page = out / "index.html"
    page.write_text(render.page(results), encoding="utf-8")
    return page


def build(data_dir: Path, out: Path, *, reps: int = 999, seed: int = 0) -> Path:
    results = _strict(collect(data_dir, reps=reps, seed=seed))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, indent=1, allow_nan=False), encoding="utf-8"
    )
    return write_page(results, out)
