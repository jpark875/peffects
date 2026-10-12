"""Command line: ``peffects fetch | validate | did | csv | synth | report``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from peffects import data, studies, validate
from peffects.panel import Panel, PanelError

DEFAULT_DATA = Path("data")


def fmt(value: float | None, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def ci(d: dict[str, Any]) -> str:
    return f"[{fmt(d.get('low'))}, {fmt(d.get('high'))}]"


def staggered_lines(name: str, r: dict[str, Any]) -> list[str]:
    cs_ = r["callaway_santanna"]
    inf = r["inference"]
    first, last = r["periods"]
    lines = [
        f"{name}: {r.get('outcome', 'outcome')}, {r['units']} units, {first:g}-{last:g}, "
        f"{r['never_treated']} never treated",
        "  cohorts: " + ", ".join(f"{k} ({v})" for k, v in r["cohorts"].items()),
        "",
        f"  {'estimator':<36}{'estimate':>10}  {'95% interval':<20} p",
        f"  {'TWFE, cluster-robust':<36}{fmt(r['twfe']['coef']):>10}  {ci(r['twfe']):<20} "
        f"{fmt(inf['cluster_robust_p'], 3)}",
        f"  {'TWFE, wild cluster bootstrap':<36}{'':>10}  {ci(inf['wild_bootstrap']):<20} "
        f"{fmt(inf['wild_bootstrap']['p'], 3)}",
        f"  {'TWFE, randomization inference':<36}{'':>10}  {'':<20} "
        f"{fmt(inf['randomization']['p'], 3)}",
    ]
    for kind in ("simple", "group", "dynamic"):
        est = cs_[kind]
        label = f"Callaway-Sant'Anna, {kind}"
        lines.append(f"  {label:<36}{fmt(est['estimate']):>10}  {ci(est):<20}")
    sa = r["sun_abraham"]["overall"]
    lines.append(f"  {'Sun-Abraham':<36}{fmt(sa['estimate']):>10}  {ci(sa):<20}")
    lines += [
        "",
        f"  Goodman-Bacon decomposition of unweighted TWFE ({fmt(r['twfe_unweighted'])}):",
    ]
    for row in r["bacon"]["summary"]:
        lines.append(
            f"    {row['kind']:<30} weight {fmt(row['weight'], 3)}  estimate {fmt(row['estimate'])}"
        )
    return lines


def _write_json(path: str | None, payload: Any) -> None:
    if path:
        Path(path).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")


def cmd_fetch(args: argparse.Namespace) -> int:
    for name in data.SOURCES:
        print(f"{name:<8} {data.fetch(name, args.data_dir, force=args.force)}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    table = validate.run(args.data_dir, placebos=not args.quick)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(table.drop(columns="source").to_string(index=False, float_format="{:.4f}".format))
    passed = int(table["passed"].sum())
    print(f"\n{passed}/{len(table)} published values reproduced")
    return 0 if passed == len(table) else 1


def _staggered(args: argparse.Namespace, panel: Panel, name: str) -> int:
    result = studies.staggered_study(
        panel, control=args.control, reps=args.reps, permutations=args.permutations, seed=args.seed
    )
    result["outcome"] = panel.outcome
    print("\n".join(staggered_lines(name, result)))
    _write_json(args.json, result)
    return 0


def cmd_did(args: argparse.Namespace) -> int:
    panel = data.castle(args.data_dir) if args.dataset == "castle" else data.mpdta(args.data_dir)
    return _staggered(args, panel, args.dataset)


def cmd_csv(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.path)
    panel = Panel.from_frame(
        frame,
        unit=args.unit,
        time=args.time,
        outcome=args.outcome,
        cohort=args.cohort,
        treat=args.treat,
        weight=args.weight,
    )
    return _staggered(args, panel, args.path.name)


def cmd_synth(args: argparse.Namespace) -> int:
    r = studies.prop99_study(args.data_dir, placebos=not args.no_placebos, seed=args.seed)
    print("Synthetic California (Proposition 99, 1989)")
    for row in r["weights"]:
        print(f"  {row['donor']:<16}{row['weight']:.3f}")
    print(
        f"  pre-1989 RMSPE {r['pre_rmspe']:.2f}; gap in 2000 {r['gap_last']:.1f} packs per capita"
    )
    if "placebo" in r:
        pl = r["placebo"]
        print(f"  placebo test: rank {pl['rank']} of {pl['units']}, p = {pl['p_value']:.3f}")
    _write_json(args.json, r)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from peffects import report

    out = report.build(args.data_dir, args.out, reps=args.reps, seed=args.seed)
    print(f"report -> {out}")
    return 0


def _inference_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--control", choices=["never", "notyet"], default="never")
    p.add_argument("--reps", type=int, default=999, help="bootstrap replicates")
    p.add_argument("--permutations", type=int, default=999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write every result as JSON to this path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="peffects", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download and verify the replication datasets")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("validate", help="reproduce published estimates")
    p.add_argument("--quick", action="store_true", help="skip the synthetic control placebos")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("did", help="every staggered DiD estimator on a replication dataset")
    p.add_argument("dataset", choices=["castle", "mpdta"])
    _inference_options(p)
    p.set_defaults(func=cmd_did)

    p = sub.add_parser("csv", help="every staggered DiD estimator on your own long panel")
    p.add_argument("path", type=Path)
    p.add_argument("--unit", required=True)
    p.add_argument("--time", required=True)
    p.add_argument("--outcome", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--cohort", help="first treated period (0 or empty = never)")
    group.add_argument("--treat", help="absorbing 0/1 treatment indicator")
    p.add_argument("--weight", help="unit weight, constant over time")
    _inference_options(p)
    p.set_defaults(func=cmd_csv)

    p = sub.add_parser("synth", help="synthetic control for California's Proposition 99")
    p.add_argument("--no-placebos", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("report", help="build the HTML report")
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.add_argument("--reps", type=int, default=999)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (data.DataError, PanelError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
