"""Static HTML for the report: inline SVG charts, each with a table view, no dependencies."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Callable, Sequence
from importlib.resources import files
from typing import Any

Scale = Callable[[float], float]
CSS = files("peffects").joinpath("assets/report.css").read_text("utf-8")
SCRIPT = files("peffects").joinpath("assets/report.js").read_text("utf-8")
ESTIMATORS = (
    ("twfe", "TWFE event study", "s1", "circle"),
    ("callaway_santanna", "Callaway–Sant'Anna", "s2", "square"),
    ("sun_abraham", "Sun–Abraham", "s3", "diamond"),
)
BACON_KINDS = {
    "treated vs never treated": "s1",
    "earlier vs later treated": "s2",
    "later vs earlier treated": "s3",
    "later vs always treated": "s3",
}


class Raw(str):
    """Markup that is already escaped."""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def num(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}".replace("-", "−")


def pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{value * 100:.{digits}f}%".replace("-", "−")


def interval(low: float | None, high: float | None, digits: int = 3) -> str:
    if low is None or high is None:
        return "–"
    return f"[{num(low, digits)}, {num(high, digits)}]"


def tip(lines: Sequence[str]) -> str:
    return f'data-tip="{esc(json.dumps(list(lines)))}"'


def linear(d0: float, d1: float, r0: float, r1: float) -> Scale:
    span = (d1 - d0) or 1.0
    return lambda v: r0 + (v - d0) / span * (r1 - r0)


def nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    if raw <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * power:
            return float(mult * power)
    return float(10 * power)


def ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick values whose range covers [lo, hi]."""
    step = nice_step(hi - lo, target)
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    count = int(round((end - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool]) -> str:
    head = "".join(
        f'<th scope="col"{" class=num" if n else ""}>{esc(h)}</th>'
        for h, n in zip(headers, numeric, strict=True)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td{' class=num' if n else ''}>{c if isinstance(c, Raw) else esc(c)}</td>"
            for c, n in zip(r, numeric, strict=True)
        )
        + "</tr>"
        for r in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def details(summary: str, inner: str) -> str:
    return f"<details><summary>{esc(summary)}</summary>{inner}</details>"


def legend(items: Sequence[tuple[str, str]]) -> str:
    keys = "".join(
        f'<span class="key"><span class="swatch {cls}"></span>{esc(name)}</span>'
        for name, cls in items
    )
    return f'<div class="legend">{keys}</div>'


def marker(shape: str, cls: str, x: float, y: float) -> str:
    if shape == "square":
        return f'<rect class="dot {cls}" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5"/>'
    if shape == "diamond":
        return (
            f'<path class="dot {cls}" d="M{x:.1f} {y - 5.5:.1f}L{x + 5.5:.1f} {y:.1f}'
            f'L{x:.1f} {y + 5.5:.1f}L{x - 5.5:.1f} {y:.1f}Z"/>'
        )
    return f'<circle class="dot {cls}" cx="{x:.1f}" cy="{y:.1f}" r="4"/>'


def y_axis(
    parts: list[str], values: list[float], y: Scale, left: float, right: float, digits: int
) -> None:
    for t in values:
        parts.append(
            f'<line class="grid" x1="{left}" x2="{right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{num(t, digits)}</text>'
        )


def axis_titles(
    parts: list[str],
    x_title: str,
    y_title: str,
    left: float,
    plot_w: float,
    top: float,
    plot_h: float,
    height: float,
) -> None:
    parts.append(
        f'<text class="axis" x="{left + plot_w / 2}" y="{height - 6}" text-anchor="middle">{esc(x_title)}</text>'
    )
    parts.append(
        f'<text class="axis" transform="rotate(-90)" x="{-(top + plot_h / 2)}" y="14" text-anchor="middle">{esc(y_title)}</text>'
    )


# Charts -------------------------------------------------------------------------------


def event_study_chart(series: dict[str, list[dict[str, Any]]], window: tuple[int, int]) -> str:
    lo_e, hi_e = window
    left, right, top, bottom, plot_w, plot_h = 60, 20, 16, 48, 580, 280
    width, height = left + plot_w + right, top + plot_h + bottom
    events = list(range(lo_e, hi_e + 1))
    step = plot_w / len(events)

    def xc(e: float) -> float:
        return left + step * (e - lo_e + 0.5)

    rows = {
        key: [r for r in series.get(key, []) if lo_e <= r["e"] <= hi_e] for key, *_ in ESTIMATORS
    }
    values = [
        v
        for rs in rows.values()
        for r in rs
        for v in (r.get("low"), r.get("high"), r["coef"])
        if v is not None
    ]
    lo, hi = min(values + [0.0]), max(values + [0.0])
    pad = (hi - lo) * 0.08
    grid = ticks(lo - pad, hi + pad, 5)
    y = linear(min(grid[0], lo - pad), max(grid[-1], hi + pad), top + plot_h, top)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="es-title">']
    parts.append(
        '<title id="es-title">Event study: effect by years since adoption, three estimators</title>'
    )
    y_axis(parts, grid, y, left, left + plot_w, 2)
    parts.append(
        f'<line class="zero" x1="{left}" x2="{left + plot_w}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>'
    )
    parts.append(
        f'<line class="reference" x1="{(xc(-1) + xc(0)) / 2:.1f}" x2="{(xc(-1) + xc(0)) / 2:.1f}" y1="{top}" y2="{top + plot_h}"/>'
    )
    parts.append(
        f'<text class="note" x="{(xc(-1) + xc(0)) / 2 + 6:.1f}" y="{top + 12}">law takes effect</text>'
    )
    for e in events:
        parts.append(
            f'<text class="tick" x="{xc(e):.1f}" y="{top + plot_h + 18}" text-anchor="middle">{e}</text>'
        )
    hits = []
    for i, (key, name, cls, shape) in enumerate(ESTIMATORS):
        offset = (i - 1) * step * 0.22
        for r in rows[key]:
            cx, cy = xc(r["e"]) + offset, y(r["coef"])
            if r.get("low") is not None and r.get("high") is not None:
                parts.append(
                    f'<line class="whisker {cls}" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(r["low"]):.1f}" y2="{y(r["high"]):.1f}"/>'
                )
            parts.append(marker(shape, cls, cx, cy))
            reference = r.get("se") is None
            e = int(r["e"])
            lines = [
                f"{name}, {e:+d} years" if e else f"{name}, adoption year",
                "Reference period (normalized to 0)" if reference else f"Estimate {num(r['coef'])}",
            ]
            if not reference:
                lines.append(f"95% interval {interval(r.get('low'), r.get('high'))}")
            hits.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="9" {tip(lines)}/>')
    parts.extend(hits)
    axis_titles(
        parts,
        "Years since the law took effect",
        "Effect on log homicides",
        left,
        plot_w,
        top,
        plot_h,
        height,
    )
    parts.append("</svg>")
    return "".join(parts)


def bacon_chart(comparisons: list[dict[str, Any]], twfe: float) -> str:
    left, right, top, bottom, plot_w, plot_h = 60, 20, 16, 48, 580, 280
    width, height = left + plot_w + right, top + plot_h + bottom
    weights = [c["weight"] for c in comparisons]
    estimates = [c["estimate"] for c in comparisons]
    xg = ticks(0, max(weights) * 1.08, 5)
    x = linear(0, max(xg[-1], max(weights)), left, left + plot_w)
    lo, hi = min(estimates + [0.0, twfe]), max(estimates + [0.0, twfe])
    pad = (hi - lo) * 0.08
    yg = ticks(lo - pad, hi + pad, 5)
    y = linear(min(yg[0], lo - pad), max(yg[-1], hi + pad), top + plot_h, top)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="bacon-title">']
    parts.append(
        '<title id="bacon-title">Every 2x2 comparison inside the TWFE estimate: weight against estimate</title>'
    )
    y_axis(parts, yg, y, left, left + plot_w, 2)
    for t in xg:
        parts.append(
            f'<text class="tick" x="{x(t):.1f}" y="{top + plot_h + 18}" text-anchor="middle">{num(t, 2)}</text>'
        )
    parts.append(
        f'<line class="zero" x1="{left}" x2="{left + plot_w}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>'
    )
    parts.append(
        f'<line class="reference" x1="{left}" x2="{left + plot_w}" y1="{y(twfe):.1f}" y2="{y(twfe):.1f}"/>'
    )
    parts.append(
        f'<text class="note" x="{left + plot_w - 4}" y="{y(twfe) - 6:.1f}" text-anchor="end">TWFE = {num(twfe)}</text>'
    )
    shapes = {"s1": "circle", "s2": "square", "s3": "diamond"}
    hits = []
    for c in sorted(comparisons, key=lambda c: -c["weight"]):
        cls = BACON_KINDS.get(c["kind"], "s1")
        cx, cy = x(c["weight"]), y(c["estimate"])
        parts.append(marker(shapes[cls], cls, cx, cy))
        lines = [
            f"{c['treated']} vs {c['control']}",
            c["kind"].capitalize(),
            f"Estimate {num(c['estimate'])}, weight {num(c['weight'], 3)}",
        ]
        hits.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="9" {tip(lines)}/>')
    parts.extend(hits)
    axis_titles(
        parts, "Weight in the TWFE estimate", "2×2 DiD estimate", left, plot_w, top, plot_h, height
    )
    parts.append("</svg>")
    return "".join(parts)


def _year_axis(parts: list[str], years: list[float], x: Scale, base: float) -> None:
    for yr in years:
        if int(yr) % 5 == 0:
            parts.append(
                f'<text class="tick" x="{x(yr):.1f}" y="{base + 18}" text-anchor="middle">{int(yr)}</text>'
            )


def synth_path_chart(path: list[dict[str, Any]], adoption: float) -> str:
    left, right, top, bottom, plot_w, plot_h = 60, 20, 16, 48, 580, 280
    width, height = left + plot_w + right, top + plot_h + bottom
    years = [p["year"] for p in path]
    x = linear(years[0], years[-1], left, left + plot_w)
    values = [p["california"] for p in path] + [p["synthetic"] for p in path]
    yg = ticks(0, max(values), 5)
    y = linear(0, yg[-1], top + plot_h, top)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="sc-title">']
    parts.append(
        '<title id="sc-title">Cigarette sales per capita: California and its synthetic control</title>'
    )
    y_axis(parts, yg, y, left, left + plot_w, 0)
    _year_axis(parts, years, x, top + plot_h)
    xa = x(adoption - 0.5)
    parts.append(
        f'<line class="reference" x1="{xa:.1f}" x2="{xa:.1f}" y1="{top}" y2="{top + plot_h}"/>'
    )
    parts.append(
        f'<text class="note" x="{xa - 6:.1f}" y="{top + 12}" text-anchor="end">Proposition 99</text>'
    )
    for key, cls in (("synthetic", "s2 dashed"), ("california", "s1")):
        pts = " ".join(f"{x(p['year']):.1f},{y(p[key]):.1f}" for p in path)
        parts.append(f'<polyline class="line {cls}" points="{pts}"/>')
    parts.append(
        f'<line class="crosshair" x1="0" x2="0" y1="{top}" y2="{top + plot_h}" visibility="hidden"/>'
    )
    parts.append('<circle class="focus s1" cx="0" cy="0" r="4" visibility="hidden"/>')
    hover = [
        [
            round(x(p["year"]), 1),
            round(y(p["california"]), 1),
            f"{int(p['year'])}",
            f"California {num(p['california'], 1)} packs",
            f"Synthetic {num(p['synthetic'], 1)} packs",
            f"Gap {num(p['california'] - p['synthetic'], 1)}",
        ]
        for p in path
    ]
    parts.append(
        f'<rect class="hit" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" data-points="{esc(json.dumps(hover))}"/>'
    )
    axis_titles(parts, "Year", "Packs per capita", left, plot_w, top, plot_h, height)
    parts.append("</svg>")
    return "".join(parts)


def placebo_chart(
    years: list[float], gaps: dict[str, list[float]], treated: str, adoption: float
) -> str:
    left, right, top, bottom, plot_w, plot_h = 60, 20, 16, 48, 580, 280
    width, height = left + plot_w + right, top + plot_h + bottom
    x = linear(years[0], years[-1], left, left + plot_w)
    flat = [v for series in gaps.values() for v in series]
    bound = max(abs(min(flat)), abs(max(flat))) * 1.05
    yg = ticks(-bound, bound, 6)
    y = linear(min(yg[0], -bound), max(yg[-1], bound), top + plot_h, top)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="pl-title">']
    parts.append('<title id="pl-title">Gap between each state and its synthetic control</title>')
    y_axis(parts, yg, y, left, left + plot_w, 0)
    _year_axis(parts, years, x, top + plot_h)
    parts.append(
        f'<line class="zero" x1="{left}" x2="{left + plot_w}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>'
    )
    xa = x(adoption - 0.5)
    parts.append(
        f'<line class="reference" x1="{xa:.1f}" x2="{xa:.1f}" y1="{top}" y2="{top + plot_h}"/>'
    )
    for name, series in gaps.items():
        if name == treated:
            continue
        pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(years, series, strict=True))
        parts.append(f'<polyline class="ghost" points="{pts}"/>')
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(years, gaps[treated], strict=True))
    parts.append(f'<polyline class="line s1" points="{pts}"/>')
    parts.append(
        f'<line class="crosshair" x1="0" x2="0" y1="{top}" y2="{top + plot_h}" visibility="hidden"/>'
    )
    parts.append('<circle class="focus s1" cx="0" cy="0" r="4" visibility="hidden"/>')
    others = [s for n, s in gaps.items() if n != treated]
    hover = []
    for i, t in enumerate(years):
        column = [s[i] for s in others]
        hover.append(
            [
                round(x(t), 1),
                round(y(gaps[treated][i]), 1),
                f"{int(t)}",
                f"{treated} gap {num(gaps[treated][i], 1)}",
                f"Placebo states {num(min(column), 1)} to {num(max(column), 1)}",
            ]
        )
    parts.append(
        f'<rect class="hit" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" data-points="{esc(json.dumps(hover))}"/>'
    )
    axis_titles(
        parts, "Year", "Gap vs synthetic (packs per capita)", left, plot_w, top, plot_h, height
    )
    parts.append("</svg>")
    return "".join(parts)


# Page ---------------------------------------------------------------------------------


def stat(label: str, value: str, note: str) -> str:
    return (
        f'<div class="stat"><div class="meta">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div><div class="muted small">{esc(note)}</div></div>'
    )


def section(ident: str, title: str, aside: str, body: str) -> str:
    return (
        f'<section id="{ident}"><div class="aside"><h2>{esc(title)}</h2>'
        f'<p class="muted">{esc(aside)}</p></div><div class="body">{body}</div></section>'
    )


def log_pct(value: float) -> str:
    """A log-point effect as a percent change."""
    return f"{(math.exp(value) - 1) * 100:+.1f}%".replace("-", "−")


def castle_section(r: dict[str, Any], window: tuple[int, int]) -> str:
    c = r["castle"]
    es = c["event_study"]
    body = legend([(name, cls) for _, name, cls, _ in ESTIMATORS])
    body += f'<figure class="chart">{event_study_chart(es, window)}</figure>'
    rows = []
    for key, name, *_ in ESTIMATORS:
        for row in es[key]:
            if window[0] <= row["e"] <= window[1]:
                rows.append(
                    [
                        name,
                        f"{int(row['e']):+d}",
                        num(row["coef"]),
                        interval(row.get("low"), row.get("high")),
                    ]
                )
    body += details(
        "Show the event-study table",
        table(["Estimator", "Years", "Estimate", "95% interval"], rows, [False, True, True, True]),
    )
    cs_ = c["callaway_santanna"]
    sa = c["sun_abraham"]["overall"]
    summary = [
        [
            "TWFE (static)",
            num(c["twfe"]["coef"]),
            log_pct(c["twfe"]["coef"]),
            interval(c["twfe"]["low"], c["twfe"]["high"]),
            "cluster-robust",
        ],
        [
            "Callaway–Sant'Anna, overall",
            num(cs_["simple"]["estimate"]),
            log_pct(cs_["simple"]["estimate"]),
            interval(cs_["simple"]["low"], cs_["simple"]["high"]),
            "influence function",
        ],
        [
            "Callaway–Sant'Anna, by cohort",
            num(cs_["group"]["estimate"]),
            log_pct(cs_["group"]["estimate"]),
            interval(cs_["group"]["low"], cs_["group"]["high"]),
            "influence function",
        ],
        [
            "Sun–Abraham, overall",
            num(sa["estimate"]),
            log_pct(sa["estimate"]),
            interval(sa["low"], sa["high"]),
            "cluster-robust, shares fixed",
        ],
    ]
    body += table(
        ["Estimator", "Log points", "Change", "95% interval", "Inference"],
        summary,
        [False, True, True, True, False],
    )
    aside = (
        f"Between 2005 and 2009, {sum(c['cohorts'].values())} states passed laws removing the duty "
        "to retreat before using lethal force. Each estimator compares their homicide rates with "
        f"the {c['never_treated']} states that never did. Points are years relative to the first "
        "full year under the law; whiskers are 95% intervals. Years far from adoption rest on one "
        "or two states and are not shown."
    )
    return section("castle", "Castle-doctrine laws", aside, body)


def bacon_section(r: dict[str, Any]) -> str:
    c = r["castle"]
    b = c["bacon"]
    keys = [
        (k.capitalize(), cls)
        for k, cls in BACON_KINDS.items()
        if any(s["kind"] == k for s in b["summary"])
    ]
    body = legend(keys)
    body += f'<figure class="chart">{bacon_chart(b["comparisons"], c["twfe_unweighted"])}</figure>'
    body += table(
        ["Comparison", "Pairs", "Total weight", "Weighted estimate"],
        [
            [s["kind"], str(s["comparisons"]), num(s["weight"], 3), num(s["estimate"])]
            for s in b["summary"]
        ],
        [False, True, True, True],
    )
    rows = [
        [f"{x['treated']} vs {x['control']}", x["kind"], num(x["weight"], 4), num(x["estimate"])]
        for x in b["comparisons"]
    ]
    body += details(
        "Show all 2×2 comparisons",
        table(["Pair", "Kind", "Weight", "Estimate"], rows, [False, False, True, True]),
    )
    later = next((s for s in b["summary"] if s["kind"] == "later vs earlier treated"), None)
    share = f"{later['weight'] * 100:.1f}%" if later else "none"
    aside = (
        "The static TWFE coefficient is a weighted average of every two-group, two-period "
        f"comparison in the data. Here {share} of the weight comes from comparisons that use "
        "already-treated states as controls, which are biased when effects change over time. "
        f"The weights reproduce the unweighted TWFE estimate exactly ({num(b['check'], 4)})."
    )
    return section("bacon", "Inside the TWFE estimate", aside, body)


def inference_section(r: dict[str, Any]) -> str:
    inf = r["castle"]["inference"]
    wb, ri = inf["wild_bootstrap"], inf["randomization"]
    tw = r["castle"]["twfe"]
    rows = [
        [
            "Cluster-robust (CR1), t with G−1 df",
            interval(tw["low"], tw["high"]),
            num(inf["cluster_robust_p"], 3),
        ],
        [
            f"Wild cluster bootstrap, {wb['reps']:,} Rademacher draws, null imposed",
            interval(wb["low"], wb["high"]),
            num(wb["p"], 3),
        ],
        [
            f"Randomization inference, {ri['permutations']:,} reassignments of adoption dates",
            "–",
            num(ri["p"], 3),
        ],
    ]
    c = r["castle"]
    treated = sum(c["cohorts"].values())
    aside = (
        f"With {c['units']} clusters and only {treated} treated, conventional cluster-robust standard errors are "
        "known to over-reject. The wild cluster restricted bootstrap and a permutation test "
        "over adoption dates ask the same question with fewer assumptions."
    )
    return section(
        "inference",
        "How sure is the TWFE estimate?",
        aside,
        table(["Method", "95% interval", "p-value"], rows, [False, True, True]),
    )


def sensitivity_section(r: dict[str, Any]) -> str:
    rows = [
        [
            s["label"],
            num(s["estimate"]),
            log_pct(s["estimate"]),
            interval(s.get("low"), s.get("high")),
        ]
        for s in r["sensitivity"]
    ]
    aside = (
        "Weighting by population makes the result about the average resident; unweighted, it is "
        "about the average state. The gap shows the effect is concentrated in large states, and "
        "that the choice of comparison group matters as much as the choice of estimator."
    )
    return section(
        "sensitivity",
        "Specification choices",
        aside,
        table(
            ["Specification", "Log points", "Change", "95% interval"],
            rows,
            [False, True, True, True],
        ),
    )


def synth_section(r: dict[str, Any]) -> str:
    p = r["prop99"]
    body = legend([("California", "s1"), ("Synthetic California", "dash")])
    body += f'<figure class="chart">{synth_path_chart(p["path"], 1989)}</figure>'
    pl = p.get("placebo")
    if pl:
        body += legend(
            [
                ("California", "s1"),
                ("Placebo states (pre-period fit within 5× California's)", "ghostkey"),
            ]
        )
        years = [row["year"] for row in p["path"]]
        body += (
            f'<figure class="chart">{placebo_chart(years, pl["gaps"], "California", 1989)}</figure>'
        )
    body += table(
        ["Donor state", "Weight"],
        [[w["donor"], num(w["weight"], 3)] for w in p["weights"]],
        [False, True],
    )
    body += details(
        "Show predictor balance",
        table(
            ["Predictor", "California", "Synthetic", "Donor mean"],
            [
                [
                    x["predictor"],
                    num(x["treated"], 3),
                    num(x["synthetic"], 3),
                    num(x["donor mean"], 3),
                ]
                for x in p["predictors"]
            ],
            [False, True, True, True],
        ),
    )
    p_text = (
        f" California's post/pre fit ratio ranks first of {pl['units']} states (p = {num(pl['p_value'], 3)})."
        if pl
        else ""
    )
    aside = (
        "In 1989 California raised its cigarette tax by 25 cents a pack. No single state is a good "
        "comparison, so a weighted blend of five is built to match California before the law. "
        f"By 2000 sales were {num(abs(p['gap_last']), 1)} packs per person lower than the blend.{p_text}"
    )
    return section("synth", "Synthetic control: Proposition 99", aside, body)


def validation_section(r: dict[str, Any]) -> str:
    rows = []
    for v in r["validation"]:
        mark = (
            Raw('<span class="pass">✓ reproduced</span>')
            if v["passed"]
            else Raw('<span class="fail">✗ differs</span>')
        )
        rows.append([v["study"], v["quantity"], num(v["published"], 4), num(v["ours"], 4), mark])
    passed = sum(v["passed"] for v in r["validation"])
    aside = (
        f"{passed} of {len(r['validation'])} published values reproduced, within the precision "
        "they were printed to. Sources: the did R package's example output (Callaway and "
        "Sant'Anna), Cunningham's Mixtape (Goodman-Bacon on castle), and Abadie, Diamond and "
        "Hainmueller (2010)."
    )
    body = table(
        ["Study", "Quantity", "Published", "peffects", ""],
        rows[:8],
        [False, False, True, True, False],
    )
    body += details(
        f"Show all {len(rows)} checks",
        table(
            ["Study", "Quantity", "Published", "peffects", ""],
            rows,
            [False, False, True, True, False],
        ),
    )
    return section("validation", "Checked against the literature", aside, body)


def method_section() -> str:
    items = [
        "TWFE: weighted least squares with unit and year effects absorbed by alternating projections; CR1 cluster-robust errors with the reghdfe/fixest small-sample correction.",
        "Goodman-Bacon: closed-form weights for every timing-group pair, including always-treated units; the weighted sum reproduces TWFE to machine precision.",
        "Callaway–Sant'Anna: ATT(g, t) against never- or not-yet-treated units, optional outcome-regression adjustment, influence-function errors, multiplier-bootstrap uniform bands, and simple, dynamic, cohort and calendar summaries.",
        "Sun–Abraham: cohort × relative-period saturated regression, aggregated with cohort-share weights; on a balanced panel it equals Callaway–Sant'Anna with a fixed base period, which the tests check.",
        "Synthetic control: simplex-constrained weights by non-negative least squares, predictor importance chosen by nested optimization of pre-period fit, and in-space placebo tests.",
        "Wild cluster restricted bootstrap: vectorized over draws, exact enumeration of all sign patterns for up to 12 clusters, confidence intervals by test inversion.",
    ]
    body = "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"
    return section(
        "method",
        "Method",
        "Every estimator is implemented from its paper with NumPy and SciPy.",
        body,
    )


def page(results: dict[str, Any], window: tuple[int, int] = (-5, 4)) -> str:
    c, p = results["castle"], results["prop99"]
    passed = sum(v["passed"] for v in results["validation"])
    effect = c["callaway_santanna"]["simple"]["estimate"]
    wb = c["inference"]["wild_bootstrap"]
    stats = "".join(
        [
            stat(
                "Published values reproduced",
                f"{passed}/{len(results['validation'])}",
                "three replication studies",
            ),
            stat("Castle-doctrine effect", log_pct(effect), "homicides, Callaway–Sant'Anna"),
            stat("Wild bootstrap p", num(wb["p"], 3), f"{wb['clusters']} state clusters"),
            stat(
                "Proposition 99, 2000", f"{num(p['gap_last'], 1)}", "packs per capita vs synthetic"
            ),
        ]
    )
    sections = "".join(
        [
            castle_section(results, window),
            bacon_section(results),
            inference_section(results),
            sensitivity_section(results),
            synth_section(results),
            validation_section(results),
            method_section(),
        ]
    )
    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500'
        '&family=JetBrains+Mono:wght@400;500&display=swap">'
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Policy Effects</title>
<meta name="description" content="Modern difference-in-differences and synthetic control, implemented from the papers and checked against published results.">
{fonts}<style>{CSS}</style></head>
<body><main>
<header><div class="top"><span class="meta">peffects</span><button id="theme" type="button">Dark</button></div>
<h1>Five ways to estimate a policy's effect, and where they disagree</h1>
<p class="muted">Two-way fixed effects, the Goodman-Bacon decomposition, Callaway–Sant'Anna, Sun–Abraham and synthetic control, each written from its paper, applied to stand-your-ground laws and California's tobacco tax, and checked against the published numbers.</p>
<div class="stats">{stats}</div></header>
{sections}
<footer class="muted"><p>Data: Cheng and Hoekstra (2013) via Cunningham's Mixtape; mpdta from the did R package; Proposition 99 data from Abadie, Diamond and Hainmueller (2010). Generated {esc(results["generated"])}. Full results in <a href="results.json">results.json</a>.</p></footer>
</main><div id="tip" role="tooltip" hidden></div>
<script>{SCRIPT}</script></body></html>
"""
