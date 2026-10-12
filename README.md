# peffects

Modern difference-in-differences and synthetic control, written from the papers in NumPy
and SciPy and checked against published results. It covers:

- **Two-way fixed effects**: static and event-study regressions, with cluster-robust
  errors.
- **Goodman-Bacon decomposition**: the TWFE coefficient as a weighted average of every 2×2
  comparison.
- **Callaway–Sant'Anna**: ATT(g, t) with simple, dynamic, cohort and calendar summaries,
  plus uniform bands.
- **Sun–Abraham**: the interaction-weighted event study.
- **Synthetic control**: nested predictor weights and in-space placebo tests.
- **Wild cluster restricted bootstrap and randomization inference**: inference for when
  there are few treated clusters.

The [report](docs/index.html) applies all of them to stand-your-ground laws and
California's 1989 tobacco tax.

## Checked against the literature

`peffects validate` reproduces **37 of 37** published values, each within the precision it
was printed to:

| Source | What is reproduced |
|---|---|
| `did` R package example (Callaway and Sant'Anna 2021), mpdta | all 12 ATT(g, t) and the simple, dynamic, group and calendar aggregates, to 4 decimals |
| Cunningham, *The Mixtape*, `bacondecomp` on castle | unweighted TWFE (0.069) and the weight and estimate of each comparison type |
| Abadie, Diamond and Hainmueller (2010), Proposition 99 | donor weights (Utah .333, Nevada .237, Montana .200, Colorado .156, Connecticut .066), synthetic predictor balance, the ~26-pack gap in 2000, placebo p = 1/39 |

Beyond the published numbers, the tests check the following properties:

- TWFE coefficients and clustered SEs match pyfixest to machine precision.
- The Bacon weights sum to one and reproduce TWFE exactly, including panels with
  always-treated units.
- Sun–Abraham equals Callaway–Sant'Anna with a fixed base period on balanced panels.
- The vectorized wild bootstrap matches brute-force re-estimation over all 2^G sign
  patterns.
- Callaway–Sant'Anna recovers simulated effects exactly, and its intervals reach nominal
  coverage.

## Findings

**Castle-doctrine laws** (21 states, 2005–2009; homicides per 100,000 in logs; weighted by
population):

| Estimator | Effect | 95% interval |
|---|---|---|
| TWFE | +7.8% | [0.009, 0.142] log points |
| Callaway–Sant'Anna, overall | +8.2% | [−0.022, 0.179] |
| Sun–Abraham, overall | +8.2% | [0.022, 0.135] |

- **The estimators agree on the size of the effect.** The Bacon decomposition shows why
  TWFE is not badly biased here: 90% of its weight comes from clean comparisons against
  never-treated states, and only 2.4% from comparisons that use already-treated states as
  controls.
- **The uncertainty is larger than the textbook TWFE standard error suggests.**
  - The cluster-robust p-value is 0.027.
  - The wild cluster bootstrap gives p = 0.055 (interval [−0.002, 0.150]).
  - Permuting adoption dates gives p = 0.045.
  - Callaway–Sant'Anna's influence-function interval includes zero.
  - Sun–Abraham's narrower interval treats cohort composition as fixed.
- **The result depends on population weighting.** Unweighted, Callaway–Sant'Anna falls to
  +2.0% [−0.056, 0.095]. The effect is concentrated in large states.

**Proposition 99.** Synthetic California tracks the real state to within 1.8 packs per
capita (RMSPE) before 1989. Afterwards actual sales fall below it, by 19 packs per person
per year on average and by 25.7 packs in 2000. Of the 39 states, California has the
largest post-to-pre fit ratio (p = 0.026).

## Usage

```bash
pip install -e ".[dev]"
peffects fetch                 # castle, mpdta and Proposition 99 data, pinned by SHA-256
peffects validate              # reproduce the published numbers (--quick skips placebos)
peffects did castle            # every staggered estimator with three kinds of inference
peffects did mpdta --control notyet
peffects synth                 # synthetic California with placebo test
peffects report --out docs     # the HTML report and results.json
```

Run everything on your own long panel. Treatment is staggered and absorbing; give either
the adoption period or a 0/1 indicator:

```bash
peffects csv panel.csv --unit county --time year --outcome y --cohort first_treated \
    --weight population --control notyet --json results.json
```

The same estimators are available from Python:

```python
from peffects import cs, bacon, inference
from peffects.panel import Panel

panel = Panel.from_frame(df, unit="state", time="year", outcome="y", treat="post")
gt = cs.att_gt(panel, control="notyet")
event_study = cs.aggregate(gt, "dynamic").table
weights = bacon.decompose(panel).summary()
p = inference.wild_cluster_bootstrap(panel).p_value
```

## Design notes

- **Fixed effects** are absorbed by alternating projections, so TWFE and Sun–Abraham never
  build dummy matrices. Small-sample corrections follow reghdfe and fixest: fixed effects
  nested in clusters are not counted.
- **Callaway–Sant'Anna** builds each ATT(g, t) from an outcome-regression DiD on the
  comparison units, together with its influence function. The aggregations carry the
  estimation error of the cohort shares, as the reference implementation does. Uniform
  bands use a Mammen multiplier bootstrap of the same influence functions.
- **The wild bootstrap** is vectorized over draws. Bootstrap residuals are re-projected off
  the fixed effects, because flipping whole clusters' signs does not preserve period means.
  Without that step the bootstrap is wrong whenever period effects are absorbed.
- **Synthetic control weights** come from non-negative least squares with the adding-up
  constraint as a penalty row. Predictor importance is chosen by Nelder–Mead from several
  starts, including Synth's regression-based one.

## Limitations

- Treatment must be binary, staggered and absorbing. Designs where treatment switches on
  and off (de Chaisemartin–D'Haultfœuille) are not covered.
- Callaway–Sant'Anna with covariates uses outcome regression, not the doubly robust
  estimator, so covariate-adjusted results will differ from `did`'s default.
- Influence-function and cluster-robust errors are unreliable for cohorts of one or two
  units, like Florida's in castle. The report trims event times that rest on so few states.

## Development

```bash
pip install -e ".[dev,crosscheck]"    # crosscheck adds pyfixest for the agreement tests
ruff check src tests && ruff format --check src tests && mypy src && pytest --cov=peffects
```

Code MIT. Data belong to their authors: Cheng and Hoekstra (2013) via Cunningham's
*Mixtape*, the `did` package's mpdta, and Abadie, Diamond and Hainmueller (2010).
