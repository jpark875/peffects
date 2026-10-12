"""The three replication datasets: download once, verify, and load as panels."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from peffects.panel import Panel


class DataError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    name: str
    filename: str
    url: str
    sha256: str
    citation: str


SOURCES = {
    "castle": Source(
        "castle",
        "castle.dta",
        "https://raw.githubusercontent.com/scunning1975/mixtape/master/castle.dta",
        "804633c161827b6c0824f86f239046386d1a8266a866f83bf5ddb2aa762a5f29",
        "Cheng and Hoekstra (2013), via Cunningham, Causal Inference: The Mixtape",
    ),
    "mpdta": Source(
        "mpdta",
        "mpdta.rda",
        "https://raw.githubusercontent.com/bcallaway11/did/master/data/mpdta.rda",
        "0be3bc5eaaedaa50e047deaf45ad0e193e7210df1644103c63a6c1baa75965d4",
        "Callaway and Sant'Anna (2021), example data from the did R package",
    ),
    "smoking": Source(
        "smoking",
        "smoking.csv",
        "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/master/"
        "causal-inference-for-the-brave-and-true/data/smoking.csv",
        "bae31804ce53635822a4bda2b2622d92b707f8c35cf017ea2918a8fdb6083b89",
        "Abadie, Diamond and Hainmueller (2010), California's Proposition 99",
    ),
}

# The Proposition 99 file numbers the 39 states alphabetically.
SMOKING_STATES = (
    "Alabama",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Georgia",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Mexico",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    name: str,
    data_dir: Path,
    *,
    force: bool = False,
    opener: Callable[[str], bytes] | None = None,
) -> Path:
    """Download ``name`` into ``data_dir`` unless a verified copy is already there."""
    source = SOURCES[name]
    path = data_dir / source.filename
    if path.exists() and not force and sha256_of(path) == source.sha256:
        return path
    data_dir.mkdir(parents=True, exist_ok=True)

    def default(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            return bytes(response.read())

    payload = (opener or default)(source.url)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(payload)
    actual = sha256_of(partial)
    if actual != source.sha256:
        partial.unlink()
        raise DataError(f"{name}: checksum mismatch (got {actual[:12]}..., expected a pinned copy)")
    partial.replace(path)
    return path


def _path(name: str, data_dir: Path) -> Path:
    path = data_dir / SOURCES[name].filename
    if not path.exists():
        raise DataError(f"{path} not found; run `peffects fetch` first")
    return path


def castle_frame(data_dir: Path) -> pd.DataFrame:
    return pd.read_stata(_path("castle", data_dir))


def mpdta_frame(data_dir: Path) -> pd.DataFrame:
    import pyreadr

    result = pyreadr.read_r(str(_path("mpdta", data_dir)))
    frame = next(iter(result.values()))
    return pd.DataFrame(frame)


def smoking_frame(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(_path("smoking", data_dir))
    frame["state"] = frame["state"].map(lambda i: SMOKING_STATES[int(i) - 1])
    return frame


def castle(data_dir: Path, outcome: str = "l_homicide") -> Panel:
    """State-year panel 2000-2010. Treatment starts in the first full year a castle-doctrine
    (stand-your-ground) law was in force; weights are state population."""
    return Panel.from_frame(
        castle_frame(data_dir),
        unit="state",
        time="year",
        outcome=outcome,
        treat="post",
        weight="popwt",
    )


def mpdta(data_dir: Path) -> Panel:
    """County-year panel 2003-2007: log teen employment and minimum wage increases."""
    return Panel.from_frame(
        mpdta_frame(data_dir),
        unit="countyreal",
        time="year",
        outcome="lemp",
        cohort="first.treat",
        covariates=["lpop"],
    )
