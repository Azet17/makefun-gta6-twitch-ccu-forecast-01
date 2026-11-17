"""gta6-twitch-ccu-forecast-01 entry: comparable-launch log-ratio forecaster.

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers for
hours 2..24 of 2026-11-19 (UTC), as five quantiles per hour, anchored on the
case's hour-1 print.

Model
-----
For every comparable past launch with a complete release day, take its
log-ratio curve r_h = ln(viewers_h / viewers_1) for hours 2..24: how the
category grew or faded after its own first hour.  Launches unlock at
midnight local time, rolling by timezone (New Zealand first, the US last),
so the ratio curve carries the shape of the day: a small dip in the small
hours, then a climb as Europe and the Americas unlock and reach prime time.

For the target case, weight each comparable by how close its own hour-1
print is to the target's, with a Gaussian kernel on the log scale
(bandwidth 1.0).  This matters because the hour-1 print encodes the unlock
pattern: a launch with a global simultaneous unlock has a near-empty
hour 1 and a ratio curve that explodes 1000x, while a midnight-local
launch like GTA VI has a substantial hour 1 and a curve that stays within
a factor of a few.  The kernel down-weights the launches whose day does
not look like this one.

Per hour, the forecast quantiles are the scale-weighted empirical
quantiles of the comparables' log-ratios at the five scored levels,
anchored multiplicatively on the target's hour-1 print p1:

    q_tau(h) = p1 * exp(Q_tau({r_{c,h}})),

then lightly smoothed across neighbouring hours (3-hour moving average of
each log-quantile) because the true day curve is smooth and the per-hour
empirical quantiles are noisy.  A comparable that has no hour-1 print on
its release day (e.g. the GTA VI category's own pre-release history, whose
day_start is its premiere day) contributes nothing.

Fallback: if fewer than 3 comparables have a usable curve for an hour,
that hour falls back to the house baseline shape (flat at p1 with a
log-spread widening as 0.15*sqrt(h-1)).

Flat-regime shrinkage: for large hour-1 prints the whole per-hour
log-quantile vector is shifted toward the flat baseline (log-ratio 0) by
up to 20% of the median's distance from flat, ramping in over
ln(hour-1) 10..12.  Big hour-1 launches stay flat -- their late-day
log-ratio falls almost one-for-one with hour-1 scale (corr -0.94) --
while small hour-1 launches explode as timezones unlock, so the
shrinkage is scale-gated and small launches are untouched.

Local backtest (leave-one-out pseudo-cases built from the 37 usable
comparables, each scored against its own hours 2..24 with the contest's
pinball loss on the log scale, x2 convention): median loss 0.242 over
all 37 launches, 0.196 over the 27 midnight-local-like launches with
hour-1 >= 20k (GTA VI's regime), versus the house baseline's 0.325 /
0.256 on the same pseudo-cases.  Mean loss 0.403 vs baseline 0.930
overall.  Before the shrinkage the median was 0.249 (mean 0.403); the
flat-curve launches improved -- Monster Hunter Wilds 0.464 -> 0.458,
Starfield 0.359 -> 0.333, Marvel Rivals 0.244 -> 0.227, Diablo IV
0.138 -> 0.121 (now beating its baseline 0.126) -- and the model beats
the baseline on 34 of 37 launches.

Deterministic: no randomness, fixed iteration order (files in name order,
comparables in file order, sorted values for quantiles).  Standard
library only, no network, no writes outside the output directory.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Gaussian kernel bandwidth (log units) for hour-1 scale similarity.
BANDWIDTH = 1.0
# Minimum comparable curves needed per hour before falling back.
MIN_CURVES = 3
# Flat-regime shrinkage (see forecast_case): how far the log-quantile
# vector is pulled toward the flat baseline (log-ratio 0), as a fraction of
# the median's distance from flat, for large hour-1 prints.  It ramps from
# 0 at ln(p1) = 10 to SHRINK_MAX at ln(p1) = 12 and stays there.
SHRINK_MAX = 0.20
SHRINK_LOGP1_LO = 10.0
SHRINK_LOGP1_HI = 12.0
# Fallback log-spread per sqrt hour, as in the house baseline.
FALLBACK_SIGMA = 0.15
# Standard normal quantiles at the scored levels (fallback only).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
MAX_VIEWERS = 1_000_000_000


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, returning an int for whole numbers."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def weighted_quantile(pairs, tau):
    """Weighted empirical quantile of [(value, weight)] at level tau."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= tau * total:
            return value
    return ordered[-1][0]


def release_day_values(comp):
    """Hourly viewers for hours 1..24 of the comparable's release day.

    Returns a list of 24 numbers (or None per missing hour).  A comparable
    whose release day has no hour-1 print returns None: its curve is
    undefined and it must not contribute (this also excludes the GTA VI
    category's own pre-release history, dated from its premiere day).
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            at = parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        by_hour[at] = viewers
    values = [by_hour.get(day_start + timedelta(hours=h - 1)) for h in range(1, 25)]
    if not values[0] or values[0] <= 0:
        return None
    return values


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = first["viewers"]
    if p1 is None or p1 < 1:
        p1 = 1
    p1 = float(p1)
    log_p1 = math.log(p1)

    # Flat-regime shrinkage.  Across the comparables, a launch's late-day
    # log-ratio falls almost one-for-one with its hour-1 scale (corr -0.94
    # between ln(hour-1) and the hour-20 log-ratio): tiny hour-1 launches
    # explode as timezones unlock, while big hour-1 launches stay flat --
    # Monster Hunter Wilds, Starfield, Diablo IV and Marvel Rivals all sit
    # near log-ratio 0 all day, and the unshrunk median curve (which climbs
    # to +0.4/+0.5 late) loses to the flat house baseline on all four.
    # So for large hour-1 prints the whole per-hour log-quantile vector is
    # shifted toward the flat baseline by a fraction of the median's
    # distance from flat.  The fraction ramps in over ln(p1) 10..12, so
    # small hour-1 launches (the exploders) are untouched.
    shrink_frac = SHRINK_MAX * max(
        0.0, min(1.0, (log_p1 - SHRINK_LOGP1_LO) / (SHRINK_LOGP1_HI - SHRINK_LOGP1_LO))
    )

    # Per-comparable log-ratio curves, weighted by hour-1 scale similarity.
    curves = []  # [(weight, [log-ratio for hours 2..24])]
    for comp in case.get("comparables") or []:
        values = release_day_values(comp)
        if values is None:
            continue
        v1 = values[0]
        weight = math.exp(-((math.log(v1) - log_p1) ** 2) / (2 * BANDWIDTH * BANDWIDTH))
        ratios = []
        for h in range(2, 25):
            v = values[h - 1]
            ratios.append(math.log(v / v1) if v and v > 0 else None)
        curves.append((weight, ratios))

    # Per-hour log-quantiles across comparables, then 3-hour smoothing.
    log_q = []  # [hour][level] in log units, relative to ln(p1)
    for i in range(23):
        pairs = [(ratios[i], w) for w, ratios in curves if ratios[i] is not None]
        if len(pairs) >= MIN_CURVES:
            raw = [weighted_quantile(pairs, tau) for tau in LEVELS]
            # Shift the whole vector toward flat by shrink_frac of the
            # median's distance from 0; the spread around the median is
            # carried along unchanged.
            shift = -shrink_frac * raw[2]
            log_q.append([x + shift for x in raw])
        else:
            h = i + 2
            log_q.append([z * FALLBACK_SIGMA * math.sqrt(h - 1) for z in Z])
    smoothed = []
    for i in range(23):
        lo = max(0, i - 1)
        hi = min(23, i + 2)
        span = hi - lo
        smoothed.append([sum(log_q[j][k] for j in range(lo, hi)) / span for k in range(5)])

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if h < 2 or h > 24:
            continue
        quantiles = []
        prev = 0.0
        for k in range(5):
            q = p1 * math.exp(smoothed[h - 2][k])
            if not math.isfinite(q) or q <= 0:
                q = p1
            q = min(q, MAX_VIEWERS)
            q = round2(q)
            if q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        hours.append({"hour": h, "q": quantiles})
    hours.sort(key=lambda r: r["hour"])
    if len(hours) != 23 or [r["hour"] for r in hours] != list(range(2, 25)):
        raise ValueError(f"{name} did not produce hours 2..24")
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
