# GTA VI Twitch CCU Forecaster — make.fun `gta6-twitch-ccu-forecast-01`

[![make.fun bounty](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![stdlib only](https://img.shields.io/badge/deps-stdlib%20only-green)](https://docs.python.org/3/library/)

Forecasts how many people watch **Grand Theft Auto VI** live on Twitch in
each hour of its release day, 2026-11-19 (UTC), after the first — five
quantiles per hour for hours 2–24, scored by pinball loss on the log scale.

Bounty: https://make.fun/bounty/gta6-twitch-ccu-forecast-01

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one
`/output/<id>.out` (else `$MAKEFUN_OUTPUT_DIR`) per case:
`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]} ...]}` for hours 2–24.

The model is a **comparable-launch log-ratio forecaster**:

1. For every comparable past launch with a complete release day, compute its
   log-ratio curve `r_h = ln(viewers_h / viewers_1)` for hours 2–24 — how the
   category grew or faded after its own first hour. Launches unlock at
   midnight local time, rolling by timezone, so the curve carries the shape
   of the day: a dip in the small hours, then a climb as Europe and the
   Americas unlock and reach prime time.
2. Weight each comparable by how close its own hour-1 print is to the
   target's, with a Gaussian kernel on the log scale (bandwidth 1.0). The
   hour-1 print encodes the unlock pattern: a global-simultaneous launch has
   a near-empty hour 1 and a ratio curve that explodes ~1000×, while a
   midnight-local launch like GTA VI has a substantial hour 1 and a curve
   within a factor of a few. The kernel down-weights launches whose day does
   not look like this one.
3. Per hour, the forecast quantiles are the scale-weighted empirical
   quantiles of the comparables' log-ratios at the five scored levels,
   anchored multiplicatively on the target's hour-1 print, then lightly
   smoothed across neighbouring hours (3-hour moving average of each
   log-quantile). A comparable with no hour-1 print on its release day
   (e.g. the GTA VI category's own pre-release history) contributes nothing.
4. If fewer than 3 comparables have a usable curve for an hour, that hour
   falls back to the house baseline shape (flat at hour 1, spread widening
   as `0.15·√(h−1)` in log space).

Deterministic: no randomness, fixed iteration order, byte-identical output
on repeat runs. Fast: ~0.07 s per case (limit 8 s).

## Backtest

Leave-one-out pseudo-cases built from the 37 usable comparables, each scored
against its own hours 2–24 with the contest's pinball loss on the log scale,
using the shipped `main.py` (the house starter scored the same way):

| set | this entry | house baseline |
| --- | --- | --- |
| all 37 launches, mean | **0.403** | 0.930 |
| all 37 launches, median | **0.249** | 0.325 |
| midnight-local-like (hour-1 ≥ 20k, n=27), mean | **0.257** | 0.321 |
| midnight-local-like (hour-1 ≥ 20k, n=27), median | **0.194** | 0.256 |

The entry beats the baseline on 33 of 37 launches. The house baseline scores
~0.387 on the four practice cases.

## Test locally

```sh
mkdir -p /tmp/gta6-out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=/tmp/gta6-out python3 main.py
```

## Files

- `main.py` — the entry (the only code file)
- `LICENSE` — MIT, created by `makefun init`
- `.makefun/owner.json` — payout wallet
