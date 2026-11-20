"""gta6-trophy-rarity-forecast-01 entry: earners/players decomposition
forecaster with a blended log-time slope (whole-history + recent).

Model (fitted at RUNTIME from the case's own history, nothing hard-coded
from the practice answers):
  - Work in COUNTS, not rates: E_i(t) = rate_i(t) * N(t) / 100 earners,
    N(t) = players_tracked. A rate is E/N, so it moves when earners grow
    AND when new players join (dilution). Exophase only refreshes its
    counts every few hours, so consecutive reads are often identical:
    collapse history to distinct states (one per refresh).
  - For each trophy and for N, fit a log-log power law in log-time,
    ln y vs ln(1 + t), t = hours since the game's launch midnight
    (2026-11-19T00:00Z), anchored at the last distinct state:
        y_target = y_last * exp(b * (ln(1+t_target) - ln(1+t_last)))
    Two slope estimators, blended in logit space (see below):
      b_w: slope over ALL distinct states (stable, lags acceleration),
           capped at 1.2 — it averages the early, fast launch growth,
           and uncapped it over-extrapolates trophies that are already
           saturating (worth ~0.3 pinball loss on synthetic finals)
      b_r: slope over the LAST 4 distinct states (tracks acceleration
           as the player base's playtime crosses each trophy's
           threshold, but noisier), capped at 2.0
    Slopes are for earners and for players separately. The median is
    the logit-mean of the two forecasts (whole-slope and recent-slope).
    On synthetic 4.5-day backtests (cohort ground truth, 3 influx
    scenarios x 3 seeds) this blend scored mean pinball loss 0.286
    (range 0.266-0.302, coverage 1.00) vs 0.856 for the persistence
    starter, 0.611 for the whole-slope forecast alone, and 0.555
    (range 0.233-0.898) for the recent-slope forecast alone. The
    blend keeps the bias correction with the stability.
  - The trophy at 100% (every tracked player holds it) stays 100%.
  - Spread (logit): sigma = 0.8 * 0.5 * sqrt(d), d = horizon in days.
    The starter's 0.5*sqrt(d) already overcovers on synthetic finals
    (coverage 1.00); a refresh-variance term on top of it made the
    synthetic loss WORSE (0.81 vs 0.61), so no extra refresh term is
    added — the refresh effect lives in the median, via the distinct
    states. 0.8x is the simulator optimum (0.6x-0.8x tied) with a
    margin for real-world extra uncertainty.
  - Fallback (<3 distinct states, e.g. the practice cases whose
    histories are a single flat read): E grows as (1+t)^0.5 and N as
    (1+t)^0.8, i.e. the rate dilutes slowly. No trend can be fitted
    from one state; the practice cases cannot validate the final
    model and are not optimised for (their dev board is gamed by
    refresh inversion — see INTAKE.md).

Deterministic, standard library only, no network, no randomness.
Runs as: python3 main.py (MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
stand in for /input and /output outside the grader container).
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA_BASE = 0.5
SIGMA_SCALE = 0.8
LAUNCH = (2026, 11, 19)  # release midnight UTC; hours are measured from here


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg: float) -> float:
    if lg > 30:
        return 99.99
    if lg < -30:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def hours_since_launch(slot: str) -> float:
    """Hours from 2026-11-19T00:00:00Z to an ISO slot string."""
    import datetime
    y, mo, d = int(slot[0:4]), int(slot[5:7]), int(slot[8:10])
    hh, mm = int(slot[11:13]), int(slot[14:16])
    t0 = datetime.datetime(2026, 11, 19, tzinfo=datetime.timezone.utc)
    t = datetime.datetime(y, mo, d, hh, mm, tzinfo=datetime.timezone.utc)
    return max(0.0, (t - t0).total_seconds() / 3600.0)


def distinct_states(history):
    """Collapse history to one entry per distinct (players, rates) state:
    Exophase refreshes every few hours, so most reads repeat the last
    refresh exactly. Returns the distinct reads in order."""
    out = []
    prev = None
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            out.append(h)
            prev = key
    return out


def loglog_slope(ts, ys):
    """Least-squares slope of ln(y) on ln(1+t); y <= 0 points dropped.
    Returns 0.0 when fewer than 2 positive points."""
    pts = [(math.log1p(t), math.log(y)) for t, y in zip(ts, ys) if y > 0]
    if len(pts) < 2:
        return 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0:
        return 0.0
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def cap_slope(b: float, cap: float = 2.0) -> float:
    return min(cap, max(0.0, b))


def extrapolate(y_last: float, slope: float, t_last: float, t_tgt: float) -> float:
    return y_last * math.exp(slope * (math.log1p(t_tgt) - math.log1p(t_last)))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    last = history[-1]
    d_days = case["horizon_minutes"] / 1440.0
    t_last = hours_since_launch(last["slot"])
    t_tgt = hours_since_launch(case["target"])
    sigma = SIGMA_BASE * SIGMA_SCALE * math.sqrt(d_days)

    states = distinct_states(history)
    st_times = [hours_since_launch(h["slot"]) for h in states]
    n_hist = [float(h["players_tracked"]) for h in states]
    n_last = float(last["players_tracked"])

    # Players growth: whole-history and recent (last 4 states) slopes.
    # The whole-history slope is capped at 1.2: it averages the early,
    # fast launch growth, and uncapped it over-extrapolates trophies
    # that are already saturating (on synthetic finals the 1.2 cap was
    # worth ~0.3 pinball loss vs a 2.0 cap). The recent slope keeps
    # the 2.0 cap — it measures current growth, not the launch surge.
    b_n_w = cap_slope(loglog_slope(st_times, n_hist), 1.2)
    b_n_r = cap_slope(loglog_slope(st_times[-4:], n_hist[-4:]), 2.0)
    n_tgt_w = max(extrapolate(n_last, b_n_w, t_last, t_tgt), n_last)
    n_tgt_r = max(extrapolate(n_last, b_n_r, t_last, t_tgt), n_last)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        if last_rate >= 99.99:
            # Everybody tracked holds this trophy; it stays 100%.
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        earners = []
        for h in states:
            r = h["rates"][i]
            earners.append(0.0 if r is None else r / 100.0 * h["players_tracked"])
        if len(states) < 3:
            # One flat state: no trend is fittable. Dilute slowly:
            # earners (1+t)^0.5, players (1+t)^0.8.
            e_tgt = earners[-1] * ((1 + t_tgt) / max(1 + t_last, 1.0)) ** 0.5
            n_tgt = n_last * ((1 + t_tgt) / max(1 + t_last, 1.0)) ** 0.8
            med = 100.0 * e_tgt / n_tgt if n_tgt > 0 else last_rate
        else:
            b_w = cap_slope(loglog_slope(st_times, earners), 1.2)
            b_r = cap_slope(loglog_slope(st_times[-4:], earners[-4:]), 2.0)
            e_tgt_w = extrapolate(earners[-1], b_w, t_last, t_tgt)
            e_tgt_r = extrapolate(earners[-1], b_r, t_last, t_tgt)
            med_w = 100.0 * min(e_tgt_w, n_tgt_w) / n_tgt_w
            med_r = 100.0 * min(e_tgt_r, n_tgt_r) / n_tgt_r
            # Logit-mean of the two forecasts; if either saturates,
            # take the lower (conservative) median.
            if med_w >= 99.99 or med_r >= 99.99:
                med = min(med_w, med_r)
            else:
                med = inv_logit(0.5 * logit(med_w) + 0.5 * logit(med_r))
        med = clamp_pct(med)
        lg = logit(med)
        q = [round4(inv_logit(lg + z * sigma)) for z in Z]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
