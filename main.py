#!/usr/bin/env python3
"""Refresh-aware forecaster for GTA VI trophy earn rates (Exophase, PS5).

An earn rate is a ratio -- earners over tracked players -- and the two sides
of that ratio move on different clocks after a launch: earners saturate as
the keen cohort finishes content, while the tracked-player denominator keeps
growing for days and mechanically dilutes every rate.  This forecaster fits
both views of the series (the rate directly, and earners/players separately)
in log time, blends them, and prices its own uncertainty from pooled fit
residuals, the forecast horizon, and the disagreement between the two views.

Histories come from a page that refreshes on its own clock: repeated reads
of an unchanged page are one observation, so the fits run over distinct
page states only.  When a case's history carries almost no information
(a handful of states, or a player count that never moved -- the launch
night regime), the forecaster falls back to a dilution-aware stale-page
model instead of pretending a curve was measured.

Stdlib only, deterministic, no network.  Reads every *.in case in the
input directory and writes one .out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)

# --- main-path design constants (see build notes) -------------------------
WINDOW_H = 48.0        # fit window: last 48 h of distinct states
HALF_LIFE_H = 16.0     # recency weight half-life inside the window
SHRINK_K = 2.0         # slope shrink factor n / (n + SHRINK_K)
TIER_BLEND = 0.25      # slope blend toward the tier mean slope
DECOMP_BLEND = 0.5     # logit blend: direct vs earners/players split
GLOBAL_BLEND = 0.3     # common-move correction weight
DAMP_K = 0.3           # extrapolation damping in log time
SIGMA_HORIZON = 0.20   # sigma growth per sqrt(horizon day)
SIGMA_DISAGREE = 0.6   # sigma per unit of direct/decomp disagreement
SIGMA_MOVE = 0.25      # sigma per unit of own predicted move
SIGMA_FLOOR = 0.12
SIGMA_CAP = 1.8
SHORT_SPAN_DAYS = 2.0  # histories shorter than this earn extra width
SHORT_SPAN_BOOST = 0.35
BASIS_MIX = 0.5
DAMP_K_T = 29.0

# --- stale-page path -------------------------------------------------------
STALE_MULT = 0.74      # flat dilution multiplier on the last rate
STALE_SIGMA = 0.62
STALE_SIGMA_HORIZON = 0.10


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.  Returns (a, b, resid_sd, n)."""
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, n
    b = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return a, b, math.sqrt(max(var, 0.0)), n


def damped_step(du):
    """Log-time extrapolation step, damped so long horizons do not run away."""
    if du <= 0:
        return 0.0
    return du * du / (du + DAMP_K)


def distinct_states(history):
    """Collapse consecutive reads of an unchanged page into one state."""
    states = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(
            -1.0 if r is None else r for r in read.get("rates", [])))
        if key == prev_key:
            continue
        prev_key = key
        states.append(read)
    return states


def quantile_row(median_logit, sigma, sigma_up=None):
    if sigma_up is None:
        sigma_up = sigma
    qs = []
    for z in Z:
        s = sigma_up if z > 0 else sigma
        qs.append(round(clamp_rate(inv_logit(median_logit + z * s)), 4))
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def persistence_row(rate, horizon_days):
    spread = 0.5 * math.sqrt(max(horizon_days, 0.0))
    lg = logit(rate if rate is not None else 0.0)
    return [round(clamp_rate(inv_logit(lg + z * spread)), 4) for z in Z]


def stale_multiplier(rate):
    return STALE_MULT


def stale_row(rate, horizon_days):
    """Launch-night regime: the page has not taught us a curve yet, so the
    forecast is the last rate moved by the refresh the page is about to get:
    a wave of newly tracked players dilutes most rates, rare trophies can
    hold or rise because the same wave also brings their earners."""
    if rate is None:
        return [0.01, 0.02, 0.05, 0.10, 0.20]
    if rate >= 99.9:
        return quantile_row(logit(99.9), 0.35)
    med = clamp_rate(rate * stale_multiplier(rate))
    sigma = STALE_SIGMA + STALE_SIGMA_HORIZON * math.sqrt(
        max(horizon_days, 0.0))
    return quantile_row(logit(med), sigma)


def saturation_gate(states, state_h, players):
    """How close is the tracked-player pool to saturating?  Two independent
    signals, both computed on clock time: (1) a logistic fit to the players
    series -- the fraction p_last / K of the fitted carrying capacity;
    (2) the decay of the pool's log-growth slope between the halves of the
    observed history.  Returns 0 (pure exponential phase: the log-time
    machinery is right) to 1 (pool bending over: earners compound in clock
    time and the clock-time basis must be heard)."""
    pts = [(h, p) for h, p in zip(state_h, players) if p and p > 0]
    if len(pts) < 4:
        return 0.0
    p_last = pts[-1][1]
    # exponential candidate: line on ln p vs t
    ea, eb, _r, _n = wls_line([h for h, _ in pts],
                              [math.log(p) for _, p in pts],
                              [1.0] * len(pts))
    sse_exp = sum((math.log(p) - (ea + eb * h)) ** 2 for h, p in pts)
    best = None
    for mult in (1.15, 1.3, 1.5, 1.8, 2.2, 2.7, 3.3, 4.0, 5.0, 6.5, 8.0,
                 10.0, 13.0, 17.0, 22.0, 30.0, 40.0, 60.0, 100.0):
        cap = p_last * mult
        if any(p >= cap for _, p in pts):
            continue
        ys = [math.log(p / (cap - p)) for _, p in pts]
        la, lb, _r, _n = wls_line([h for h, _ in pts], ys, [1.0] * len(pts))
        sse = 0.0
        for (h, p), y in zip(pts, ys):
            pred = la + lb * h
            sse += (y - pred) ** 2
        if best is None or sse < best[0]:
            best = (sse, cap)
    if best is None or best[0] > 0.98 * sse_exp:
        return 0.0
    frac = p_last / best[1]
    w1 = min(1.0, max(0.0, (frac - 0.08) / 0.27))
    mid = pts[len(pts) // 2][0]
    early = [(h, p) for h, p in pts if h <= mid]
    late = [(h, p) for h, p in pts if h > mid]
    if len(early) < 2 or len(late) < 2:
        return 0.0
    _a, s1, _r, _n = wls_line([h for h, _ in early],
                              [math.log(p) for _, p in early],
                              [1.0] * len(early))
    _a, s2, _r, _n = wls_line([h for h, _ in late],
                              [math.log(p) for _, p in late],
                              [1.0] * len(late))
    if s1 <= 0:
        return 0.0
    ratio = s2 / s1
    w2 = min(1.0, max(0.0, (0.95 - ratio) / 0.13))
    return w1 * w2


def forecast_case(case):
    trophies = case["trophies"]
    history = sorted(case["history"], key=lambda r: r["slot"])
    states = distinct_states(history)
    horizon_days = case["horizon_minutes"] / 1440.0

    t0 = parse_ts(states[0]["slot"]) if states else parse_ts(case["at"])
    target = parse_ts(case["target"])

    def hours(ts):
        return (ts - t0).total_seconds() / 3600.0

    state_h = [hours(parse_ts(s["slot"])) for s in states]
    state_u = [math.log1p(max(h, 0.0)) for h in state_h]
    u_target = math.log1p(max(hours(target), 0.0))
    players = [s.get("players_tracked") or 0 for s in states]
    span_days = (state_h[-1] - state_h[0]) / 24.0 if len(state_h) > 1 else 0.0

    last_rate = []
    for i in range(len(trophies)):
        r = None
        for s in reversed(states):
            if s["rates"][i] is not None:
                r = s["rates"][i]
                break
        last_rate.append(r)

    stale = len(states) < 3 or len(set(players)) <= 1
    if stale or not states:
        return {"trophies": [
            {"id": t["id"], "q": stale_row(last_rate[i], horizon_days)}
            for i, t in enumerate(trophies)]}

    # ---- main path --------------------------------------------------------
    u_last = state_u[-1]
    h_last = state_h[-1]
    du = damped_step(u_target - u_last)
    win = [j for j in range(len(states)) if h_last - state_h[j] <= WINDOW_H]
    weights = [0.5 ** ((h_last - state_h[j]) / HALF_LIFE_H) for j in win]

    tiers = [t.get("tier") or "" for t in trophies]

    # Direct fits: logit(rate) on u.
    direct = []       # (pred_logit, slope_used, last_logit)
    resid_pool = []
    raw_slopes = []
    raw_slopes_t = []
    for i in range(len(trophies)):
        xs, ys, ws = [], [], []
        for k, j in enumerate(win):
            r = states[j]["rates"][i]
            if r is None:
                continue
            xs.append(state_u[j])
            ys.append(logit(r))
            ws.append(weights[k])
        a, b, rsd, n = wls_line(xs, ys, ws)
        resid_pool.extend(
            (y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        raw_slopes.append((b * n / (n + SHRINK_K)) if n else 0.0)
        at, bt, _rt, _nt = wls_line(
            [state_h[j] for k, j in enumerate(win)
             if states[j]["rates"][i] is not None],
            ys, ws)
        raw_slopes_t.append((bt * n / (n + SHRINK_K)) if n else 0.0)
        direct.append([a, b, n, xs, ys, at, bt])
    tier_mean = {}
    tier_mean_t = {}
    for tier in set(tiers):
        vals = [raw_slopes[i] for i in range(len(trophies))
                if tiers[i] == tier and direct[i][2] > 0]
        tier_mean[tier] = sum(vals) / len(vals) if vals else 0.0
        vals = [raw_slopes_t[i] for i in range(len(trophies))
                if tiers[i] == tier and direct[i][2] > 0]
        tier_mean_t[tier] = sum(vals) / len(vals) if vals else 0.0
    all_slopes = [raw_slopes[i] for i in range(len(trophies))
                  if direct[i][2] > 0]
    global_mean_slope = (sum(all_slopes) / len(all_slopes)
                         if all_slopes else 0.0)
    all_slopes_t = [raw_slopes_t[i] for i in range(len(trophies))
                    if direct[i][2] > 0]
    global_mean_slope_t = (sum(all_slopes_t) / len(all_slopes_t)
                           if all_slopes_t else 0.0)

    # Decomposition fits: ln(earners) and ln(players) on u.
    px, py, pw = [], [], []
    for k, j in enumerate(win):
        if players[j] > 0:
            px.append(state_u[j])
            py.append(math.log(players[j]))
            pw.append(weights[k])
    pa, pb, _pn, _pr = wls_line(px, py, pw)
    pb_s = pb * len(px) / (len(px) + SHRINK_K) if px else 0.0
    players_pred = pa + pb * u_last + pb_s * damped_step(u_target - u_last)

    # Global pace fit: ln(trophies_earned / players) on u.
    gx, gy, gw = [], [], []
    for k, j in enumerate(win):
        earned = states[j].get("trophies_earned") or 0
        if players[j] > 0 and earned > 0:
            gx.append(state_u[j])
            gy.append(math.log(earned / players[j]))
            gw.append(weights[k])
    ga, gb, _gn, _gr = wls_line(gx, gy, gw)
    global_move = gb * du if gx else 0.0

    medians = []
    sigmas = []
    sigma_res = math.sqrt(sum(resid_pool) / len(resid_pool)) \
        if resid_pool else 0.30
    du_t = (hours(target) - h_last)
    step_t = du_t * du_t / (du_t + DAMP_K_T) if du_t > 0 else 0.0
    mix = BASIS_MIX * saturation_gate(states, state_h, players)
    for i in range(len(trophies)):
        a, b, n, xs, ys, at, bt = direct[i]
        if n == 0 or last_rate[i] is None:
            medians.append(None)
            sigmas.append(None)
            continue
        b_use = (1.0 - TIER_BLEND) * raw_slopes[i] + TIER_BLEND * \
            tier_mean.get(tiers[i], global_mean_slope)
        y_last = a + b * u_last
        pred_u = y_last + b_use * du
        bt_use = (1.0 - TIER_BLEND) * raw_slopes_t[i] + TIER_BLEND * \
            tier_mean_t.get(tiers[i], global_mean_slope_t)
        pred_t = at + bt * h_last + bt_use * step_t
        pred_direct = (1.0 - mix) * pred_u + mix * pred_t

        ex, ey, ew = [], [], []
        for k, j in enumerate(win):
            r = states[j]["rates"][i]
            if r is None or r <= 0.0 or players[j] <= 0:
                continue
            ex.append(state_u[j])
            ey.append(math.log(max(r / 100.0 * players[j], 1e-9)))
            ew.append(weights[k])
        ea, eb, _en, _er = wls_line(ex, ey, ew)
        if ex:
            eb_s = eb * len(ex) / (len(ex) + SHRINK_K)
            ln_rate = (ea + eb * u_last + eb_s * du) - players_pred
            rate_dec = clamp_rate(100.0 * math.exp(ln_rate))
            pred_decomp = logit(rate_dec)
        else:
            pred_decomp = pred_direct
        med = (1.0 - DECOMP_BLEND) * pred_direct + \
            DECOMP_BLEND * pred_decomp
        medians.append([med, pred_direct, pred_decomp])
        move = med - logit(last_rate[i])
        sig = sigma_res + SIGMA_HORIZON * math.sqrt(horizon_days) \
            + SIGMA_DISAGREE * abs(pred_direct - pred_decomp) \
            + SIGMA_MOVE * abs(move)
        if span_days < SHORT_SPAN_DAYS:
            sig *= 1.0 + SHORT_SPAN_BOOST * (1.0 - span_days / SHORT_SPAN_DAYS)
        sigmas.append(min(SIGMA_CAP, max(SIGMA_FLOOR, sig)))

    moves = [medians[i][0] - logit(last_rate[i])
             for i in range(len(trophies)) if medians[i] is not None]
    mean_move = sum(moves) / len(moves) if moves else 0.0
    correction = GLOBAL_BLEND * (global_move - mean_move)

    rows = []
    for i, t in enumerate(trophies):
        if medians[i] is None:
            rows.append({"id": t["id"],
                         "q": persistence_row(last_rate[i], horizon_days)})
            continue
        med = medians[i][0] + correction
        rows.append({"id": t["id"], "q": quantile_row(med, sigmas[i])})
    return {"trophies": rows}


def forecast_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        case = json.load(fh)
    try:
        return forecast_case(case)
    except Exception:
        rows = []
        hist = case.get("history", [])
        days = case.get("horizon_minutes", 0) / 1440.0
        for i, t in enumerate(case.get("trophies", [])):
            r = None
            for read in reversed(hist):
                if read["rates"][i] is not None:
                    r = read["rates"][i]
                    break
            rows.append({"id": t["id"], "q": persistence_row(r, days)})
        return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        result = forecast_file(os.path.join(in_dir, name))
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
