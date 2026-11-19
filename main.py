"""gta6-trophy-rarity-forecast-01 entry: a refresh-aware earn-rate forecaster.

The program reads one makefun/gta6-trophies-case/v1 case file per run and
writes, for every trophy in the case, five quantiles of its Exophase earn rate
(the percent of tracked players who hold it) at the case's target instant.

Why it is built the way it is
-----------------------------
An earn rate is a ratio: earners / players tracked. After a launch the two
sides move on different clocks. The denominator keeps growing for days as more
owners sync a profile, and it mainstreams as it grows; the numerator for any
one trophy saturates as the keen players earn it and the curve flattens toward
the share of the whole base that will ever reach it. So a trophy's rate bends:
it can climb while its earners outrun the incoming crowd, then fall as the
crowd dilutes it. Two failure modes wait on either side: extend the launch's
first steep climb in a straight line and you overshoot wildly; repeat the last
rate and you miss the motion entirely.

The model forecasts each trophy two ways and blends them, both in the basis
the earning process itself uses: log time, u = ln(1 + hours since the first
read). A trophy's earners follow a cumulative curve whose logit is nearly
linear in log time while it is being earned, so local slopes in this basis
stay meaningful for days, long after slopes in clock time have bent. Each
fit is a recency-weighted line in u, extrapolated with a light damping
factor du / (du + kappa): some of every slope is assumed to bend away over a
four-and-a-half-day horizon, but never most of it.

  * Direct: the trophy's logit rate fitted on u over the recent page
    states, its slope shrunk toward the tier mean (platinum, gold, silver
    and bronze trophies behave alike), extrapolated to the target.
  * Decomposition: the same fits applied separately to the trophy's earners
    (rate x players, which the case lets us recover) and to the tracked-
    player count; the forecast rate is the ratio at the target, anchored so
    the decomposition reproduces the last read. This is the side that sees
    dilution coming: when the denominator's growth outruns a trophy's
    earners, the ratio bends down even if the trophy's own slope points up,
    and when the pool fills up and stops diluting, it sees that too.

The two are averaged in logit space. A common correction nudges the
cross-trophy average move toward the move of the global ratio (total
trophies earned per tracked player), which the case reports directly and no
single trophy's noise can dominate. The quantile width pools the fits'
residuals, grows with the horizon and with the size of the move itself, and
widens by how far the two forecasts disagree, so the 90% band covers about
90%.

Everything is fitted on distinct page states only: Exophase refreshes a
game's page every few hours, so runs of identical reads are one observation,
not many. If the history holds fewer than three distinct states (a
launch-night cut, where the page has never refreshed), no fit is possible
and the model switches to a stale-page prior: a refresh is overdue, the
tracked crowd is still the small pre-launch one, and rates dilute toward the
refreshed population's lower holding rate.

Guards: every case is wrapped so an unexpected shape falls back to a
last-read forecast with a random-walk spread; the output is always the exact
schema the grader wants. No randomness: two runs are byte-identical.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

RATE_LO = 0.01
RATE_HI = 99.99

# Model constants (chosen on synthetic long-horizon cuts, not on the two
# practice cases: see the build notes in the repository's README).
WINDOW_H = 48.0        # local window for the fits, hours before the cut
WINDOW_MIN_STATES = 6  # ... or at least this many recent states
HALF_LIFE_H = 16.0     # recency half-life inside the window, in hours
KAPPA_U = 0.3          # damping scale for slope extrapolation in u
SLOPE_SHRINK_N = 2.0   # pseudo-count shrinking a slope toward its tier mean
TIER_BLEND = 0.25      # weight of the tier-mean slope after shrinkage
DECOMP_BLEND = 0.5     # weight of the decomposition forecast in the blend
GLOBAL_BLEND = 0.3     # weight of the global-ratio slope correction
SIGMA_FLOOR = 0.12     # smallest per-trophy logit spread
SIGMA_PER_SQRT_DAY = 0.20
SIGMA_DISAGREE = 0.6   # weight of the two forecasts' disagreement
SIGMA_MOVE = 0.25      # spread growth per unit of forecast logit move
SIGMA_CAP = 1.8


def clamp_rate(x):
    return min(RATE_HI, max(RATE_LO, x))


def logit_pct(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def ls_line_w(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, n_eff)."""
    sw = sum(ws)
    if sw <= 0:
        return ys[-1], 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    n_eff = sw * sw / sum(w * w for w in ws)
    if sxx <= 1e-12:
        return my, 0.0, n_eff
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return my - b * mx, b, n_eff


class Fit:
    """One series' local weighted line fit in u.

    move(damp) is the fitted change from the last point to the target with
    the slope damped by `damp`; slope and n_eff are what shrinkage and tier
    pooling trade in; resid feeds the spread.
    """

    def __init__(self, xs, ys, ws, u_last, u_target):
        self.u_last = u_last
        self.u_target = u_target
        a, b, n_eff = ls_line_w(xs, ys, ws)
        self.n_eff = n_eff
        self.slope = b
        self.resid = [y - (a + b * x) for x, y in zip(xs, ys)]

    def move(self, damp):
        return self.slope * (self.u_target - self.u_last) * damp


def distinct_states(case):
    """Collapse runs of identical reads into page states.

    Returns (times_hours, players, earned, rates_matrix), times in hours
    from the first read's slot.
    """
    history = case["history"]
    t0 = parse_time(history[0]["slot"])
    times, players, earned, rates = [], [], [], []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(
            -1.0 if r is None else r for r in read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        times.append((parse_time(read["slot"]) - t0) / 3600.0)
        players.append(float(read["players_tracked"]))
        earned.append(float(read["trophies_earned"]))
        rates.append(read["rates"])
    return times, players, earned, rates


def fallback_rows(case):
    """Last read plus a random-walk spread: the never-invalid safety net."""
    last = case["history"][-1] if case["history"] else None
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0) + 0.05
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = 0.5
        if last is not None and last["rates"][i] is not None:
            rate = last["rates"][i]
        lg = logit_pct(rate)
        rows.append({"id": trophy["id"],
                     "q": [round(inv_logit_pct(lg + z * spread), 4)
                           for z in Z]})
    return rows


def stale_page_rows(case, last_rates, sigma):
    """Forecast when the page has (almost) never refreshed in the history.

    The tracked crowd is still the small pre-launch one and at least one
    refresh lands before the target: the incoming crowd holds each trophy
    at a lower rate, so rates dilute. A rate already at 100 (an auto-granted
    trophy) is the exception: it stays.
    """
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i]
        if rate is None:
            rate = 0.5
        med = 99.9 if rate >= 99.99 else clamp_rate(rate * 0.74)
        lg = logit_pct(med)
        rows.append({"id": trophy["id"],
                     "q": [round(inv_logit_pct(lg + z * sigma), 4)
                           for z in Z]})
    return rows


def model_rows(case):
    times, players, earned, rates = distinct_states(case)
    n_trophies = len(case["trophies"])
    t0 = parse_time(case["history"][0]["slot"])
    at_hours = (parse_time(case["at"]) - t0) / 3600.0
    horizon_hours = case["horizon_minutes"] / 60.0
    target_hours = at_hours + horizon_hours
    us = [math.log1p(t) for t in times]
    u_last = us[-1]
    u_target = math.log1p(target_hours)
    du = max(0.0, u_target - u_last)
    damp = du / (du + KAPPA_U)
    horizon_days = horizon_hours / 24.0
    last_rates = rates[-1]

    # Stale-page regime: too few distinct states to fit anything.
    if len(times) < 3 or max(players) - min(players) <= 0:
        sigma = 0.62 + 0.10 * math.sqrt(horizon_days)
        return stale_page_rows(case, last_rates, sigma)

    # Local window of states.
    start = len(times)
    for j in range(len(times)):
        if times[-1] - times[j] <= WINDOW_H:
            start = j
            break
    start = min(start, max(0, len(times) - WINDOW_MIN_STATES))
    win = list(range(start, len(times)))
    weights = [0.5 ** ((times[-1] - times[j]) / HALF_LIFE_H) for j in win]
    wus = [us[j] for j in win]

    def fit(get_y):
        pts = [(wus[k], get_y(j), weights[k])
               for k, j in enumerate(win) if get_y(j) is not None]
        if len(pts) < 3:
            return None
        return Fit([p[0] for p in pts], [p[1] for p in pts],
                   [p[2] for p in pts], u_last, u_target)

    tiers = [t["tier"] for t in case["trophies"]]

    # Players: their own fit, extrapolated by its move.
    p_fit = fit(lambda j: math.log(max(players[j], 1.0)))
    ln_p_last = math.log(max(players[-1], 1.0))
    if p_fit is not None:
        p_target = math.exp(ln_p_last + p_fit.move(damp))
    else:
        p_target = math.exp(ln_p_last)

    # Direct per-trophy fits; tier means of the shrunk slopes first.
    direct_fits = []
    resid_sq = []
    for i in range(n_trophies):
        f = fit(lambda j, i=i: (logit_pct(rates[j][i])
                                if rates[j][i] is not None else None))
        direct_fits.append(f)
        if f is not None:
            resid_sq.extend(r * r for r in f.resid)

    def shrunk(f):
        return f.slope * f.n_eff / (f.n_eff + SLOPE_SHRINK_N)

    tier_mean = {}
    for tier in set(tiers):
        vals = [shrunk(direct_fits[i]) for i in range(n_trophies)
                if tiers[i] == tier and direct_fits[i] is not None]
        tier_mean[tier] = sum(vals) / len(vals) if vals else 0.0
    all_vals = [shrunk(f) for f in direct_fits if f is not None]
    global_mean = sum(all_vals) / len(all_vals) if all_vals else 0.0

    # Global ratio slope (log trophies earned per tracked player).
    g_fit = fit(lambda j: math.log(max(earned[j], 1.0)
                                   / max(players[j], 1.0)))
    g_move = g_fit.move(damp) if g_fit is not None else 0.0

    move_list = []
    trophy_data = []
    for i in range(n_trophies):
        rate_last = last_rates[i]
        if rate_last is None:
            rate_last = 0.5
        lg_last = logit_pct(rate_last)
        # Direct forecast: the fit's move, with its slope first shrunk and
        # tier-blended (the blend trades on slopes; the curvature part of
        # the move is the fit's own and is kept whole).
        f = direct_fits[i]
        if f is not None:
            slope_adj = ((1.0 - TIER_BLEND) * shrunk(f)
                         + TIER_BLEND * tier_mean.get(tiers[i], global_mean))
            scale = slope_adj / f.slope if abs(f.slope) > 1e-9 else 0.0
            lg_direct = lg_last + f.move(damp) * scale
        else:
            lg_direct = lg_last + tier_mean.get(
                tiers[i], global_mean) * du * damp
        # Decomposition forecast: earners and players fitted separately.
        e_fit = fit(lambda j, i=i: (
            math.log(max(rates[j][i], 0.0) / 100.0 * players[j])
            if rates[j][i] is not None and rates[j][i] > 0 else None))
        lg_decomp = None
        if e_fit is not None and rate_last > 0:
            ln_e_last = math.log(rate_last / 100.0) + ln_p_last
            ln_e_target = ln_e_last + e_fit.move(damp)
            rate_dec = 100.0 * math.exp(ln_e_target) / p_target
            lg_decomp = logit_pct(rate_dec)
        if lg_decomp is not None:
            lg = (1.0 - DECOMP_BLEND) * lg_direct + DECOMP_BLEND * lg_decomp
            disagree = abs(lg_direct - lg_decomp)
        else:
            lg = lg_direct
            disagree = 0.3
        move_list.append(lg - lg_last)
        trophy_data.append((lg, disagree))

    mean_move = sum(move_list) / len(move_list) if move_list else 0.0
    correction = GLOBAL_BLEND * (g_move - mean_move)

    if resid_sq:
        sigma_resid = math.sqrt(sum(resid_sq) / len(resid_sq))
    else:
        sigma_resid = 0.2
    span_days = (times[-1] - times[0]) / 24.0

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate_last = last_rates[i]
        if rate_last is None:
            rate_last = 0.5
        lg, disagree = trophy_data[i]
        lg += correction
        med = inv_logit_pct(lg)
        if rate_last >= 99.99:
            med = max(med, 99.0)
        if rate_last <= 0.0:
            med = min(med, 0.05)
        sigma = math.sqrt(max(sigma_resid, SIGMA_FLOOR) ** 2
                          + (SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                          + (SIGMA_DISAGREE * disagree) ** 2
                          + (SIGMA_MOVE * abs(lg - logit_pct(rate_last))) ** 2)
        if span_days < 2.0:
            sigma += 0.25 * (1.0 - span_days / 2.0)
        sigma = min(SIGMA_CAP, sigma)
        rows.append({"id": trophy["id"],
                     "q": [round(inv_logit_pct(
                         logit_pct(med) + z * sigma), 4) for z in Z]})
    return rows


def forecast_case(case):
    if list(case["levels"]) != LEVELS:
        raise ValueError("unexpected levels")
    if not case["history"]:
        return {"trophies": fallback_rows(case)}
    try:
        return {"trophies": model_rows(case)}
    except Exception:
        return {"trophies": fallback_rows(case)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
