# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nughair/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nughair/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A refresh-aware earners/players forecaster for Grand Theft Auto VI's trophy
earn rates on Exophase. Given the house's recorded reads of the game's trophy
page up to a checkpoint, `main.py` (Python 3, standard library only, offline,
deterministic) writes, for each of the game's trophies, five quantiles
(5/25/50/75/95%) of its earn rate at the case's target — in the final, the
rates at 2026-11-26T00:00:00Z, a week after launch.

## How it works

An earn rate is a ratio: earners / players tracked. After a launch the two
sides move on different clocks — the tracked-player denominator keeps growing
and mainstreaming for days, while each trophy's earners saturate as the keen
players earn it. So rates bend: they climb while earners outrun the crowd,
then fall as the crowd dilutes them.

The model forecasts every trophy two ways and blends them, both in log time
(u = ln(1 + hours since the first read)), the basis in which an earning
curve's logit is nearly linear while it is being earned:

- **Direct** — a recency-weighted least-squares line of the trophy's logit
  rate on u over the recent *distinct page states* (Exophase refreshes only
  every few hours, so identical reads are one observation), its slope shrunk
  toward the tier mean, extrapolated with light damping.
- **Decomposition** — the same fits applied separately to the trophy's
  earners (rate × players, recovered from the case) and to the tracked-player
  count; the forecast rate is the ratio at the target. This side sees
  dilution coming, and sees it ending when the pool fills up.

A common correction nudges the cross-trophy average move toward the move of
the global ratio (total trophies earned per tracked player), which the case
reports directly. Quantile widths pool the fits' residuals, grow with the
horizon and with the size of the move, and widen when the two forecasts
disagree — calibrated so the 90% band covers about 90%.

If a case's history holds fewer than three distinct page states (a
launch-night cut, where the page has never refreshed), no fit is possible and
the model uses a stale-page prior: a refresh is overdue, the tracked crowd is
still the small pre-launch one, and rates dilute toward the refreshed
population's lower holding rate. Any unexpected input shape falls back to a
last-read forecast with a random-walk spread, so a case never goes invalid.

## Validation

The two practice cases in the kit are 90-minute launch-night cuts whose
histories are perfectly flat; they cannot grade a 4.5-day forecaster. The
model was therefore validated on synthetic long-horizon cuts with the final's
exact geometry (cut 59.5 h after launch, target at 168 h), generated from
cohort player-growth + per-trophy earn-curve worlds behind a refresh
staircase, and scored with the contest's own pinball-on-logit rule against
exact replicas of the two house baselines. Over eight worlds (player pools
40k–500k, slow/fast trophies, dilution-dominant and saturating regimes):

| forecaster | mean pinball loss (logit) |
|---|---|
| this model | **0.285** |
| house trend baseline | 0.330 |
| house persistence baseline | 0.430 |

The model beats both baselines in seven of eight worlds (in the eighth, a
small saturating pool with a mid-window regime flip, it still beats
persistence), with mean 90% coverage of 0.88. On the kit's practice cases it
runs in ~0.07 s and, unlike persistence (which the refresh strands at a loss
of ~1.51), it forecasts the refresh dilution.

## Run it

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

MIT licensed (see `LICENSE`).
