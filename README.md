# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/garyadrianto1/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/garyadrianto1/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's PlayStation 5 trophy earn rates on Exophase, one week after launch. It reads one case and forecasts, for each of the game's 55 trophies, five quantiles of its earn rate at the case's target.

## Method

Work in **counts, not rates**: earners E_i(t) = rate_i(t) * players_tracked / 100, so a rate moves both when earners grow and when new players join (dilution). Exophase refreshes its counts every few hours, so the history is collapsed to distinct refresh states first.

For each trophy and for the player count, a log-log power law in log-time is fitted (ln y vs ln(1 + t), t = hours since launch), anchored at the last distinct state. Two slope estimators are blended in logit space: the whole-history slope (stable, capped at 1.2) and the recent slope over the last 4 distinct states (tracks acceleration as the player base's playtime crosses each trophy's threshold, capped at 2.0). The trophy every tracked player holds (100%) stays 100%. Spread is 0.8 * 0.5 * sqrt(horizon in days) on the logit scale.

On synthetic 4.5-day backtests (cohort ground truth, 3 player-influx scenarios x 3 seeds) this scored mean pinball loss 0.286 (range 0.266–0.302) vs 0.856 for the persistence starter. The practice cases (90-minute horizon, flat histories) cannot validate the final model and are not optimised for — their dev board is gamed by refresh inversion; see INTAKE.md in the build notes.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Deterministic, no network, no randomness.

---

<sub>Entered on [make.fun](https://make.fun/?ref=m1ceh0z1&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
