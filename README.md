# DS 4021 — Pokémon Battle Outcome Prediction

## Research question
How accurately can machine learning models predict simulated Pokémon battle outcomes using base stats and type matchups, and which characteristics contribute most to prediction accuracy?

## Data
- `Data/pokemon.csv`: 800 Pokémon entries with attributes (types, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Generation, Legendary).
- `Data/reporting_set.csv`: 10,000 held-out battle records (20% of the 50,000-row original combat dataset). **Do not use, modify, inspect for model development, or tune against this set until final reporting.**
- Original source: [Pokémon Combat Dataset (Kaggle)](https://www.kaggle.com/datasets/terminus7/pokemon-challenge). The `pokemon.csv` and `combats.csv` source copies used here are hosted at [dannypadilla/pokemon-battle-predictor](https://github.com/dannypadilla/pokemon-battle-predictor).

## How features and target are constructed
The battle records identify `First_pokemon`, `Second_pokemon`, and `Winner`. Both Pokémon ID columns refer to the `#` column in `pokemon.csv` (these are dataset entry IDs, **not** necessarily National Pokédex numbers).

For **each** battle, join the attributes table twice:
1. `First_pokemon` → `pokemon.csv` `#` to obtain Pokémon 1's attributes.
2. `Second_pokemon` → `pokemon.csv` `#` to obtain Pokémon 2's attributes.

Input features will include both Pokémon's base stats and type information, with optional engineered features such as stat differences and type matchups. The binary **target** is `pokemon_1_wins = 1` if `Winner == First_pokemon`, otherwise `0` if `Winner == Second_pokemon`. **Do not include `Winner` or any target-derived column as a model input.** Pokémon ID columns are join keys, not intended predictive features.

## Holdout methodology
The 10,000 reporting records were sampled from the original 50,000 battles using a deterministic Fisher–Yates shuffle with a 32-bit linear congruential generator initialized at seed `4021`; the first 10,000 shuffled zero-based indices were selected, and their rows were saved in original source order. The holdout was committed before model development. The remaining 40,000 battle rows (not yet uploaded) will be used for training, validation, and model selection. **The holdout must never overlap with those training rows.** Preserve the original row indices or reproduce the documented split when preparing training data.

## Limitation
The battles are simulated, not observed competitive matches. Model findings will describe patterns in this simulated dataset and may not generalize to full game mechanics or player-driven battles.
