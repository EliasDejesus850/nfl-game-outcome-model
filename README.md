# NFL Game Outcome Model

Predicts whether the **home team wins** an NFL game from each team's recent form, and checks the result against honest baselines.

Built with Python, pandas, scikit-learn and Matplotlib on 13 seasons of game logs (2002-2014, 3,300+ games).

## Results

Trained on 2002-2012, tested on the 2013-2014 seasons (512 games):

| Approach | Accuracy |
|---|---|
| Always pick the home team (baseline) | 58.2% |
| **Logistic regression on home-minus-away rolling features (this model)** | **63.5%** (95% interval about 59-68%) |

The model setup was chosen using 2011-2012 validation games only, then scored once on the test seasons.
With about 500 test games the standard error is roughly 2 points, so this is a modest edge over the baseline, not a proven one.
Across all 32 setups tried, test accuracy ranged from 59.8% to 67.0% (median 62.9%).

## Approach

1. **Scrape** game logs from Pro-Football-Reference with pandas (`notebooks/01_scrape_game_logs.ipynb` -> `data/raw/nfl_game_logs.csv`).
2. **Build** a one-row-per-team-per-game table (`src/build_matches.py` -> `data/processed/team_games.csv`). Values are matched by team and game, not by row position.
3. **Engineer features** (`notebooks/02_game_outcome_model.ipynb`): rolling averages over each team's previous games (the current game is excluded to prevent leakage), then the home team's averages **minus** the away team's, so both teams' recent form appear in the same row.
4. **Validate by date**: train / validation / test splits follow the calendar, never shuffled.
5. **Compare** windows of 3, 6, 10 and 16 games, two feature sets, logistic regression and shallow random forests.

## Key takeaways

- Putting both teams' recent form in the same row (home minus away) mattered: features that describe only one team cannot capture a matchup.
- Comparing against a baseline and keeping a validation period separate from the test period mattered more than the choice of model. Simple, regularized logistic regression beat the random forests.
- Building the processed table directly from the scraped data, matched by team and game, avoids misaligned stats. `src/build_matches.py` checks row counts, game pairing and yards-per-play consistency.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/02_game_outcome_model.ipynb
```

The raw and processed data are included in `data/`, so the model notebook runs as is.

To regenerate the data from scratch:

```bash
# 1. Re-scrape (takes several minutes; pauses between requests)
#    run notebooks/01_scrape_game_logs.ipynb  -> data/raw/nfl_game_logs.csv
# 2. Rebuild the model input
python src/build_matches.py data/raw/nfl_game_logs.csv data/processed/team_games.csv
```

## Project structure

```
nfl-game-outcome-model/
├── data/
│   ├── raw/nfl_game_logs.csv          # scraped output
│   └── processed/team_games.csv       # one row per team per game, model input
├── notebooks/
│   ├── 01_scrape_game_logs.ipynb      # scrapes 13 seasons of game logs
│   └── 02_game_outcome_model.ipynb    # features, validation, model, results
├── src/
│   └── build_matches.py               # builds team_games.csv from the raw logs
├── requirements.txt
└── README.md
```

## Limits

- Team-level game stats only: no injuries, quarterbacks, weather or betting lines.
- The first games of the dataset are dropped because there is no earlier history to average.
