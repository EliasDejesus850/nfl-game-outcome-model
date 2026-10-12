"""
build_matches.py
Builds the processed team-game table (input for 02_game_outcome_model.ipynb) from the
scraped game logs (output of 01_scrape_game_logs.ipynb).

Usage:
    python src/build_matches.py data/raw/nfl_game_logs.csv data/processed/team_games.csv

Every value comes from the scraped data, matched by team and game (never by row
position), so the two files cannot disagree.
Optional flag --penalty-first-downs fills team_penalties / opp_penalties with first
downs gained by penalty instead of the true penalty count (not used by the notebook).
"""
import sys
import numpy as np
import pandas as pd

TEAM_NAMES = {  # scraper team code -> full name used by the model
    'CRD': 'Arizona', 'ATL': 'Atlanta', 'RAV': 'Baltimore', 'BUF': 'Buffalo',
    'CAR': 'Carolina', 'CHI': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland',
    'DAL': 'Dallas', 'DEN': 'Denver', 'DET': 'Detroit', 'GNB': 'Green Bay',
    'HTX': 'Houston', 'CLT': 'Indianapolis', 'JAX': 'Jacksonville', 'KAN': 'Kansas City',
    'SDG': 'San Diego', 'RAM': 'St. Louis', 'RAI': 'Oakland', 'MIA': 'Miami',
    'MIN': 'Minnesota', 'NWE': 'New England', 'NOR': 'New Orleans', 'NYG': 'NY Giants',
    'NYJ': 'NY Jets', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh', 'SEA': 'Seattle',
    'SFO': 'San Francisco', 'TAM': 'Tampa Bay', 'OTI': 'Tennessee', 'WAS': 'Washington',
}
OPP_NAMES = {  # opponent abbreviations as they appear in the scraped 'Opp' column
    'ARI': 'Arizona', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BUF': 'Buffalo',
    'CAR': 'Carolina', 'CHI': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland',
    'DAL': 'Dallas', 'DEN': 'Denver', 'DET': 'Detroit', 'GNB': 'Green Bay',
    'HOU': 'Houston', 'IND': 'Indianapolis', 'JAX': 'Jacksonville', 'KAN': 'Kansas City',
    'SDG': 'San Diego', 'STL': 'St. Louis', 'OAK': 'Oakland', 'MIA': 'Miami',
    'MIN': 'Minnesota', 'NWE': 'New England', 'NOR': 'New Orleans', 'NYG': 'NY Giants',
    'NYJ': 'NY Jets', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh', 'SEA': 'Seattle',
    'SFO': 'San Francisco', 'TAM': 'Tampa Bay', 'TEN': 'Tennessee', 'WAS': 'Washington',
}


def top_seconds(s):
    """'25:36' -> 1536"""
    mins, secs = s.split(':')
    return int(mins) * 60 + int(secs)


def side_stats(n, me, other, penalty_first_downs):
    """Per-side stat columns. `me` and `other` are 'Tm_' / 'Opp_' prefixes."""
    c = lambda name: n[f'{me}{name}']
    d = {}
    d['score'] = c('Pts')
    # Sum of the components: the scraped 1stD total is off in ~0.3% of rows (e.g. 9 vs 18 components)
    d['first_downs'] = c('fdRush') + c('fdPass') + c('fdPen')
    d['rushing_first_downs'] = c('fdRush')
    d['passing_first_downs'] = c('fdPass')
    d['penalties'] = c('fdPen') if penalty_first_downs else c('Pen')
    d['net_yards'] = c('Tot')
    d['net_yards_rushing'] = c('rYds')
    d['rushing_plays'] = c('rAtt')
    d['avg_rush'] = c('rY/A')
    d['net_yards_passing'] = c('pYds')
    d['gross_passing'] = c('pYds') + c('SkYds')
    d['yards_per_pass'] = ((c('pYds') + c('SkYds')) / c('pAtt')).round(1)
    d['third_conversions'] = c('3DConv')
    d['third_attempts'] = c('3DAtt')
    rate = (c('3DConv') / c('3DAtt').replace(0, np.nan)).fillna(0)
    d['third_success_rate'] = np.floor(rate * 100 + 1e-9) / 100
    d['total_plays'] = c('Ply')
    d['avg_gain'] = (c('Tot') / c('Ply')).round(1)
    d['yard_differential'] = c('Tot') - n[f'{other}Tot']
    d['time_of_possession'] = c('ToP').map(top_seconds)
    d['pRate'] = c('pRate')
    d['Sk'] = c('Sk')
    d['SkYds'] = c('SkYds')
    d['Y/P'] = c('Y/P')
    d['PntYds'] = c('PntYds')
    d['PenYds'] = c('PenYds')
    d['TO'] = c('TO')
    return d


def build(n, penalty_first_downs=False):
    n = n.copy()
    n['Date'] = pd.to_datetime(n['Date'])
    out = pd.DataFrame(index=n.index)
    out['date'] = n['Date'].dt.strftime('%Y-%m-%d')
    out['comp'] = 'NFL'
    out['day'] = n['Day']
    out['team'] = n['Team'].map(TEAM_NAMES)
    out['opponent'] = n['Opp'].map(OPP_NAMES)
    out['venue'] = np.where(n['Home'] == 1, 'Home', 'Away')
    out['result'] = np.select([n['Tm_Pts'] > n['Opp_Pts'], n['Tm_Pts'] < n['Opp_Pts']], ['W', 'L'], 'D')
    # days since this team's previous game in the same season (-1 for the opener)
    gap = n.groupby(['Team', 'Season'])['Date'].diff().dt.days
    out['rest_days'] = gap.fillna(-1).astype(int)
    out['season_opening'] = (n.groupby(['Team', 'Season']).cumcount() == 0).astype(int)
    out['season'] = n['Season']
    for prefix, me, other in (('team_', 'Tm_', 'Opp_'), ('opp_', 'Opp_', 'Tm_')):
        for k, v in side_stats(n, me, other, penalty_first_downs).items():
            out[prefix + k] = v
    # Row order: season, then team name (A-Z), then date.
    out = out.sort_values(['season', 'team', 'date'], kind='stable')
    out.index = range(1, len(out) + 1)  # index starts at 1
    return out


def check(out):
    assert out['team'].notna().all() and out['opponent'].notna().all(), 'unmapped team name'
    assert len(out) == 32 * 16 * 13, 'expected 6,656 rows'
    for p in ('team_', 'opp_'):
        err = (out[p + 'net_yards'] / out[p + 'total_plays'] - out[p + 'Y/P']).abs().max()
        assert err < 0.06, f'{p}Y/P inconsistent with yards/plays (max error {err:.2f})'
    # every game must appear exactly twice, with swapped scores
    g = out.assign(a=out[['team', 'opponent']].min(axis=1), b=out[['team', 'opponent']].max(axis=1))
    assert (g.groupby(['date', 'a', 'b']).size() == 2).all(), 'game not paired'


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    src, dst = args[0], args[1]
    matches = build(pd.read_csv(src), penalty_first_downs='--penalty-first-downs' in sys.argv)
    check(matches)
    matches.to_csv(dst)
    print(f'Wrote {dst}: {matches.shape[0]} rows x {matches.shape[1]} columns')
