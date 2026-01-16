"""
Filter matches to only include the 68 players with detailed stats
"""

import pandas as pd
from pathlib import Path

def filter_to_stats_players():
    data_folder = Path("data")
    processed_folder = data_folder / "processed"
    
    print("Loading data...")
    
    # Load processed data
    matches = pd.read_csv(processed_folder / "matches.csv")
    players = pd.read_csv(processed_folder / "players.csv")
    
    print(f"Total matches: {len(matches)}")
    print(f"Total players: {len(players)}")
    
    # Get players with serve stats (has detailed Tennis Abstract data)
    stats_players = players[players['serve_matches'].notna()]['player_id'].tolist()
    
    print(f"\nPlayers with detailed stats: {len(stats_players)}")
    
    # Filter matches where BOTH players have stats
    filtered_matches = matches[
        matches['p1_id'].isin(stats_players) & 
        matches['p2_id'].isin(stats_players)
    ].copy()
    
    print(f"\nFiltered matches (both players have stats): {len(filtered_matches)}")
    print(f"Date range: {filtered_matches['date'].min()} to {filtered_matches['date'].max()}")
    print(f"Surfaces: {filtered_matches['surface'].value_counts().to_dict()}")
    
    # Save filtered matches
    filtered_file = processed_folder / "matches_filtered.csv"
    filtered_matches.to_csv(filtered_file, index=False)
    print(f"\nSaved: {filtered_file}")
    
    # Filter players to just the 68
    filtered_players = players[players['player_id'].isin(stats_players)].copy()
    players_file = processed_folder / "players_filtered.csv"
    filtered_players.to_csv(players_file, index=False)
    print(f"Saved: {players_file}")
    
    print("\n" + "="*60)
    print("FILTERING COMPLETE")
    print("="*60)
    print(f"Dataset reduced from {len(matches):,} to {len(filtered_matches):,} matches")
    print(f"Player pool: {len(stats_players)} players with full stats")
    
    # Show some sample players
    print("\nSample players:")
    print(filtered_players[['player_name', 'serve_matches', 'return_matches']].head(10))

if __name__ == "__main__":
    filter_to_stats_players()
