"""
Data Processor
Cleans and normalizes tennis data from multiple sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


class TennisDataProcessor:
    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)
        self.processed_folder = self.data_folder / "processed"
        self.processed_folder.mkdir(exist_ok=True)
        
    def normalize_player_name(self, name):
        """Normalize player names for consistency"""
        if pd.isna(name):
            return None
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Standardize format: "Last F." -> "F. Last"
        # Example: "Federer R." -> "R. Federer"
        parts = name.split()
        if len(parts) == 2 and parts[1].endswith('.'):
            return f"{parts[1]} {parts[0]}"
        
        return name
    
    def normalize_surface(self, surface):
        """Standardize surface names"""
        if pd.isna(surface):
            return None
        
        surface = surface.lower().strip()
        
        surface_map = {
            'hard': 'Hard',
            'clay': 'Clay',
            'grass': 'Grass',
            'carpet': 'Carpet'
        }
        
        return surface_map.get(surface, surface.capitalize())
    
    def load_match_data(self):
        """Load the main ATP match dataset"""
        print("Loading ATP match data...")
        
        matches_file = self.data_folder / "atp_tennis.csv"
        if not matches_file.exists():
            raise FileNotFoundError(f"Match data not found: {matches_file}")
        
        df = pd.read_csv(matches_file)
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Normalize player names
        df['player_1'] = df['player_1'].apply(self.normalize_player_name)
        df['player_2'] = df['player_2'].apply(self.normalize_player_name)
        df['winner'] = df['winner'].apply(self.normalize_player_name)
        
        # Normalize surface
        df['surface'] = df['surface'].apply(self.normalize_surface)
        
        # Clean ranks (convert to int, handle missing)
        df['rank_1'] = pd.to_numeric(df['rank_1'], errors='coerce')
        df['rank_2'] = pd.to_numeric(df['rank_2'], errors='coerce')
        
        print(f"Loaded {len(df)} matches from {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def load_player_stats(self):
        """Load player statistics from Tennis Abstract"""
        print("\nLoading player statistics...")
        
        stats_files = {
            'serve': 'serve_leaders.csv',
            'return': 'return_leaders.csv',
            'rally': 'rally_leaders.csv',
            'tactics': 'tactics_leaders.csv'
        }
        
        stats = {}
        
        for stat_type, filename in stats_files.items():
            filepath = self.data_folder / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                
                # Normalize player names
                df['Player'] = df['Player'].apply(self.normalize_player_name)
                
                # Add prefix to all stat columns except Player
                df.columns = [col if col == 'Player' 
                             else f"{stat_type}_{col.lower().replace(' ', '_').replace('%', 'pct')}"
                             for col in df.columns]
                
                stats[stat_type] = df
                print(f"  Loaded {len(df)} players from {filename}")
            else:
                print(f"  Warning: {filename} not found")
        
        return stats
    
    def create_player_mapping(self, matches_df, stats_dict):
        """Create unified player ID mapping"""
        print("\nCreating player mapping...")
        
        # Get all unique players from matches
        players_from_matches = set()
        players_from_matches.update(matches_df['player_1'].dropna().unique())
        players_from_matches.update(matches_df['player_2'].dropna().unique())
        
        # Get all unique players from stats
        players_from_stats = set()
        for stat_df in stats_dict.values():
            players_from_stats.update(stat_df['Player'].dropna().unique())
        
        # Combine all players
        all_players = sorted(players_from_matches | players_from_stats)
        
        # Create player ID mapping
        player_mapping = pd.DataFrame({
            'player_id': range(1, len(all_players) + 1),
            'player_name': all_players
        })
        
        print(f"Created mapping for {len(player_mapping)} unique players")
        print(f"  From matches: {len(players_from_matches)}")
        print(f"  From stats: {len(players_from_stats)}")
        
        return player_mapping
    
    def process_matches(self, matches_df, player_mapping):
        """Add player IDs to match data"""
        print("\nProcessing match data...")
        
        # Create player name to ID mapping
        name_to_id = dict(zip(player_mapping['player_name'], 
                             player_mapping['player_id']))
        
        # Add player IDs
        matches_df['p1_id'] = matches_df['player_1'].map(name_to_id)
        matches_df['p2_id'] = matches_df['player_2'].map(name_to_id)
        matches_df['winner_id'] = matches_df['winner'].map(name_to_id)
        
        # Calculate loser_id
        matches_df['loser_id'] = matches_df.apply(
            lambda row: row['p2_id'] if row['winner_id'] == row['p1_id'] else row['p1_id'],
            axis=1
        )
        
        # Clean up column names for consistency
        matches_df = matches_df.rename(columns={
            'rank_1': 'p1_rank',
            'rank_2': 'p2_rank',
            'pts_1': 'p1_points',
            'pts_2': 'p2_points',
            'odd_1': 'p1_odds',
            'odd_2': 'p2_odds',
            'series': 'level',
            'best_of': 'best_of'
        })
        
        # Select and order columns
        columns = [
            'date', 'tournament', 'level', 'surface', 'round', 'best_of',
            'player_1', 'player_2', 'winner',
            'p1_id', 'p2_id', 'winner_id', 'loser_id',
            'p1_rank', 'p2_rank', 'p1_points', 'p2_points',
            'p1_odds', 'p2_odds', 'score'
        ]
        
        matches_df = matches_df[columns]
        
        # Remove matches with missing player IDs
        before = len(matches_df)
        matches_df = matches_df.dropna(subset=['p1_id', 'p2_id', 'winner_id'])
        after = len(matches_df)
        
        if before > after:
            print(f"  Removed {before - after} matches with unmapped players")
        
        print(f"Processed {len(matches_df)} matches")
        
        return matches_df
    
    def merge_player_stats(self, player_mapping, stats_dict):
        """Merge all player statistics into one dataframe"""
        print("\nMerging player statistics...")
        
        # Start with player mapping
        player_stats = player_mapping.copy()
        
        # Merge each stat type
        for stat_type, stat_df in stats_dict.items():
            # Rename Player column to player_name for merging
            stat_df = stat_df.rename(columns={'Player': 'player_name'})
            
            # Merge
            player_stats = player_stats.merge(
                stat_df,
                on='player_name',
                how='left'
            )
            
            print(f"  Merged {stat_type} stats: {len(stat_df)} players matched")
        
        print(f"Final player stats: {len(player_stats)} players, {len(player_stats.columns)} columns")
        
        return player_stats
    
    def save_processed_data(self, matches_df, player_stats):
        """Save processed data to CSV files"""
        print("\nSaving processed data...")
        
        # Save matches
        matches_file = self.processed_folder / "matches.csv"
        matches_df.to_csv(matches_file, index=False)
        print(f"  Saved matches: {matches_file}")
        
        # Save player stats
        players_file = self.processed_folder / "players.csv"
        player_stats.to_csv(players_file, index=False)
        print(f"  Saved players: {players_file}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETE")
        print("="*60)
        print(f"\nMatches: {len(matches_df)}")
        print(f"  Date range: {matches_df['date'].min()} to {matches_df['date'].max()}")
        print(f"  Surfaces: {matches_df['surface'].value_counts().to_dict()}")
        print(f"  Tournaments: {matches_df['tournament'].nunique()}")
        
        print(f"\nPlayers: {len(player_stats)}")
        print(f"  With serve stats: {player_stats['serve_matches'].notna().sum()}")
        print(f"  With return stats: {player_stats['return_matches'].notna().sum()}")
        print(f"  With rally stats: {player_stats['rally_matches'].notna().sum()}")
        print(f"  With tactics stats: {player_stats['tactics_matches'].notna().sum()}")
        
        print("\nSample match data:")
        print(matches_df.head())
        
        print("\nSample player data:")
        print(player_stats[['player_id', 'player_name', 'serve_matches', 'return_matches']].head())
    
    def run(self):
        """Run the complete data processing pipeline"""
        print("="*60)
        print("TENNIS DATA PROCESSOR")
        print("="*60)
        
        # Load data
        matches_df = self.load_match_data()
        stats_dict = self.load_player_stats()
        
        # Create player mapping
        player_mapping = self.create_player_mapping(matches_df, stats_dict)
        
        # Process matches with player IDs
        matches_df = self.process_matches(matches_df, player_mapping)
        
        # Merge player statistics
        player_stats = self.merge_player_stats(player_mapping, stats_dict)
        
        # Save processed data
        self.save_processed_data(matches_df, player_stats)
        
        return matches_df, player_stats


if __name__ == "__main__":
    processor = TennisDataProcessor()
    matches, players = processor.run()