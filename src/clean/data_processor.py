"""
Tennis Data Processing Pipeline
Standardizes player names, creates player IDs, and cleans all tennis datasets
"""

import pandas as pd
import numpy as np
import hashlib
import re
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TennisDataProcessor:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.raw_folder = self.data_folder / "raw"
        self.processed_folder = self.data_folder / "processed"
        self.processed_folder.mkdir(exist_ok=True)
        
        # Player normalization mapping
        self.player_normalization = {}
        self.players_db = {}  # playerId -> {name, hand, birthyear}
        
    def normalize_player_name(self, name):
        """Standardize player names for consistent identification"""
        if pd.isna(name) or name == "":
            return None
            
        # Remove extra whitespace and convert to title case
        name = str(name).strip()
        
        # Handle common abbreviations and formats
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
        
        # Common name format fixes
        # "Lastname F." -> "F. Lastname" 
        if re.match(r'^[A-Z][a-z]+ [A-Z]\.$', name):
            parts = name.split()
            name = f"{parts[1]} {parts[0]}"
        
        # Handle "Lastname First" -> "First Lastname"
        # This is tricky, we'll keep current format but standardize case
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def create_player_id(self, name):
        """Create consistent player ID using hash of normalized name"""
        if not name:
            return None
        normalized = self.normalize_player_name(name)
        if not normalized:
            return None
        # Create 8-character hash
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
    def extract_players_from_match_data(self, df):
        """Extract unique players from match data"""
        players = set()
        
        # Get players from Player_1 and Player_2 columns
        if 'Player_1' in df.columns:
            players.update(df['Player_1'].dropna().unique())
        if 'Player_2' in df.columns:
            players.update(df['Player_2'].dropna().unique())
        if 'Winner' in df.columns:
            players.update(df['Winner'].dropna().unique())
            
        return players
    
    def extract_players_from_performance_data(self, df):
        """Extract players from performance datasets"""
        players = set()
        if 'Player' in df.columns:
            players.update(df['Player'].dropna().unique())
        return players
    
    def build_players_database(self):
        """Build comprehensive players database from all datasets"""
        print("Building players database...")
        all_players = set()
        
        # Load main match data
        try:
            atp_df = pd.read_csv(self.raw_folder / "atp_tennis.csv")
            all_players.update(self.extract_players_from_match_data(atp_df))
            print(f"Found {len(all_players)} players from ATP matches")
        except FileNotFoundError:
            print("atp_tennis.csv not found")
        
        # Load performance data
        performance_files = [
            "serve_leaders.csv", "return_leaders.csv", 
            "rally_leaders.csv", "tactics_leaders.csv"
        ]
        
        for file in performance_files:
            try:
                df = pd.read_csv(self.raw_folder / file)
                players_from_file = self.extract_players_from_performance_data(df)
                all_players.update(players_from_file)
                print(f"Added {len(players_from_file)} players from {file}")
            except FileNotFoundError:
                print(f"{file} not found")
        
        # Create players database
        players_data = []
        for player_name in all_players:
            if player_name and pd.notna(player_name):
                normalized_name = self.normalize_player_name(player_name)
                if normalized_name:
                    player_id = self.create_player_id(normalized_name)
                    players_data.append({
                        'player_id': player_id,
                        'name': normalized_name,
                        'hand': None,  # To be filled from additional data sources
                        'birth_year': None  # To be filled from additional data sources
                    })
        
        # Remove duplicates based on player_id
        players_df = pd.DataFrame(players_data).drop_duplicates('player_id')
        
        # Create mapping for quick lookups
        self.player_id_mapping = dict(zip(players_df['name'], players_df['player_id']))
        
        # Save players table
        players_df.to_csv(self.processed_folder / "players.csv", index=False)
        print(f"Created players.csv with {len(players_df)} unique players")
        
        return players_df
    
    def standardize_surface(self, surface):
        """Standardize surface labels"""
        if pd.isna(surface):
            return None
            
        surface = str(surface).strip().lower()
        
        surface_mapping = {
            'hard': 'Hard',
            'clay': 'Clay', 
            'grass': 'Grass',
            'carpet': 'Carpet',
            'acrylic': 'Hard',
            'decoturf': 'Hard',
            'plexicushion': 'Hard',
            'rebound ace': 'Hard',
            'greenset': 'Hard'
        }
        
        return surface_mapping.get(surface, 'Hard')  # Default to Hard
    
    def standardize_round(self, round_str):
        """Standardize round names"""
        if pd.isna(round_str):
            return None
            
        round_str = str(round_str).strip()
        
        round_mapping = {
            '1st Round': 'R1',
            '2nd Round': 'R2', 
            '3rd Round': 'R3',
            '4th Round': 'R4',
            'Round of 128': 'R1',
            'Round of 64': 'R2',
            'Round of 32': 'R3',
            'Round of 16': 'R4',
            'Quarterfinals': 'QF',
            'Quarter-finals': 'QF',
            'Semifinals': 'SF',
            'Semi-finals': 'SF',
            'The Final': 'F',
            'Final': 'F',
            'Finals': 'F'
        }
        
        return round_mapping.get(round_str, round_str)
    
    def process_matches_data(self):
        """Process main ATP matches data"""
        print("\nProcessing ATP matches data...")
        
        try:
            df = pd.read_csv(self.raw_folder / "atp_tennis.csv")
            print(f"Loaded {len(df)} matches")
        except FileNotFoundError:
            print("atp_tennis.csv not found")
            return None
        
        # Rename columns to standard format
        column_mapping = {
            'Tournament': 'tournament',
            'Date': 'date', 
            'Series': 'level',
            'Court': 'court',
            'Surface': 'surface',
            'Round': 'round',
            'Best of': 'best_of',
            'Player_1': 'player1',
            'Player_2': 'player2', 
            'Winner': 'winner',
            'Rank_1': 'rank1',
            'Rank_2': 'rank2',
            'Pts_1': 'pts1',
            'Pts_2': 'pts2',
            'Odd_1': 'odd1',
            'Odd_2': 'odd2',
            'Score': 'score'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Standardize surface
        df['surface'] = df['surface'].apply(self.standardize_surface)
        
        # Standardize round
        df['round'] = df['round'].apply(self.standardize_round)
        
        # Convert best_of to integer
        df['best_of'] = pd.to_numeric(df['best_of'], errors='coerce')
        
        # Replace player names with IDs
        df['p1_id'] = df['player1'].apply(lambda x: self.player_id_mapping.get(self.normalize_player_name(x)))
        df['p2_id'] = df['player2'].apply(lambda x: self.player_id_mapping.get(self.normalize_player_name(x)))
        df['winner_id'] = df['winner'].apply(lambda x: self.player_id_mapping.get(self.normalize_player_name(x)))
        
        # Remove rows with missing essential data
        initial_count = len(df)
        df = df.dropna(subset=['p1_id', 'p2_id', 'winner_id', 'date'])
        final_count = len(df)
        print(f"Removed {initial_count - final_count} incomplete rows")
        
        # Sort by date
        df = df.sort_values('date')
        
        # Select final columns
        final_columns = [
            'date', 'tournament', 'level', 'round', 'surface', 'best_of',
            'player1', 'player2', 'winner', 'p1_id', 'p2_id', 'winner_id',
            'rank1', 'rank2', 'pts1', 'pts2', 'odd1', 'odd2', 'score'
        ]
        
        df = df[final_columns]
        
        # Save processed matches
        df.to_csv(self.processed_folder / "matches.csv", index=False)
        print(f"Saved {len(df)} processed matches to matches.csv")
        
        return df
    
    def process_performance_data(self):
        """Process serve, return, rally, and tactics data"""
        print("\nProcessing performance data...")
        
        performance_files = {
            "serve_leaders.csv": "serve_stats.csv",
            "return_leaders.csv": "return_stats.csv", 
            "rally_leaders.csv": "rally_stats.csv",
            "tactics_leaders.csv": "tactics_stats.csv"
        }
        
        for input_file, output_file in performance_files.items():
            try:
                df = pd.read_csv(self.raw_folder / input_file)
                print(f"Processing {input_file}...")
                
                # Normalize player names and add player IDs
                df['player_normalized'] = df['Player'].apply(self.normalize_player_name)
                df['player_id'] = df['player_normalized'].apply(lambda x: self.player_id_mapping.get(x))
                
                # Remove rows without player IDs
                initial_count = len(df)
                df = df.dropna(subset=['player_id'])
                final_count = len(df)
                
                if initial_count > final_count:
                    print(f"  Removed {initial_count - final_count} rows with unmatched players")
                
                # Clean column names - remove special characters and standardize
                def clean_column_name(col):
                    # Remove special characters and normalize spaces
                    col = re.sub(r'[^\w\s%]', '', str(col))
                    col = re.sub(r'\s+', '_', col.strip())
                    col = col.lower()
                    return col
                
                # Rename columns
                new_columns = {}
                for col in df.columns:
                    if col not in ['Player', 'player_normalized', 'player_id']:
                        new_columns[col] = clean_column_name(col)
                
                df = df.rename(columns=new_columns)
                
                # Reorder columns to put player info first
                cols = ['player_id', 'player_normalized'] + [col for col in df.columns if col not in ['Player', 'player_id', 'player_normalized']]
                df = df[cols]
                
                # Drop original player column
                df = df.drop('Player', axis=1, errors='ignore')
                
                # Save processed data
                df.to_csv(self.processed_folder / output_file, index=False)
                print(f"  Saved {len(df)} records to {output_file}")
                
            except FileNotFoundError:
                print(f"  {input_file} not found, skipping...")
            except Exception as e:
                print(f"  Error processing {input_file}: {e}")
    
    def generate_summary_report(self):
        """Generate a summary report of the processing"""
        print("\n" + "="*50)
        print("DATA PROCESSING SUMMARY")
        print("="*50)
        
        # Check processed files
        processed_files = list(self.processed_folder.glob("*.csv"))
        
        for file in processed_files:
            try:
                df = pd.read_csv(file)
                print(f"{file.name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"{file.name}: Error reading - {e}")
        
        print("="*50)
        print("Processing complete!")
    
    def run_full_pipeline(self):
        """Run the complete data processing pipeline"""
        print("Starting Tennis Data Processing Pipeline...")
        print("="*50)
        
        # Step 1: Build players database
        self.build_players_database()
        
        # Step 2: Process matches data  
        self.process_matches_data()
        
        # Step 3: Process performance data
        self.process_performance_data()
        
        # Step 4: Generate summary
        self.generate_summary_report()

if __name__ == "__main__":
    # Initialize processor
    data_folder = Path(__file__).parent.parent.parent / "data"
    processor = TennisDataProcessor(data_folder)
    
    # Run full pipeline
    processor.run_full_pipeline()