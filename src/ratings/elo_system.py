"""
Final Tennis Elo System - With Surface Ratings + Save Functionality
"""

import pandas as pd
from collections import defaultdict
from pathlib import Path
import time

class TennisEloSystem:
    def __init__(self, initial_rating=1500, k_factor=32):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        
        # Player ratings: overall + per surface
        self.ratings = defaultdict(lambda: {
            'overall': initial_rating,
            'Hard': initial_rating,
            'Clay': initial_rating,
            'Grass': initial_rating,
            'Carpet': initial_rating,
            'matches_played': 0
        })
        
        # Store rating snapshots for each match (for ML features)
        self.match_ratings = []
    
    def expected_probability(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner_id, loser_id, surface, date, match_idx):
        # Get current ratings
        winner_overall = self.ratings[winner_id]['overall']
        loser_overall = self.ratings[loser_id]['overall']
        winner_surface = self.ratings[winner_id][surface]
        loser_surface = self.ratings[loser_id][surface]
        
        # Store pre-match ratings (for ML features)
        self.match_ratings.append({
            'match_idx': match_idx,
            'date': date,
            'surface': surface,
            'winner_id': winner_id,
            'loser_id': loser_id,
            'winner_elo_overall': winner_overall,
            'loser_elo_overall': loser_overall,
            'winner_elo_surface': winner_surface,
            'loser_elo_surface': loser_surface
        })
        
        # Update overall ratings
        expected_overall = self.expected_probability(winner_overall, loser_overall)
        change_overall = self.k_factor * (1 - expected_overall)
        
        self.ratings[winner_id]['overall'] += change_overall
        self.ratings[loser_id]['overall'] -= change_overall
        
        # Update surface-specific ratings
        expected_surface = self.expected_probability(winner_surface, loser_surface)
        change_surface = self.k_factor * (1 - expected_surface)
        
        self.ratings[winner_id][surface] += change_surface
        self.ratings[loser_id][surface] -= change_surface
        
        # Update match counts
        self.ratings[winner_id]['matches_played'] += 1
        self.ratings[loser_id]['matches_played'] += 1
    
    def process_matches(self, matches_df):
        print(f"Processing {len(matches_df):,} matches with Elo system...")
        
        # Sort by date
        matches_df = matches_df.sort_values('date').copy()
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        start_time = time.time()
        processed = 0
        skipped = 0
        
        for idx, (_, match) in enumerate(matches_df.iterrows()):
            # Progress every 5000 matches
            if (idx + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"Progress: {idx+1:,}/{len(matches_df)} ({(idx+1)/len(matches_df)*100:.1f}%) - "
                      f"{rate:.0f} matches/sec")
            
            # Get match details
            winner_id = match['winner_id']
            p1_id = match['p1_id']
            p2_id = match['p2_id']
            surface = match['surface']
            date = match['date']
            
            # Determine loser
            if winner_id == p1_id:
                loser_id = p2_id
            elif winner_id == p2_id:
                loser_id = p1_id
            else:
                skipped += 1
                continue
            
            # Update ratings
            self.update_ratings(winner_id, loser_id, surface, date, processed)
            processed += 1
        
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time:.1f} seconds!")
        print(f"Processed: {processed:,} matches")
        print(f"Skipped: {skipped} matches")
        print(f"Players tracked: {len(self.ratings)}")
        print(f"Speed: {processed/total_time:.0f} matches/second")
        
        return processed
    
    def get_ratings_df(self):
        """Convert ratings to DataFrame"""
        ratings_data = []
        for player_id, ratings in self.ratings.items():
            ratings_data.append({
                'player_id': player_id,
                'elo_overall': ratings['overall'],
                'elo_hard': ratings['Hard'],
                'elo_clay': ratings['Clay'],
                'elo_grass': ratings['Grass'],
                'elo_carpet': ratings['Carpet'],
                'matches_played': ratings['matches_played']
            })
        return pd.DataFrame(ratings_data)
    
    def get_match_ratings_df(self):
        """Convert match ratings to DataFrame"""
        return pd.DataFrame(self.match_ratings)
    
    def save_results(self, output_folder="data/features"):
        """Save ratings and match history"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Save current ratings
        ratings_df = self.get_ratings_df()
        ratings_df.to_csv(output_path / "elo_ratings.csv", index=False)
        print(f"Saved current ratings: {len(ratings_df)} players")
        
        # Save match-by-match ratings history
        match_ratings_df = self.get_match_ratings_df()
        match_ratings_df.to_csv(output_path / "match_ratings.csv", index=False)
        print(f"Saved match ratings: {len(match_ratings_df):,} matches")
        
        # Show top players by overall Elo
        print(f"\nTop 15 players by overall Elo:")
        top_players = ratings_df.nlargest(15, 'elo_overall')
        for i, (_, player) in enumerate(top_players.iterrows(), 1):
            print(f"{i:2d}. {player['player_id']}: {player['elo_overall']:.0f} "
                  f"({player['matches_played']} matches)")
        
        # Show surface specialists
        print(f"\nClay specialists (Clay - Overall):")
        ratings_df['clay_advantage'] = ratings_df['elo_clay'] - ratings_df['elo_overall']
        clay_specialists = ratings_df.nlargest(5, 'clay_advantage')
        for _, player in clay_specialists.iterrows():
            if player['matches_played'] >= 20:  # Minimum matches
                print(f"{player['player_id']}: Clay {player['elo_clay']:.0f} "
                      f"vs Overall {player['elo_overall']:.0f} "
                      f"(+{player['clay_advantage']:.0f})")
        
        return ratings_df, match_ratings_df

def main():
    # Load match data
    try:
        matches_df = pd.read_csv("data/processed/matches.csv")
        print(f"Loaded {len(matches_df):,} matches")
    except FileNotFoundError:
        print("Error: data/processed/matches.csv not found!")
        return
    
    # Create Elo system
    elo_system = TennisEloSystem(initial_rating=1500, k_factor=32)
    
    # Process all matches
    processed_count = elo_system.process_matches(matches_df)
    
    # Save results
    ratings_df, match_ratings_df = elo_system.save_results()
    
    print(f"\n🎾 Elo rating system complete!")
    print(f"Files saved to data/features/:")
    print(f"  - elo_ratings.csv ({len(ratings_df)} players)")
    print(f"  - match_ratings.csv ({len(match_ratings_df):,} matches)")

if __name__ == "__main__":
    main()