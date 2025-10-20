import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class PokerFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        
    def extract_action_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract patterns from action sequences"""
        action_cols = [col for col in df.columns if col.startswith('action_')]
        
        # Action frequency features
        for i in range(10):  # 10 possible actions
            df[f'action_freq_{i}'] = (df[action_cols] == i).sum(axis=1)
        
        # Action sequence length (non-padding actions)
        df['action_sequence_length'] = (df[action_cols] != -1).sum(axis=1)
        
        # Aggression ratio (betting/raising vs calling/checking)
        aggressive_actions = [3, 4, 5, 6, 7, 8, 9]  # BET_*, RAISE_*, ALL_IN
        passive_actions = [1, 2]  # CHECK, CALL
        
        df['aggression_ratio'] = (
            (df[action_cols].isin(aggressive_actions)).sum(axis=1) /
            ((df[action_cols].isin(aggressive_actions)).sum(axis=1) + 
             (df[action_cols].isin(passive_actions)).sum(axis=1) + 1e-8)
        )
        
        # Bet sizing patterns
        small_bets = [3, 6]  # BET_SMALL, RAISE_SMALL
        medium_bets = [4, 7]  # BET_MEDIUM, RAISE_MEDIUM
        large_bets = [5, 8, 9]  # BET_LARGE, RAISE_LARGE, ALL_IN
        
        df['small_bet_ratio'] = (df[action_cols].isin(small_bets)).sum(axis=1) / (df['action_sequence_length'] + 1e-8)
        df['medium_bet_ratio'] = (df[action_cols].isin(medium_bets)).sum(axis=1) / (df['action_sequence_length'] + 1e-8)
        df['large_bet_ratio'] = (df[action_cols].isin(large_bets)).sum(axis=1) / (df['action_sequence_length'] + 1e-8)
        
        return df
    
    def extract_street_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for each street"""
        streets = ['preflop', 'flop', 'turn', 'river']
        
        for street in streets:
            # Street-specific action features
            street_action_cols = [f'action_{i}' for i in range(10 * streets.index(street), 10 * (streets.index(street) + 1))]
            
            if street_action_cols:
                # Aggression on this street
                aggressive_actions = [3, 4, 5, 6, 7, 8, 9]
                df[f'{street}_aggression'] = (df[street_action_cols].isin(aggressive_actions)).sum(axis=1)
                
                # Action frequency on this street
                df[f'{street}_action_count'] = (df[street_action_cols] != -1).sum(axis=1)
                
                # Bet sizing on this street
                large_bets = [5, 8, 9]
                df[f'{street}_large_bet_count'] = (df[street_action_cols].isin(large_bets)).sum(axis=1)
        
        return df
    
    def extract_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract position-based features"""
        # Position categories
        df['early_position'] = (df['preflop_position'] <= 2).astype(int)
        df['middle_position'] = ((df['preflop_position'] > 2) & (df['preflop_position'] <= 5)).astype(int)
        df['late_position'] = (df['preflop_position'] > 5).astype(int)
        
        # Position interaction with aggression
        df['early_aggression'] = df['early_position'] * df['preflop_aggression']
        df['late_aggression'] = df['late_position'] * df['preflop_aggression']
        
        return df
    
    def extract_pot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract pot and stack-related features"""
        # Pot odds features
        df['pot_stack_ratio'] = df['preflop_pot_size'] / (df['preflop_stack_size'] + 1e-8)
        
        # Stack depth
        df['deep_stack'] = (df['preflop_stack_size'] > 2000).astype(int)
        df['short_stack'] = (df['preflop_stack_size'] < 500).astype(int)
        
        # Multi-way pot features
        df['multi_way'] = (df['preflop_num_players'] > 3).astype(int)
        df['heads_up'] = (df['preflop_num_players'] == 2).astype(int)
        
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features that capture how actions change over streets"""
        # Action progression
        df['action_progression'] = (
            df['flop_action_count'] - df['preflop_action_count']
        )
        
        # Aggression progression
        df['aggression_progression'] = (
            df['flop_aggression'] - df['preflop_aggression']
        )
        
        # Bet sizing progression
        df['bet_sizing_progression'] = (
            df['flop_large_bet_count'] - df['preflop_large_bet_count']
        )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables"""
        # Position and aggression interaction
        df['position_aggression'] = df['preflop_position'] * df['aggression_ratio']
        
        # Stack and bet sizing interaction
        df['stack_bet_interaction'] = df['preflop_stack_size'] * df['large_bet_ratio']
        
        # Pot size and aggression interaction
        df['pot_aggression'] = df['preflop_pot_size'] * df['aggression_ratio']
        
        # Multi-way and aggression interaction
        df['multiway_aggression'] = df['multi_way'] * df['aggression_ratio']
        
        return df
    
    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from action sequences"""
        action_cols = [col for col in df.columns if col.startswith('action_')]
        
        # Action transitions
        for i in range(len(action_cols) - 1):
            col1, col2 = action_cols[i], action_cols[i + 1]
            df[f'transition_{i}'] = ((df[col1] != -1) & (df[col2] != -1)).astype(int)
        
        # Action consistency (same action type repeated)
        for action_type in range(10):
            action_cols_filtered = [col for col in action_cols if col in df.columns]
            df[f'consistency_{action_type}'] = (
                (df[action_cols_filtered] == action_type).sum(axis=1) > 1
            ).astype(int)
        
        # Additional features to ensure consistent count
        # Bet sizing consistency
        df['bet_sizing_consistent'] = (
            (df['small_bet_ratio'] > 0) & (df['medium_bet_ratio'] == 0) & (df['large_bet_ratio'] == 0)
        ).astype(int)
        
        # Position-based betting patterns
        df['late_position_aggressive'] = (
            (df['late_position'] == 1) & (df['aggression_ratio'] > 0.5)
        ).astype(int)
        
        # Stack-based betting patterns
        df['deep_stack_aggressive'] = (
            (df['deep_stack'] == 1) & (df['aggression_ratio'] > 0.3)
        ).astype(int)
        
        # Multi-way pot aggression
        df['multiway_aggressive'] = (
            (df['multi_way'] == 1) & (df['aggression_ratio'] > 0.4)
        ).astype(int)
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("Extracting action patterns...")
        df = self.extract_action_patterns(df)
        
        print("Extracting street features...")
        df = self.extract_street_features(df)
        
        print("Extracting position features...")
        df = self.extract_position_features(df)
        
        print("Extracting pot features...")
        df = self.extract_pot_features(df)
        
        print("Extracting temporal features...")
        df = self.extract_temporal_features(df)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Extracting sequence features...")
        df = self.extract_sequence_features(df)
        
        # Remove original action columns to reduce dimensionality
        action_cols = [col for col in df.columns if col.startswith('action_')]
        df = df.drop(action_cols, axis=1)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Ensure consistent feature count
        df = self.ensure_feature_count(df, expected_count=113)
        
        # Validate feature count
        self.validate_feature_count(df, expected_count=113)
        
        print(f"Final feature count: {len(df.columns) - 1}")  # -1 for target variable
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'hand_strength', fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for modeling"""
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Scale features - only fit the scaler if requested
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def get_feature_names(self, df: pd.DataFrame, target_col: str = 'hand_strength') -> List[str]:
        """Get list of feature names"""
        return [col for col in df.columns if col != target_col]
    
    def validate_feature_count(self, df: pd.DataFrame, expected_count: int = 113) -> bool:
        """Validate that the engineered dataset has the expected number of features"""
        feature_count = len(df.columns) - 1  # -1 for target variable
        if feature_count != expected_count:
            print(f"⚠️  Feature count mismatch: Expected {expected_count}, got {feature_count}")
            print(f"Available features: {list(df.columns)}")
            return False
        print(f"✅ Feature count validated: {feature_count} features")
        return True
    
    def ensure_feature_count(self, df: pd.DataFrame, expected_count: int = 113) -> pd.DataFrame:
        """Ensure the dataset has exactly the expected number of features"""
        feature_count = len(df.columns) - 1  # -1 for target variable
        
        if feature_count < expected_count:
            # Add dummy features to match expected count
            missing_features = expected_count - feature_count
            print(f"Adding {missing_features} dummy features to match expected count")
            
            for i in range(missing_features):
                df[f'dummy_feature_{i}'] = 0
                
        elif feature_count > expected_count:
            # Remove extra features to match expected count
            extra_features = feature_count - expected_count
            print(f"Removing {extra_features} extra features to match expected count")
            
            # Keep only the first expected_count features (excluding target)
            target_col = 'hand_strength'
            feature_cols = [col for col in df.columns if col != target_col][:expected_count]
            df = df[feature_cols + [target_col]]
        
        return df

if __name__ == "__main__":
    # Test the feature engineering
    from data_generator import PokerDataGenerator
    
    # Generate sample data
    generator = PokerDataGenerator(num_hands=1000)
    df = generator.generate_dataset()
    
    # Apply feature engineering
    engineer = PokerFeatureEngineer()
    df_engineered = engineer.engineer_all_features(df)
    
    print(f"Original features: {len([col for col in df.columns if col.startswith('action_')])}")
    print(f"Engineered features: {len(df_engineered.columns) - 1}")
    print(f"Sample engineered features: {list(df_engineered.columns[:20])}")
