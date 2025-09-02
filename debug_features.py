#!/usr/bin/env python3
"""
Debug feature engineering to see why different inputs produce identical predictions
"""

from feature_engineering import PokerFeatureEngineer
import pandas as pd
import numpy as np

def debug_features():
    """Debug feature engineering for different inputs"""
    print("🔍 Debugging feature engineering...")
    
    # Test cases with very different inputs
    test_cases = [
        {
            'name': 'Aggressive',
            'position': 6,
            'pot_size': 500,
            'stack_size': 2000,
            'num_players': 3,
            'actions': ['RAISE_LARGE', 'BET_LARGE', 'ALL_IN']
        },
        {
            'name': 'Passive',
            'position': 0,
            'pot_size': 50,
            'stack_size': 800,
            'num_players': 8,
            'actions': ['CHECK', 'CHECK', 'CHECK']
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"📋 Test Case {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        # Create synthetic hand data
        hand_data = create_test_hand(
            test_case['position'],
            test_case['pot_size'],
            test_case['stack_size'],
            test_case['num_players'],
            test_case['actions']
        )
        
        # Create DataFrame
        df = pd.DataFrame([hand_data])
        
        print(f"📊 Original data shape: {df.shape}")
        print(f"🔢 Sample original values:")
        for col in ['preflop_position', 'preflop_pot_size', 'preflop_stack_size', 'preflop_num_players']:
            print(f"   {col}: {df[col].iloc[0]}")
        
        print(f"🎯 Actions: {test_case['actions']}")
        action_values = [hand_data[f'action_{i}'] for i in range(min(10, len(test_case['actions'])))]
        print(f"📝 Action values: {action_values}")
        
        # Feature engineering
        engineer = PokerFeatureEngineer()
        df_engineered = engineer.engineer_all_features(df)
        
        print(f"\n🔧 After feature engineering:")
        print(f"   Shape: {df_engineered.shape}")
        print(f"   Features: {len(df_engineered.columns) - 1}")  # -1 for target
        
        # Show some key engineered features
        key_features = [
            'aggression_ratio', 'pot_stack_ratio', 'small_bet_ratio',
            'medium_bet_ratio', 'large_bet_ratio', 'early_position',
            'late_position', 'multi_way', 'deep_stack'
        ]
        
        print(f"\n🎯 Key engineered features:")
        for feature in key_features:
            if feature in df_engineered.columns:
                value = df_engineered[feature].iloc[0]
                print(f"   {feature}: {value}")
            else:
                print(f"   {feature}: NOT FOUND")
        
        # Show action-related features
        action_features = [col for col in df_engineered.columns if 'action' in col.lower()]
        print(f"\n🎮 Action-related features (first 10):")
        for feature in action_features[:10]:
            value = df_engineered[feature].iloc[0]
            print(f"   {feature}: {value}")
        
        # Show position features
        position_features = [col for col in df_engineered.columns if 'position' in col.lower()]
        print(f"\n📍 Position features:")
        for feature in position_features:
            value = df_engineered[feature].iloc[0]
            print(f"   {feature}: {value}")
        
        # Show pot/stack features
        pot_features = [col for col in df_engineered.columns if 'pot' in col.lower() or 'stack' in col.lower()]
        print(f"\n💰 Pot/Stack features:")
        for feature in pot_features:
            value = df_engineered[feature].iloc[0]
            print(f"   {feature}: {value}")

def create_test_hand(position, pot_size, stack_size, num_players, actions):
    """Create a test hand with the specified parameters"""
    hand_data = {
        'hand_strength': 2,  # Default to marginal
        'preflop_position': position,
        'preflop_pot_size': pot_size,
        'preflop_stack_size': stack_size,
        'preflop_num_players': num_players,
        'flop_position': position,
        'flop_pot_size': pot_size + (pot_size * 0.3),
        'flop_stack_size': stack_size - (pot_size * 0.2),
        'flop_num_players': max(2, num_players - 1),
        'turn_position': position,
        'turn_pot_size': pot_size + (pot_size * 0.6),
        'turn_stack_size': stack_size - (pot_size * 0.4),
        'turn_num_players': max(2, num_players - 2),
        'river_position': position,
        'river_pot_size': pot_size + (pot_size * 0.9),
        'river_stack_size': stack_size - (pot_size * 0.6),
        'river_num_players': max(2, num_players - 3)
    }
    
    # Add action columns
    action_mapping = {
        'FOLD': 0, 'CHECK': 1, 'CALL': 2, 'BET_SMALL': 3, 'BET_MEDIUM': 4,
        'BET_LARGE': 5, 'RAISE_SMALL': 6, 'RAISE_MEDIUM': 7, 'RAISE_LARGE': 8, 'ALL_IN': 9
    }
    
    for i in range(40):
        if i < len(actions):
            action_val = action_mapping.get(actions[i], 0)
        else:
            action_val = -1
        hand_data[f'action_{i}'] = action_val
    
    return hand_data

if __name__ == "__main__":
    debug_features()
