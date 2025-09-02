#!/usr/bin/env python3
"""
Deep debug of feature engineering to find why different inputs produce identical features
"""

from feature_engineering import PokerFeatureEngineer
import pandas as pd
import numpy as np

def debug_feature_engineering_deep():
    """Deep debug of feature engineering"""
    print("🔍 Deep debugging feature engineering...")
    
    # Test cases with very different inputs
    test_cases = [
        {
            'name': 'Aggressive',
            'position': 6, 'pot_size': 1000, 'stack_size': 5000, 'num_players': 2,
            'actions': ['RAISE_LARGE', 'BET_LARGE', 'ALL_IN']
        },
        {
            'name': 'Passive',
            'position': 0, 'pot_size': 50, 'stack_size': 400, 'num_players': 9,
            'actions': ['CHECK', 'CHECK', 'CHECK']
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"📋 Test Case {i+1}: {test_case['name']}")
        print(f"{'='*80}")
        
        # Create synthetic hand data
        hand_data = create_test_hand(
            test_case['position'], test_case['pot_size'],
            test_case['stack_size'], test_case['num_players'], test_case['actions']
        )
        
        # Create DataFrame
        df = pd.DataFrame([hand_data])
        
        print(f"📊 Original data shape: {df.shape}")
        print(f"🔢 Key original values:")
        for col in ['preflop_position', 'preflop_pot_size', 'preflop_stack_size', 'preflop_num_players']:
            print(f"   {col}: {df[col].iloc[0]}")
        
        print(f"🎯 Actions: {test_case['actions']}")
        action_values = [hand_data[f'action_{i}'] for i in range(min(10, len(test_case['actions'])))]
        print(f"📝 Action values: {action_values}")
        
        # Feature engineering step by step
        engineer = PokerFeatureEngineer()
        
        print(f"\n🔧 Step 1: Action patterns...")
        df_action = engineer.extract_action_patterns(df.copy())
        print(f"   Shape after action patterns: {df_action.shape}")
        print(f"   Sample action features:")
        action_cols = [col for col in df_action.columns if 'action' in col.lower() and col not in df.columns]
        for col in action_cols[:5]:
            print(f"     {col}: {df_action[col].iloc[0]}")
        
        print(f"\n🔧 Step 2: Street features...")
        df_street = engineer.extract_street_features(df_action.copy())
        print(f"   Shape after street features: {df_street.shape}")
        print(f"   Sample street features:")
        street_cols = [col for col in df_street.columns if 'street' in col.lower() and col not in df_action.columns]
        for col in street_cols[:5]:
            print(f"     {col}: {df_street[col].iloc[0]}")
        
        print(f"\n🔧 Step 3: Position features...")
        df_position = engineer.extract_position_features(df_street.copy())
        print(f"   Shape after position features: {df_position.shape}")
        print(f"   Sample position features:")
        position_cols = [col for col in df_position.columns if 'position' in col.lower() and col not in df_street.columns]
        for col in position_cols[:5]:
            print(f"     {col}: {df_position[col].iloc[0]}")
        
        print(f"\n🔧 Step 4: Pot features...")
        df_pot = engineer.extract_pot_features(df_position.copy())
        print(f"   Shape after pot features: {df_pot.shape}")
        print(f"   Sample pot features:")
        pot_cols = [col for col in df_pot.columns if 'pot' in col.lower() and col not in df_position.columns]
        for col in pot_cols[:5]:
            print(f"     {col}: {df_pot[col].iloc[0]}")
        
        print(f"\n🔧 Step 5: Temporal features...")
        df_temporal = engineer.extract_temporal_features(df_pot.copy())
        print(f"   Shape after temporal features: {df_temporal.shape}")
        print(f"   Sample temporal features:")
        temporal_cols = [col for col in df_temporal.columns if 'temporal' in col.lower() and col not in df_pot.columns]
        for col in temporal_cols[:5]:
            print(f"     {col}: {df_temporal[col].iloc[0]}")
        
        print(f"\n🔧 Step 6: Interaction features...")
        df_interaction = engineer.create_interaction_features(df_temporal.copy())
        print(f"   Shape after interaction features: {df_interaction.shape}")
        print(f"   Sample interaction features:")
        interaction_cols = [col for col in df_interaction.columns if col not in df_temporal.columns]
        for col in interaction_cols[:5]:
            print(f"     {col}: {df_interaction[col].iloc[0]}")
        
        print(f"\n🔧 Step 7: Sequence features...")
        df_sequence = engineer.extract_sequence_features(df_interaction.copy())
        print(f"   Shape after sequence features: {df_sequence.shape}")
        print(f"   Sample sequence features:")
        sequence_cols = [col for col in df_sequence.columns if col not in df_interaction.columns]
        for col in sequence_cols[:5]:
            print(f"     {col}: {df_sequence[col].iloc[0]}")
        
        # Final engineered features
        print(f"\n🔧 Final engineered features:")
        print(f"   Total features: {len(df_sequence.columns) - 1}")
        
        # Show key engineered features
        key_features = [
            'aggression_ratio', 'pot_stack_ratio', 'small_bet_ratio',
            'medium_bet_ratio', 'large_bet_ratio', 'early_position',
            'late_position', 'multi_way', 'deep_stack'
        ]
        
        print(f"\n🎯 Key engineered features:")
        for feature in key_features:
            if feature in df_sequence.columns:
                value = df_sequence[feature].iloc[0]
                print(f"   {feature}: {value}")
            else:
                print(f"   {feature}: NOT FOUND")
        
        # Show some raw action features
        action_features = [col for col in df_sequence.columns if 'action_' in col]
        print(f"\n🎮 Raw action features (first 10):")
        for feature in action_features[:10]:
            value = df_sequence[feature].iloc[0]
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
    debug_feature_engineering_deep()
