#!/usr/bin/env python3
"""
Test the fixed model to see if it now gives different predictions for different inputs
"""

from range_classifier import PokerRangeClassifier
from feature_engineering import PokerFeatureEngineer
import pandas as pd

def test_fixed_model():
    """Test the fixed model with different inputs"""
    print("🧪 Testing fixed model...")
    
    # Load the fixed model
    try:
        classifier = PokerRangeClassifier()
        classifier.load_model('poker_range_classifier_fixed.pkl')
        print("✅ Fixed model loaded successfully")
    except:
        print("❌ Could not load fixed model, using original")
        classifier = PokerRangeClassifier()
        classifier.load_model('poker_range_classifier.pkl')
    
    # Create test cases with very different inputs
    test_cases = [
        {
            'name': 'Very Aggressive',
            'position': 6,  # BTN
            'pot_size': 1000,
            'stack_size': 5000,
            'num_players': 2,
            'actions': ['RAISE_LARGE', 'BET_LARGE', 'ALL_IN']
        },
        {
            'name': 'Very Passive',
            'position': 0,  # UTG
            'pot_size': 25,
            'stack_size': 400,
            'num_players': 9,
            'actions': ['CHECK', 'CHECK', 'CHECK', 'CHECK']
        },
        {
            'name': 'Mixed Strategy',
            'position': 4,  # HJ
            'pot_size': 300,
            'stack_size': 1500,
            'num_players': 4,
            'actions': ['RAISE_MEDIUM', 'CHECK', 'BET_SMALL', 'CALL']
        },
        {
            'name': 'Early Fold',
            'position': 1,  # UTG+1
            'pot_size': 100,
            'stack_size': 1000,
            'num_players': 7,
            'actions': ['FOLD']
        }
    ]
    
    print("\n" + "="*60)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test {i+1}: {test_case['name']}")
        print(f"   Position: {test_case['position']}")
        print(f"   Pot: ${test_case['pot_size']}")
        print(f"   Stack: ${test_case['stack_size']}")
        print(f"   Players: {test_case['num_players']}")
        print(f"   Actions: {test_case['actions']}")
        
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
        
        # Feature engineering
        engineer = PokerFeatureEngineer()
        df_engineered = engineer.engineer_all_features(df)
        df_engineered = engineer.ensure_feature_count(df_engineered, expected_count=113)
        
        # Prepare features
        X, _ = engineer.prepare_features(df_engineered)
        
        # Make prediction
        prediction = classifier.predict_single_hand(X[0])
        
        print(f"   Prediction: {prediction.get('predicted_class', 'Unknown')}")
        print(f"   Confidence: {prediction.get('confidence', 'N/A'):.3f}")
        print(f"   Probabilities:")
        probs = prediction.get('probabilities', {})
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            print(f"     {class_name}: {prob:.3f}")
        
        print("-" * 40)

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
    test_fixed_model()
