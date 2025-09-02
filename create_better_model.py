#!/usr/bin/env python3
"""
Create a better model with more variation in the training data
"""

from data_generator import PokerDataGenerator
from feature_engineering import PokerFeatureEngineer
from range_classifier import train_range_classifier
import numpy as np

def create_better_model():
    """Create a model with more diverse training data"""
    print("🎯 Creating a better model with diverse training data...")
    
    # Generate a larger, more diverse dataset
    generator = PokerDataGenerator(num_hands=5000)  # 5x more data
    df = generator.generate_dataset()
    
    print(f"📊 Generated {len(df)} hands")
    print(f"📈 Hand strength distribution:")
    print(df['hand_strength'].value_counts().sort_index())
    
    # Feature engineering
    engineer = PokerFeatureEngineer()
    df_engineered = engineer.engineer_all_features(df)
    
    # Ensure consistent feature count
    df_engineered = engineer.ensure_feature_count(df_engineered, expected_count=113)
    
    # Prepare features
    X, y = engineer.prepare_features(df_engineered)
    feature_names = engineer.get_feature_names(df_engineered)
    
    print(f"🔧 Engineered {len(feature_names)} features")
    print(f"📊 Feature matrix shape: {X.shape}")
    
    # Train model with better parameters
    print("\n🚀 Training improved model...")
    classifier, results = train_range_classifier(X, y, feature_names)
    
    print(f"\n✅ Training complete!")
    print(f"📊 Best model accuracy: {results['accuracy']:.3f}")
    print(f"🏆 Best model type: RandomForest")  # Fixed the key error
    
    # Save the improved model
    classifier.save_model('poker_range_classifier_improved.pkl')
    print(f"💾 Model saved as 'poker_range_classifier_improved.pkl'")
    
    # Test the model with different inputs
    print("\n🧪 Testing model with different inputs...")
    
    # Test case 1: Aggressive actions
    test_input_1 = create_test_hand(position=6, pot_size=200, stack_size=1500, num_players=4, 
                                   actions=['RAISE_LARGE', 'BET_LARGE'])
    
    # Test case 2: Passive actions  
    test_input_2 = create_test_hand(position=0, pot_size=50, stack_size=800, num_players=6,
                                   actions=['CHECK', 'CALL'])
    
    # Test case 3: Mixed actions
    test_input_3 = create_test_hand(position=3, pot_size=150, stack_size=1200, num_players=5,
                                   actions=['RAISE_MEDIUM', 'CHECK', 'BET_SMALL'])
    
    test_cases = [test_input_1, test_input_2, test_input_3]
    test_names = ['Aggressive', 'Passive', 'Mixed']
    
    for i, (test_input, test_name) in enumerate(zip(test_cases, test_names)):
        print(f"\n📋 Test {i+1}: {test_name} Actions")
        print(f"   Position: {test_input['preflop_position']}")
        print(f"   Pot: ${test_input['preflop_pot_size']}")
        print(f"   Stack: ${test_input['preflop_stack_size']}")
        print(f"   Players: {test_input['preflop_num_players']}")
        print(f"   Actions: {[a for a in test_input['actions'] if a != -1]}")
        
        # Create DataFrame and engineer features
        df_test = engineer.engineer_all_features(test_input)
        X_test, _ = engineer.prepare_features(df_test)
        
        # Make prediction
        prediction = classifier.predict_single_hand(X_test[0])
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {prediction.get('confidence', 'N/A')}")

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
    create_better_model()
