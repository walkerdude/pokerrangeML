#!/usr/bin/env python3
"""
Fix the model by retraining with better feature scaling and more diverse data
"""

from data_generator import PokerDataGenerator
from feature_engineering import PokerFeatureEngineer
from range_classifier import train_range_classifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_fixed_model():
    """Create a fixed model that properly responds to different inputs"""
    print("🔧 Creating a fixed model...")
    
    # Generate a much larger, more diverse dataset
    print("📊 Generating diverse training data...")
    generator = PokerDataGenerator(num_hands=10000)  # 10x more data
    df = generator.generate_dataset()
    
    print(f"📈 Generated {len(df)} hands")
    print(f"📊 Hand strength distribution:")
    print(df['hand_strength'].value_counts().sort_index())
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    engineer = PokerFeatureEngineer()
    df_engineered = engineer.engineer_all_features(df)
    
    # Ensure consistent feature count
    df_engineered = engineer.ensure_feature_count(df_engineered, expected_count=113)
    
    # Prepare features
    X, y = engineer.prepare_features(df_engineered)
    feature_names = engineer.get_feature_names(df_engineered)
    
    print(f"✅ Engineered {len(feature_names)} features")
    print(f"📊 Feature matrix shape: {X.shape}")
    
    # Check feature variation
    print(f"\n📊 Feature variation analysis:")
    for i in range(min(10, X.shape[1])):
        feature_name = feature_names[i]
        feature_values = X[:, i]
        print(f"   {feature_name}: min={feature_values.min():.4f}, max={feature_values.max():.4f}, std={feature_values.std():.4f}")
    
    # Split data properly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n📊 Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Train model with better parameters
    print("\n🚀 Training improved model...")
    classifier, results = train_range_classifier(X_train, y_train, feature_names)
    
    print(f"\n✅ Training complete!")
    print(f"📊 Best model accuracy: {results['accuracy']:.3f}")
    
    # Test on test set
    test_results = classifier.evaluate_model(X_test, y_test)
    if isinstance(test_results, dict):
        test_accuracy = test_results.get('accuracy', 0)
    else:
        test_accuracy = test_results
    print(f"🧪 Test set accuracy: {test_accuracy:.3f}")
    
    # Save the fixed model
    classifier.save_model('poker_range_classifier_fixed.pkl')
    print(f"💾 Fixed model saved as 'poker_range_classifier_fixed.pkl'")
    
    # Test the fixed model with different inputs
    print("\n🧪 Testing fixed model with different inputs...")
    test_diverse_inputs(classifier, engineer, feature_names)

def test_diverse_inputs(classifier, engineer, feature_names):
    """Test the model with very different inputs to ensure it responds differently"""
    
    test_cases = [
        {
            'name': 'Very Aggressive',
            'position': 6, 'pot_size': 1000, 'stack_size': 5000, 'num_players': 2,
            'actions': ['RAISE_LARGE', 'BET_LARGE', 'ALL_IN']
        },
        {
            'name': 'Very Passive',
            'position': 0, 'pot_size': 25, 'stack_size': 400, 'num_players': 9,
            'actions': ['CHECK', 'CHECK', 'CHECK', 'CHECK']
        },
        {
            'name': 'Mixed Strategy',
            'position': 4, 'pot_size': 300, 'stack_size': 1500, 'num_players': 4,
            'actions': ['RAISE_MEDIUM', 'CHECK', 'BET_SMALL', 'CALL']
        }
    ]
    
    print("\n" + "="*60)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test {i+1}: {test_case['name']}")
        
        # Create test data
        hand_data = create_test_hand(
            test_case['position'], test_case['pot_size'],
            test_case['stack_size'], test_case['num_players'], test_case['actions']
        )
        
        df_test = pd.DataFrame([hand_data])
        df_engineered = engineer.engineer_all_features(df_test)
        df_engineered = engineer.ensure_feature_count(df_engineered, expected_count=113)
        
        X_test, _ = engineer.prepare_features(df_engineered)
        
        # Make prediction
        prediction = classifier.predict_single_hand(X_test[0])
        
        print(f"   Input: Position {test_case['position']}, Pot ${test_case['pot_size']}, Actions: {test_case['actions']}")
        print(f"   Prediction: {prediction.get('predicted_class', 'Unknown')}")
        print(f"   Confidence: {prediction.get('confidence', 'N/A'):.3f}")
        
        # Show top probabilities
        probs = prediction.get('probabilities', {})
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top probabilities:")
        for class_name, prob in sorted_probs[:2]:
            print(f"     {class_name}: {prob:.3f}")

def create_test_hand(position, pot_size, stack_size, num_players, actions):
    """Create a test hand with the specified parameters"""
    hand_data = {
        'hand_strength': 2,
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
    create_fixed_model()
