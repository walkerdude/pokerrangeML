#!/usr/bin/env python3
"""
Create a proper working model that saves and loads correctly
"""

from data_generator import PokerDataGenerator
from feature_engineering import PokerFeatureEngineer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def create_proper_working_model():
    """Create a model that saves and loads correctly"""
    print("🔧 Creating a proper working model...")
    
    # Generate training data
    print("📊 Generating training data...")
    generator = PokerDataGenerator(num_hands=3000)
    df = generator.generate_dataset()
    
    print(f"📈 Generated {len(df)} hands")
    print(f"📊 Hand strength distribution:")
    print(df['hand_strength'].value_counts().sort_index())
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    engineer = PokerFeatureEngineer()
    df_engineered = engineer.engineer_all_features(df)
    df_engineered = engineer.ensure_feature_count(df_engineered, expected_count=113)
    
    # Prepare features and fit scaler
    X, y = engineer.prepare_features(df_engineered, fit_scaler=True)
    feature_names = engineer.get_feature_names(df_engineered)
    
    print(f"✅ Engineered {len(feature_names)} features")
    print(f"📊 Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    print("\n🚀 Training RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"📊 Training accuracy: {train_accuracy:.4f}")
    print(f"🧪 Test accuracy: {test_accuracy:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['Bluff', 'Strong', 'Marginal', 'Nuts']))
    
    # Test responsiveness
    print(f"\n🧪 Testing model responsiveness...")
    test_responsiveness(rf_model, engineer, feature_names)
    
    # Save the complete model package
    model_package = {
        'model': rf_model,
        'feature_names': feature_names,
        'feature_engineer': engineer,
        'class_names': ['Bluff', 'Strong', 'Marginal', 'Nuts'],
        'scaler': engineer.scaler
    }
    
    joblib.dump(model_package, 'poker_range_classifier_proper.pkl')
    print(f"💾 Proper model saved as 'poker_range_classifier_proper.pkl'")
    
    return rf_model, engineer, feature_names

def test_responsiveness(model, engineer, feature_names):
    """Test if the model responds differently to different inputs"""
    
    test_cases = [
        {
            'name': 'Very Aggressive',
            'position': 6, 'pot_size': 1000, 'stack_size': 5000, 'num_players': 2,
            'actions': ['RAISE_LARGE', 'BET_LARGE', 'ALL_IN']
        },
        {
            'name': 'Very Passive',
            'position': 0, 'pot_size': 50, 'stack_size': 400, 'num_players': 9,
            'actions': ['CHECK', 'CHECK', 'CHECK']
        },
        {
            'name': 'Mixed Strategy',
            'position': 3, 'pot_size': 200, 'stack_size': 1200, 'num_players': 5,
            'actions': ['RAISE_MEDIUM', 'CHECK', 'BET_SMALL']
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
        
        df = pd.DataFrame([hand_data])
        df_engineered = engineer.engineer_all_features(df)
        df_engineered = engineer.ensure_feature_count(df_engineered, expected_count=113)
        
        X_test, _ = engineer.prepare_features(df_engineered, fit_scaler=False)
        
        # Make prediction
        prediction = model.predict(X_test.reshape(1, -1))[0]
        probabilities = model.predict_proba(X_test.reshape(1, -1))[0]
        
        print(f"   Input: Position {test_case['position']}, Pot ${test_case['pot_size']}, Actions: {test_case['actions']}")
        print(f"   Prediction: {['Bluff', 'Strong', 'Marginal', 'Nuts'][prediction]}")
        print(f"   Probabilities:")
        for j, (class_name, prob) in enumerate(zip(['Bluff', 'Strong', 'Marginal', 'Nuts'], probabilities)):
            print(f"     {class_name}: {prob:.3f}")
        
        print("-" * 40)

def create_test_hand(position, pot_size, stack_size, num_players, actions):
    """Create a test hand"""
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
    
    # Add actions
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
    create_proper_working_model()
