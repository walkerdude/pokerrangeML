#!/usr/bin/env python3
"""
Create a working model that properly responds to different inputs
"""

from data_generator import PokerDataGenerator
from feature_engineering import PokerFeatureEngineer
from range_classifier import train_range_classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def create_working_model():
    """Create a model that actually works and responds to different inputs"""
    print("🔧 Creating a working model...")
    
    # Generate a large, diverse dataset
    print("📊 Generating diverse training data...")
    generator = PokerDataGenerator(num_hands=15000)  # 15k hands for better variety
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
    for i in range(min(5, X.shape[1])):
        feature_name = feature_names[i]
        feature_values = X[:, i]
        print(f"   {feature_name}: min={feature_values.min():.4f}, max={feature_values.max():.4f}, std={feature_values.std():.4f}")
    
    # Split data properly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n📊 Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Train a simple but effective RandomForest
    print("\n🚀 Training RandomForest model...")
    
    # Use simpler parameters to avoid overfitting
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Evaluate on training set
    train_pred = rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"📊 Training accuracy: {train_accuracy:.4f}")
    
    # Evaluate on test set
    test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"🧪 Test accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['Bluff', 'Strong', 'Marginal', 'Nuts']))
    
    # Test the model with different inputs to ensure it responds differently
    print(f"\n🧪 Testing model responsiveness...")
    test_model_responsiveness(rf_model, engineer, feature_names)
    
    # Save the working model
    model_data = {
        'model': rf_model,
        'feature_names': feature_names,
        'feature_engineer': engineer,
        'class_names': ['Bluff', 'Strong', 'Marginal', 'Nuts']
    }
    
    joblib.dump(model_data, 'poker_range_classifier_working.pkl')
    print(f"💾 Working model saved as 'poker_range_classifier_working.pkl'")
    
    return rf_model, engineer, feature_names

def test_model_responsiveness(model, engineer, feature_names):
    """Test if the model actually responds differently to different inputs"""
    
    test_cases = [
        {
            'name': 'Aggressive Large Bet',
            'position': 6, 'pot_size': 800, 'stack_size': 3000, 'num_players': 2,
            'actions': ['RAISE_LARGE', 'BET_LARGE', 'ALL_IN']
        },
        {
            'name': 'Passive Check',
            'position': 0, 'pot_size': 50, 'stack_size': 600, 'num_players': 8,
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
        
        # Create test data using the same method as the web app
        hand_data = create_realistic_test_hand(
            test_case['position'], test_case['pot_size'],
            test_case['stack_size'], test_case['num_players'], test_case['actions']
        )
        
        # Create DataFrame
        df = pd.DataFrame([hand_data])
        
        # Feature engineering
        df_engineered = engineer.engineer_all_features(df)
        df_engineered = engineer.ensure_feature_count(df_engineered, expected_count=113)
        
        # Prepare features
        X_test, _ = engineer.prepare_features(df_engineered)
        
        # Make prediction
        prediction = model.predict(X_test.reshape(1, -1))[0]
        probabilities = model.predict_proba(X_test.reshape(1, -1))[0]
        
        print(f"   Input: Position {test_case['position']}, Pot ${test_case['pot_size']}, Actions: {test_case['actions']}")
        print(f"   Prediction: {['Bluff', 'Strong', 'Marginal', 'Nuts'][prediction]}")
        print(f"   Probabilities:")
        for j, (class_name, prob) in enumerate(zip(['Bluff', 'Strong', 'Marginal', 'Nuts'], probabilities)):
            print(f"     {class_name}: {prob:.3f}")
        
        print("-" * 40)

def create_realistic_test_hand(position, pot_size, stack_size, num_players, actions):
    """Create a realistic test hand that matches training data structure"""
    
    # Generate a base hand from the data generator
    generator = PokerDataGenerator(num_hands=1)
    base_df = generator.generate_dataset()
    hand_data = base_df.iloc[0].copy()
    
    # Update with user inputs
    hand_data['preflop_position'] = position
    hand_data['preflop_pot_size'] = pot_size
    hand_data['preflop_stack_size'] = stack_size
    hand_data['preflop_num_players'] = num_players
    
    # Update all street positions and pot/stack info
    for street in ['flop', 'turn', 'river']:
        hand_data[f'{street}_position'] = position
        hand_data[f'{street}_pot_size'] = pot_size + (pot_size * (0.3 if street == 'flop' else 0.6 if street == 'turn' else 0.9))
        hand_data[f'{street}_stack_size'] = stack_size - (pot_size * (0.2 if street == 'flop' else 0.4 if street == 'turn' else 0.6))
        hand_data[f'{street}_num_players'] = max(2, num_players - (1 if street == 'flop' else 2 if street == 'turn' else 3))
    
    # Update actions
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
    
    # Update hand strength based on actions
    if 'FOLD' in actions:
        hand_data['hand_strength'] = 0  # Bluff
    elif 'ALL_IN' in actions or 'RAISE_LARGE' in actions:
        hand_data['hand_strength'] = 3  # Nuts
    elif 'BET_LARGE' in actions or 'RAISE_MEDIUM' in actions:
        hand_data['hand_strength'] = 1  # Strong
    elif 'CHECK' in actions and 'CALL' in actions:
        hand_data['hand_strength'] = 2  # Marginal
    else:
        hand_data['hand_strength'] = 2  # Default to marginal
    
    return hand_data

if __name__ == "__main__":
    create_working_model()
