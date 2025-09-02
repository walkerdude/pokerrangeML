#!/usr/bin/env python3
"""
Test script for the Poker Range Classification System
"""

import sys
import os
import numpy as np
import pandas as pd

def test_data_generation():
    """Test data generation functionality"""
    print("Testing data generation...")
    
    try:
        from data_generator import PokerDataGenerator
        
        # Generate small dataset
        generator = PokerDataGenerator(num_hands=100)
        df = generator.generate_dataset()
        
        # Check basic structure
        assert len(df) == 100, f"Expected 100 hands, got {len(df)}"
        assert 'hand_strength' in df.columns, "Missing hand_strength column"
        assert df['hand_strength'].min() >= 0, "Invalid hand strength values"
        assert df['hand_strength'].max() <= 3, "Invalid hand strength values"
        
        # Check action columns (40 total: 10 per street * 4 streets)
        action_cols = [col for col in df.columns if col.startswith('action_')]
        assert len(action_cols) == 40, f"Expected 40 action columns, got {len(action_cols)}"
        
        print("✓ Data generation test passed")
        return df
        
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return None

def test_feature_engineering(df):
    """Test feature engineering functionality"""
    print("Testing feature engineering...")
    
    try:
        from feature_engineering import PokerFeatureEngineer
        
        engineer = PokerFeatureEngineer()
        df_engineered = engineer.engineer_all_features(df)
        
        # Check that features were added (accounting for action columns being removed)
        original_action_cols = len([col for col in df.columns if col.startswith('action_')])
        non_action_cols = len(df.columns) - original_action_cols
        print(f"Original columns: {len(df.columns)}, Action columns: {original_action_cols}, Non-action: {non_action_cols}")
        print(f"Engineered columns: {len(df_engineered.columns)}")
        # Should have engineered features (action columns are removed and replaced)
        # The engineered dataset should have a reasonable number of features
        assert len(df_engineered.columns) >= 100, f"Expected at least 100 features, got {len(df_engineered.columns)}"
        
        # Check for specific engineered features
        expected_features = ['aggression_ratio', 'pot_stack_ratio', 'small_bet_ratio']
        for feature in expected_features:
            assert feature in df_engineered.columns, f"Missing feature: {feature}"
        
        # Test feature preparation
        X, y = engineer.prepare_features(df_engineered)
        assert X.shape[0] == len(df), "Feature matrix size mismatch"
        assert len(y) == len(df), "Target vector size mismatch"
        
        print("✓ Feature engineering test passed")
        return df_engineered, X, y
        
    except Exception as e:
        print(f"✗ Feature engineering test failed: {e}")
        return None, None, None

def test_model_training(X, y, feature_names):
    """Test model training functionality"""
    print("Testing model training...")
    
    try:
        from range_classifier import PokerRangeClassifier
        
        # Create classifier
        classifier = PokerRangeClassifier()
        classifier.feature_names = feature_names
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train models (with smaller dataset for testing)
        results = classifier.train_models(X_train, y_train, X_test, y_test)
        
        # Check that models were trained
        assert len(classifier.models) > 0, "No models were trained"
        assert classifier.best_model is not None, "No best model selected"
        
        # Test predictions
        predictions, probabilities = classifier.predict_range(X_test)
        assert len(predictions) == len(y_test), "Prediction size mismatch"
        assert probabilities.shape[1] == 4, "Expected 4 class probabilities"
        
        print("✓ Model training test passed")
        return classifier, X_test, y_test
        
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return None, None, None

def test_evaluation(classifier, X_test, y_test):
    """Test evaluation functionality"""
    print("Testing model evaluation...")
    
    try:
        from evaluation import PokerRangeEvaluator
        
        evaluator = PokerRangeEvaluator(classifier)
        
        # Test comprehensive evaluation
        results = evaluator.comprehensive_evaluation(X_test, y_test)
        
        # Check evaluation results
        assert 'accuracy' in results, "Missing accuracy in results"
        assert 'confusion_matrix' in results, "Missing confusion matrix"
        assert 'class_metrics' in results, "Missing class metrics"
        
        # Check accuracy is reasonable
        assert 0 <= results['accuracy'] <= 1, "Invalid accuracy value"
        
        print("✓ Model evaluation test passed")
        return results
        
    except Exception as e:
        print(f"✗ Model evaluation test failed: {e}")
        return None

def test_prediction_demo(classifier):
    """Test prediction demonstration"""
    print("Testing prediction demo...")
    
    try:
        # Generate sample data
        from data_generator import PokerDataGenerator
        from feature_engineering import PokerFeatureEngineer
        
        generator = PokerDataGenerator(num_hands=5)
        sample_df = generator.generate_dataset()
        
        engineer = PokerFeatureEngineer()
        sample_df_engineered = engineer.engineer_all_features(sample_df)
        X_sample, y_sample = engineer.prepare_features(sample_df_engineered)
        
        # Make predictions
        predictions, probabilities = classifier.predict_range(X_sample)
        
        # Test single hand prediction
        single_prediction = classifier.predict_single_hand(X_sample[0])
        
        assert 'predicted_class' in single_prediction, "Missing predicted class"
        assert 'probabilities' in single_prediction, "Missing probabilities"
        assert 'confidence' in single_prediction, "Missing confidence"
        
        print("✓ Prediction demo test passed")
        return True
        
    except Exception as e:
        print(f"✗ Prediction demo test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("POKER RANGE CLASSIFICATION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    # Test 1: Data Generation
    df = test_data_generation()
    if df is None:
        print("❌ Stopping tests due to data generation failure")
        return False
    
    # Test 2: Feature Engineering
    df_engineered, X, y = test_feature_engineering(df)
    if X is None:
        print("❌ Stopping tests due to feature engineering failure")
        return False
    
    # Test 3: Model Training
    feature_names = [col for col in df_engineered.columns if col != 'hand_strength']
    classifier, X_test, y_test = test_model_training(X, y, feature_names)
    if classifier is None:
        print("❌ Stopping tests due to model training failure")
        return False
    
    # Test 4: Model Evaluation
    results = test_evaluation(classifier, X_test, y_test)
    if results is None:
        print("❌ Stopping tests due to evaluation failure")
        return False
    
    # Test 5: Prediction Demo
    demo_success = test_prediction_demo(classifier)
    if not demo_success:
        print("❌ Prediction demo test failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 60)
    print("\nSystem is ready for use!")
    print("\nNext steps:")
    print("1. Generate more data: python main.py --generate-data --num-hands 10000")
    print("2. Train full model: python main.py --train")
    print("3. Evaluate model: python main.py --evaluate")
    print("4. Run demo: python main.py --demo")
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
