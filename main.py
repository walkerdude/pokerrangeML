#!/usr/bin/env python3
"""
Poker Range Classification System
================================

This script provides a complete pipeline for training a machine learning model
to classify opponent hand ranges into four categories: Nuts, Strong, Marginal, and Bluff.

Usage:
    python main.py [--generate-data] [--train] [--evaluate] [--demo]

Options:
    --generate-data: Generate new training data
    --train: Train the range classification model
    --evaluate: Evaluate the trained model
    --demo: Run a demonstration with sample predictions
"""

import argparse
import sys
import os
from typing import Dict, Any
import numpy as np
import pandas as pd

# Import our modules
from data_generator import PokerDataGenerator
from feature_engineering import PokerFeatureEngineer
from range_classifier import PokerRangeClassifier, train_range_classifier
from evaluation import PokerRangeEvaluator

def generate_training_data(num_hands: int = 10000, filename: str = "poker_range_data.csv"):
    """Generate training data for the range classification model"""
    
    print("=" * 60)
    print("GENERATING POKER RANGE CLASSIFICATION TRAINING DATA")
    print("=" * 60)
    
    generator = PokerDataGenerator(num_hands=num_hands)
    df = generator.save_dataset(filename)
    
    print(f"\nData generation complete!")
    print(f"Generated {len(df)} poker hands")
    print(f"Hand strength distribution:")
    print(df['hand_strength'].value_counts().sort_index())
    
    return df

def train_model(data_file: str = "poker_range_data.csv", 
               model_file: str = "poker_range_classifier.pkl",
               test_size: float = 0.2):
    """Train the range classification model"""
    
    print("=" * 60)
    print("TRAINING POKER RANGE CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Load data
    print("Loading training data...")
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Generating new data...")
        df = generate_training_data(10000, data_file)
    else:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} hands from {data_file}")
    
    # Feature engineering
    print("\nEngineering features...")
    engineer = PokerFeatureEngineer()
    df_engineered = engineer.engineer_all_features(df)
    
    # Prepare features
    X, y = engineer.prepare_features(df_engineered)
    feature_names = engineer.get_feature_names(df_engineered)
    
    print(f"Feature engineering complete!")
    print(f"Total features: {len(feature_names)}")
    print(f"Feature examples: {feature_names[:10]}")
    
    # Train model
    print("\nTraining range classification model...")
    classifier, results = train_range_classifier(X, y, feature_names, test_size=test_size)
    
    # Save model
    classifier.save_model(model_file)
    
    print(f"\nTraining complete!")
    print(f"Best model: {classifier.best_model_name}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    
    return classifier, results

def evaluate_model(model_file: str = "poker_range_classifier.pkl",
                  data_file: str = "poker_range_data.csv"):
    """Evaluate the trained model"""
    
    print("=" * 60)
    print("EVALUATING POKER RANGE CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Load model
    print("Loading trained model...")
    classifier = PokerRangeClassifier()
    classifier.load_model(model_file)
    
    # Load data for evaluation
    print("Loading evaluation data...")
    df = pd.read_csv(data_file)
    
    # Feature engineering
    print("Engineering features for evaluation...")
    engineer = PokerFeatureEngineer()
    df_engineered = engineer.engineer_all_features(df)
    
    # Prepare features
    X, y = engineer.prepare_features(df_engineered)
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create evaluator
    evaluator = PokerRangeEvaluator(classifier)
    
    # Perform comprehensive evaluation
    print("Performing comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(X_test, y_test)
    
    # Generate visualizations and analysis
    print("Generating evaluation visualizations...")
    evaluator.plot_comprehensive_results(results)
    evaluator.analyze_prediction_confidence(results)
    evaluator.feature_importance_analysis()
    evaluator.error_analysis(X_test, y_test, classifier.feature_names)
    
    # Generate evaluation report
    evaluator.generate_evaluation_report(results)
    
    print(f"\nEvaluation complete!")
    print(f"Overall accuracy: {results['accuracy']:.4f}")
    
    return results

def run_demo(model_file: str = "poker_range_classifier.pkl"):
    """Run a demonstration of the range classification model"""
    
    print("=" * 60)
    print("POKER RANGE CLASSIFICATION DEMONSTRATION")
    print("=" * 60)
    
    # Load model
    try:
        classifier = PokerRangeClassifier()
        classifier.load_model(model_file)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return
    
    # Create sample hands for demonstration
    print("\nDemonstrating range classification with sample hands...")
    
    # Generate sample data
    generator = PokerDataGenerator(num_hands=10)
    sample_df = generator.generate_dataset()
    
    # Feature engineering
    engineer = PokerFeatureEngineer()
    sample_df_engineered = engineer.engineer_all_features(sample_df)
    
    # Prepare features
    X_sample, y_sample = engineer.prepare_features(sample_df_engineered)
    
    # Make predictions
    predictions, probabilities = classifier.predict_range(X_sample)
    
    print("\nSample Hand Classifications:")
    print("-" * 80)
    print(f"{'Hand':<4} {'True':<8} {'Predicted':<10} {'Confidence':<12} {'Probabilities'}")
    print("-" * 80)
    
    for i in range(len(predictions)):
        true_class = classifier.class_names[y_sample[i]]
        pred_class = classifier.class_names[predictions[i]]
        confidence = np.max(probabilities[i])
        
        # Format probabilities
        prob_str = " | ".join([
            f"{cls}: {prob:.2f}" for cls, prob in zip(classifier.class_names, probabilities[i])
        ])
        
        print(f"{i+1:<4} {true_class:<8} {pred_class:<10} {confidence:<12.3f} {prob_str}")
    
    # Show feature importance
    print(f"\nTop 10 Most Important Features:")
    print("-" * 50)
    importance_df = classifier.get_feature_importance()
    if importance_df is not None:
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    print(f"\nDemonstration complete!")

def main():
    """Main function to run the poker range classification system"""
    
    parser = argparse.ArgumentParser(
        description="Poker Range Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --generate-data          # Generate new training data
  python main.py --train                  # Train the model
  python main.py --evaluate               # Evaluate the model
  python main.py --demo                   # Run demonstration
  python main.py --generate-data --train  # Generate data and train
  python main.py --train --evaluate       # Train and evaluate
        """
    )
    
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate new training data')
    parser.add_argument('--train', action='store_true',
                       help='Train the range classification model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the trained model')
    parser.add_argument('--demo', action='store_true',
                       help='Run a demonstration with sample predictions')
    parser.add_argument('--num-hands', type=int, default=10000,
                       help='Number of hands to generate (default: 10000)')
    parser.add_argument('--data-file', type=str, default='poker_range_data.csv',
                       help='Data file path (default: poker_range_data.csv)')
    parser.add_argument('--model-file', type=str, default='poker_range_classifier.pkl',
                       help='Model file path (default: poker_range_classifier.pkl)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        # Generate data if requested
        if args.generate_data:
            generate_training_data(args.num_hands, args.data_file)
        
        # Train model if requested
        if args.train:
            train_model(args.data_file, args.model_file)
        
        # Evaluate model if requested
        if args.evaluate:
            evaluate_model(args.model_file, args.data_file)
        
        # Run demo if requested
        if args.demo:
            run_demo(args.model_file)
        
        print("\n" + "=" * 60)
        print("POKER RANGE CLASSIFICATION SYSTEM COMPLETE")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
