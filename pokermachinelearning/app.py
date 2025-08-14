#!/usr/bin/env python3
"""
Poker Range Classification Web Application
=========================================

A Flask web app that provides a user-friendly interface for
classifying opponent hand ranges based on action history.
"""

from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

# Import our poker modules
from data_generator import PokerDataGenerator, Action, HandStrength
from feature_engineering import PokerFeatureEngineer
from range_classifier import PokerRangeClassifier

app = Flask(__name__)
app.secret_key = 'poker_range_classifier_secret_key'

# Global variables
classifier = None
feature_engineer = None

def load_model():
    """Load the trained model"""
    global classifier, feature_engineer
    
    try:
        classifier = PokerRangeClassifier()
        classifier.load_model('poker_range_classifier.pkl')
        feature_engineer = PokerFeatureEngineer()
        return True
    except FileNotFoundError:
        return False

def create_sample_hand():
    """Create a sample hand for demonstration"""
    generator = PokerDataGenerator(num_hands=1)
    df = generator.generate_dataset()
    
    # Extract action sequence
    actions = []
    for i in range(40):
        action_val = df[f'action_{i}'].iloc[0]
        if action_val != -1:
            action_name = list(Action)[action_val].name
            actions.append(action_name)
    
    # Get other features
    hand_data = {
        'position': df['preflop_position'].iloc[0],
        'pot_size': df['preflop_pot_size'].iloc[0],
        'stack_size': df['preflop_stack_size'].iloc[0],
        'num_players': df['preflop_num_players'].iloc[0],
        'actions': actions[:10],  # First 10 actions
        'hand_strength': HandStrength(df['hand_strength'].iloc[0]).name
    }
    
    return hand_data

@app.route('/')
def index():
    """Main page"""
    model_loaded = load_model()
    sample_hand = create_sample_hand() if model_loaded else None
    
    return render_template('index.html', 
                         model_loaded=model_loaded,
                         sample_hand=sample_hand)

@app.route('/classify', methods=['POST'])
def classify_hand():
    """Classify a hand based on input data"""
    try:
        data = request.get_json()
        
        if not classifier:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
        
        # Extract input data
        position = int(data.get('position', 0))
        pot_size = int(data.get('pot_size', 100))
        stack_size = int(data.get('stack_size', 1000))
        num_players = int(data.get('num_players', 6))
        actions = data.get('actions', [])
        
        # Create a synthetic hand with the provided data
        hand_data = {
            'hand_strength': 2,  # Default to marginal
            'preflop_position': position,
            'preflop_pot_size': pot_size,
            'preflop_stack_size': stack_size,
            'preflop_num_players': num_players,
            'flop_position': position,
            'flop_pot_size': pot_size + 50,
            'flop_stack_size': stack_size - 50,
            'flop_num_players': num_players,
            'turn_position': position,
            'turn_pot_size': pot_size + 100,
            'turn_stack_size': stack_size - 100,
            'turn_num_players': num_players,
            'river_position': position,
            'river_pot_size': pot_size + 150,
            'river_stack_size': stack_size - 150,
            'river_num_players': num_players
        }
        
        # Add action columns (40 total)
        for i in range(40):
            if i < len(actions):
                # Convert action name to value
                try:
                    action_val = next(j for j, action in enumerate(Action) if action.name == actions[i])
                except StopIteration:
                    action_val = 0
            else:
                action_val = -1  # Padding
            hand_data[f'action_{i}'] = action_val
        
        # Create DataFrame
        df = pd.DataFrame([hand_data])
        
        # Feature engineering
        df_engineered = feature_engineer.engineer_all_features(df)
        
        # Prepare features
        X, _ = feature_engineer.prepare_features(df_engineered)
        
        # Make prediction
        prediction = classifier.predict_single_hand(X[0])
        
        # Get feature importance for this prediction
        importance_df = classifier.get_feature_importance()
        top_features = importance_df.head(10).to_dict('records') if importance_df is not None else []
        
        return jsonify({
            'prediction': prediction,
            'top_features': top_features,
            'input_data': {
                'position': position,
                'pot_size': pot_size,
                'stack_size': stack_size,
                'num_players': num_players,
                'actions': actions
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train a new model"""
    try:
        data = request.get_json()
        num_hands = int(data.get('num_hands', 1000))
        
        # Generate data
        generator = PokerDataGenerator(num_hands=num_hands)
        df = generator.generate_dataset()
        
        # Feature engineering
        engineer = PokerFeatureEngineer()
        df_engineered = engineer.engineer_all_features(df)
        
        # Prepare features
        X, y = engineer.prepare_features(df_engineered)
        feature_names = engineer.get_feature_names(df_engineered)
        
        # Train model
        from range_classifier import train_range_classifier
        classifier_trained, results = train_range_classifier(X, y, feature_names)
        
        # Save model
        classifier_trained.save_model('poker_range_classifier.pkl')
        
        # Update global variables
        global classifier, feature_engineer
        classifier = classifier_trained
        feature_engineer = engineer
        
        return jsonify({
            'success': True,
            'accuracy': results['accuracy'],
            'num_hands': num_hands,
            'num_features': len(feature_names)
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/demo')
def demo():
    """Demo page with sample hands"""
    model_loaded = load_model()
    
    if not model_loaded:
        return render_template('demo.html', model_loaded=False)
    
    # Generate sample hands
    generator = PokerDataGenerator(num_hands=5)
    df = generator.generate_dataset()
    
    sample_hands = []
    for i in range(5):
        hand = {
            'id': i + 1,
            'position': df['preflop_position'].iloc[i],
            'pot_size': df['preflop_pot_size'].iloc[i],
            'stack_size': df['preflop_stack_size'].iloc[i],
            'num_players': df['preflop_num_players'].iloc[i],
            'true_strength': HandStrength(df['hand_strength'].iloc[i]).name
        }
        
        # Extract actions
        actions = []
        for j in range(40):
            action_val = df[f'action_{j}'].iloc[i]
            if action_val != -1:
                action_name = list(Action)[action_val].name
                actions.append(action_name)
        hand['actions'] = actions[:10]  # First 10 actions
        
        sample_hands.append(hand)
    
    return render_template('demo.html', 
                         model_loaded=True,
                         sample_hands=sample_hands)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/actions')
def get_actions():
    """Get available actions"""
    actions = [action.name for action in Action]
    return jsonify(actions)

@app.route('/api/positions')
def get_positions():
    """Get position descriptions"""
    positions = [
        {'value': 0, 'name': 'UTG (Under the Gun)'},
        {'value': 1, 'name': 'UTG+1'},
        {'value': 2, 'name': 'UTG+2'},
        {'value': 3, 'name': 'LJ (Lojack)'},
        {'value': 4, 'name': 'HJ (Hijack)'},
        {'value': 5, 'name': 'CO (Cutoff)'},
        {'value': 6, 'name': 'BTN (Button)'},
        {'value': 7, 'name': 'SB (Small Blind)'},
        {'value': 8, 'name': 'BB (Big Blind)'}
    ]
    return jsonify(positions)

if __name__ == '__main__':
    # Try to load model on startup
    model_loaded = load_model()
    if model_loaded:
        print("✓ Model loaded successfully")
    else:
        print("⚠ No trained model found. Please train a model first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
