#!/usr/bin/env python3
"""
Test script for the Poker Range Classifier web application
"""

import requests
import json
import time

def test_webapp():
    """Test the web application endpoints"""
    base_url = "http://localhost:8080"
    
    print("🧪 Testing Poker Range Classifier Web App")
    print("=" * 50)
    
    # Test 1: Home page
    print("\n1. Testing home page...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Home page loaded successfully")
        else:
            print(f"❌ Home page failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Home page error: {e}")
    
    # Test 2: Train model
    print("\n2. Testing model training...")
    try:
        train_data = {"num_hands": 1000}  # Increased from 100 to 1000
        response = requests.post(f"{base_url}/train", json=train_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model trained successfully")
            print(f"   Accuracy: {result.get('accuracy', 'N/A')}")
            print(f"   Features: {result.get('num_features', 'N/A')}")
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Training error: {e}")
    
    # Test 3: Classify hand
    print("\n3. Testing hand classification...")
    try:
        classify_data = {
            "position": 6,  # BTN
            "pot_size": 150,
            "stack_size": 1200,
            "num_players": 6,
            "actions": ["RAISE_MEDIUM", "BET_MEDIUM", "CHECK"]
        }
        response = requests.post(f"{base_url}/classify", json=classify_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Classification successful")
            print(f"   Prediction: {result.get('prediction', 'N/A')}")
            print(f"   Top features: {len(result.get('top_features', []))}")
        else:
            print(f"❌ Classification failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Classification error: {e}")
    
    # Test 4: API endpoints
    print("\n4. Testing API endpoints...")
    try:
        response = requests.get(f"{base_url}/api/actions")
        if response.status_code == 200:
            actions = response.json()
            print(f"✅ Actions API: {len(actions)} actions available")
        else:
            print(f"❌ Actions API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Actions API error: {e}")
    
    try:
        response = requests.get(f"{base_url}/api/positions")
        if response.status_code == 200:
            positions = response.json()
            print(f"✅ Positions API: {len(positions)} positions available")
        else:
            print(f"❌ Positions API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Positions API error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Testing complete! Check the results above.")

if __name__ == "__main__":
    # Wait a moment for the app to be ready
    print("⏳ Waiting for web app to be ready...")
    time.sleep(3)
    
    test_webapp()
