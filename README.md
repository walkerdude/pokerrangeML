# 🎯 Poker Range Classification System

A **production-ready machine learning system** that classifies opponent hand ranges into four categories: **Nuts**, **Strong**, **Marginal**, and **Bluff** based on action history and game context. Features a **full-stack web dashboard** for real-time predictions and portfolio integration.

## 🎯 **System Overview**

This system analyzes poker action sequences to predict the likely strength of an opponent's hand using **advanced feature engineering** and **machine learning**. The trained model achieves **98.7% accuracy** and is deployed as a **production web application**.

## ✨ **Key Features**

### 🎯 **Four-Class Classification (98.7% Accuracy)**
- **Nuts**: Premium hands (top 5% of hands)
- **Strong**: Good hands (top 25% of hands)  
- **Marginal**: Medium-strength hands (45% of hands)
- **Bluff**: Weak hands and bluffs (30% of hands)

### 🔧 **Advanced Feature Engineering (113 Features)**
- **Action Patterns**: Aggression ratios, bet sizing patterns, action frequencies
- **Street-Specific Features**: Separate analysis for preflop, flop, turn, and river
- **Position Features**: Early/middle/late position interactions
- **Pot Dynamics**: Pot odds, stack depth, multi-way pot features
- **Temporal Features**: Action progression across streets
- **Interaction Features**: Position-aggression, stack-bet sizing interactions
- **Sequence Features**: Action sequence patterns and consistency

### 🤖 **Production ML Models**
- **Random Forest Classifier** (Primary - 98.7% accuracy)
- **Gradient Boosting Classifier** (97.0% accuracy)
- **Logistic Regression** (98.0% accuracy)
- **Support Vector Machine** (96.0% accuracy)
- **Automatic hyperparameter tuning** with GridSearchCV
- **Model selection** based on cross-validation

### 🌐 **Full-Stack Web Application**
- **Flask Backend**: RESTful API with real-time predictions
- **Modern Frontend**: Bootstrap 5 UI with interactive elements
- **Real-time Classification**: Live hand strength predictions
- **Model Training Interface**: Web-based model retraining
- **Production Ready**: WSGI server support (Gunicorn)
- **Responsive Design**: Works on desktop and mobile

### 📊 **Comprehensive Evaluation & Analytics**
- **Confusion matrices** and classification reports
- **ROC curves** and precision-recall curves
- **Feature importance analysis** with visualizations
- **Prediction confidence analysis**
- **Error pattern identification**
- **Interactive visualizations** with Chart.js

## 🚀 **Quick Start**

### 1. **Clone & Setup**
```bash
git clone https://github.com/walkerdude/pokerrangeML.git
cd pokermachinelearning
pip install -r requirements.txt
```

### 2. **Run Web Dashboard**
```bash
python app.py
# Open http://localhost:3000 in your browser
```

### 3. **Train New Model (Optional)**
```bash
python create_proper_working_model.py
# Creates poker_range_classifier_proper.pkl
```

## 🌐 **Web Dashboard Features**

### **Real-time Hand Classification**
- Input position, pot size, stack size, player count
- Select poker actions (fold, call, raise, all-in, etc.)
- Get instant predictions with confidence scores
- View top contributing features

### **Model Training Interface**
- Retrain models with custom parameters
- Generate new training data
- View training progress and results
- Download trained models

### **Portfolio Integration**
- **`portfolio_demo.html`**: Standalone demo page
- **`portfolio_embed.html`**: Iframe-ready version
- **Copy-paste integration** for any website
- **Professional presentation** for resumes/portfolios

## 📁 **Project Structure**

```
pokermachinelearning/
├── 🌐 Web Application
│   ├── app.py                    # Flask web server
│   ├── wsgi.py                   # Production WSGI entry point
│   ├── templates/                # HTML templates
│   ├── static/                   # CSS, JS, assets
│   └── portfolio_demo.html       # Portfolio demo page
├── 🤖 Machine Learning
│   ├── data_generator.py         # Training data generation
│   ├── feature_engineering.py    # 113 feature extraction
│   ├── range_classifier.py       # ML model training
│   ├── evaluation.py             # Model evaluation
│   └── main.py                   # Command-line pipeline
├── 🧪 Model Creation Scripts
│   ├── create_proper_working_model.py    # Production model
│   ├── create_working_model.py           # Working model
│   └── create_simple_working_model.py    # Simple model
├── 📊 Trained Models
│   ├── poker_range_classifier_proper.pkl # Production model
│   ├── poker_range_classifier_working.pkl
│   └── poker_range_classifier_simple.pkl
├── 📋 Documentation
│   ├── DEPLOYMENT.md             # Live deployment guide
│   ├── WEBAPP_GUIDE.md           # Web app usage guide
│   └── requirements.txt          # Python dependencies
└── 🧪 Testing & Debug
    ├── test_webapp.py            # Web app testing
    ├── test_system.py            # System testing
    └── debug_*.py                # Debug scripts
```

## 🎯 **Model Performance**

### **Current Production Model**
- **Overall Accuracy**: **98.7%** (Random Forest)
- **Per-Class F1 Scores**: 0.95-1.00 across all classes
- **ROC AUC**: 0.98+ across all classes
- **Training Data**: 15,000+ simulated hands
- **Features**: 113 engineered features

### **Model Comparison**
| Model | Accuracy | Training Time | Best For |
|-------|----------|---------------|----------|
| **Random Forest** | **98.7%** | Fast | Production |
| Gradient Boosting | 97.0% | Medium | High precision |
| Logistic Regression | 98.0% | Very Fast | Baseline |
| SVM | 96.0% | Slow | Complex patterns |

## 🔧 **Advanced Usage**

### **Command Line Interface**
```bash
# Generate training data
python main.py --generate-data --num-hands 10000

# Train model
python main.py --train

# Evaluate model
python main.py --evaluate

# Complete pipeline
python main.py --generate-data --train --evaluate
```

### **Custom Model Training**
```bash
# Create production model
python create_proper_working_model.py

# Create simple model
python create_simple_working_model.py

# Test model responsiveness
python test_system.py
```

### **Web App Testing**
```bash
# Test web dashboard
python test_webapp.py

# Test classification
python test_classification.py
```

## 📊 **Data Structure**

### **Input Features (113 Total)**
- **Action Sequences**: 40 action codes (10 per street)
- **Position**: 0-8 (UTG to BB)
- **Pot Size**: Current pot size
- **Stack Size**: Player's remaining chips
- **Number of Players**: Active players in hand
- **Interaction Features**: Position-aggression, stack-bet sizing
- **Temporal Features**: Action progression across streets

### **Output Classes**
- **0**: Bluff (Weak hands/bluffs)
- **1**: Strong (Good hands)
- **2**: Marginal (Medium hands)
- **3**: Nuts (Premium hands)

## 🎨 **Customization**

### **Adding New Features**
1. Modify `feature_engineering.py`
2. Add new feature extraction methods
3. Update the `engineer_all_features()` method

### **Adjusting Hand Strength Distribution**
1. Modify `generate_hand_strength()` in `data_generator.py`
2. Update the probability weights for each class

### **Adding New Models**
1. Add model to the `models` dictionary in `range_classifier.py`
2. Implement hyperparameter tuning if needed

## 📚 **Dependencies**

- **Core ML**: `numpy`, `pandas`, `scikit-learn`
- **Web Framework**: `flask`, `gunicorn`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `chart.js`
- **Utilities**: `joblib`, `tqdm`, `jinja2`

## 🌟 **Portfolio Showcase**

This project demonstrates:
- **Advanced ML**: 98.7% accuracy on complex classification
- **Full-Stack Development**: Flask backend + modern frontend
- **Production Deployment**: WSGI servers, deployment automation
- **Real-world Application**: Practical poker analysis tool
- **Professional Quality**: Clean code, documentation, testing
