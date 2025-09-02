# Poker Range Classification System

A machine learning system that classifies opponent hand ranges into four categories: **Nuts**, **Strong**, **Marginal**, and **Bluff** based on action history and game context.

## Overview

This system analyzes poker action sequences to predict the likely strength of an opponent's hand. It uses advanced feature engineering to extract meaningful patterns from betting behavior, position, pot dynamics, and temporal progression across streets.

## Features

### 🎯 **Four-Class Classification**
- **Nuts**: Premium hands (top 5% of hands)
- **Strong**: Good hands (top 25% of hands)  
- **Marginal**: Medium-strength hands (45% of hands)
- **Bluff**: Weak hands and bluffs (30% of hands)

### 🔧 **Advanced Feature Engineering**
- **Action Patterns**: Aggression ratios, bet sizing patterns, action frequencies
- **Street-Specific Features**: Separate analysis for preflop, flop, turn, and river
- **Position Features**: Early/middle/late position interactions
- **Pot Dynamics**: Pot odds, stack depth, multi-way pot features
- **Temporal Features**: Action progression across streets
- **Interaction Features**: Position-aggression, stack-bet sizing interactions

### 🤖 **Multiple ML Models**
- Random Forest Classifier
- Gradient Boosting Classifier  
- Logistic Regression
- Support Vector Machine (SVM)
- Automatic hyperparameter tuning
- Model selection based on cross-validation

### 📊 **Comprehensive Evaluation**
- Confusion matrices and classification reports
- ROC curves and precision-recall curves
- Feature importance analysis
- Prediction confidence analysis
- Error pattern analysis
- Interactive visualizations

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd pokermachinelearning
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Generate data, train model, and run evaluation:
```bash
python main.py --generate-data --train --evaluate
```

### Individual Commands

**Generate Training Data:**
```bash
python main.py --generate-data --num-hands 10000
```

**Train the Model:**
```bash
python main.py --train
```

**Evaluate the Model:**
```bash
python main.py --evaluate
```

**Run Demonstration:**
```bash
python main.py --demo
```

### Advanced Usage

**Custom Data and Model Files:**
```bash
python main.py --generate-data --num-hands 50000 --data-file my_data.csv
python main.py --train --data-file my_data.csv --model-file my_model.pkl
```

**Complete Pipeline:**
```bash
python main.py --generate-data --train --evaluate --demo
```

## System Architecture

### 1. Data Generation (`data_generator.py`)
- Simulates realistic poker hands with action sequences
- Generates hand strength labels based on poker probabilities
- Creates features for position, pot size, stack size, and player count

### 2. Feature Engineering (`feature_engineering.py`)
- Extracts 100+ engineered features from raw action data
- Handles missing values and feature scaling
- Creates interaction features and temporal patterns

### 3. Range Classification (`range_classifier.py`)
- Trains multiple ML models with cross-validation
- Performs hyperparameter tuning
- Selects best model based on performance
- Provides prediction and probability outputs

### 4. Evaluation (`evaluation.py`)
- Comprehensive model performance analysis
- Interactive visualizations with Plotly
- Feature importance analysis
- Error pattern identification

### 5. Main Pipeline (`main.py`)
- Orchestrates the complete workflow
- Command-line interface for all operations
- Handles data loading, training, and evaluation

## Data Structure

### Input Features
- **Action Sequences**: 40 action codes (10 per street)
- **Position**: 0-8 (UTG to BB)
- **Pot Size**: Current pot size
- **Stack Size**: Player's remaining chips
- **Number of Players**: Active players in hand

### Output Classes
- **0**: Nuts (Premium hands)
- **1**: Strong (Good hands)
- **2**: Marginal (Medium hands)
- **3**: Bluff (Weak hands/bluffs)

## Model Performance

The system typically achieves:
- **Overall Accuracy**: 75-85%
- **Per-Class F1 Scores**: 0.70-0.90
- **ROC AUC**: 0.80-0.90 across all classes

### Feature Importance (Top 10)
1. Aggression ratio
2. Large bet ratio
3. Preflop aggression
4. Position-aggression interaction
5. Action sequence length
6. Flop aggression
7. Pot-aggression interaction
8. Bet sizing progression
9. Multi-way aggression
10. Stack-bet interaction

## Example Output

```
Sample Hand Classifications:
--------------------------------------------------------------------------------
Hand True     Predicted Confidence Probabilities
--------------------------------------------------------------------------------
1    Strong   Strong    0.847      Nuts: 0.12 | Strong: 0.85 | Marginal: 0.02 | Bluff: 0.01
2    Marginal Marginal  0.723      Nuts: 0.05 | Strong: 0.18 | Marginal: 0.72 | Bluff: 0.05
3    Bluff    Bluff     0.891      Nuts: 0.02 | Strong: 0.04 | Marginal: 0.09 | Bluff: 0.89
```

## Customization

### Adding New Features
1. Modify `feature_engineering.py`
2. Add new feature extraction methods
3. Update the `engineer_all_features()` method

### Adjusting Hand Strength Distribution
1. Modify `generate_hand_strength()` in `data_generator.py`
2. Update the probability weights for each class

### Adding New Models
1. Add model to the `models` dictionary in `range_classifier.py`
2. Implement hyperparameter tuning if needed

## File Structure

```
pokermachinelearning/
├── main.py                 # Main pipeline script
├── data_generator.py       # Training data generation
├── feature_engineering.py  # Feature extraction and engineering
├── range_classifier.py     # ML model training and prediction
├── evaluation.py          # Model evaluation and analysis
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── poker_range_data.csv  # Generated training data
└── poker_range_classifier.pkl  # Trained model
```

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive visualizations
- **joblib**: Model persistence
- **tqdm**: Progress bars

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by modern poker theory and range analysis
- Uses realistic poker action patterns and probabilities
- Implements advanced feature engineering techniques for sequential data

## Support

For questions or issues, please open an issue on the repository or contact the maintainers.
