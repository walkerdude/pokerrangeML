import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent macOS crashes
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PokerRangeClassifier:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.class_names = ['Nuts', 'Strong', 'Marginal', 'Bluff']
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train multiple models and select the best one"""
        
        print("Training multiple models for range classification...")
        
        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # Train each model
        results = {}
        for name, model in tqdm(models.items(), desc="Training models"):
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                results[name] = float(accuracy)
                print(f"{name} validation accuracy: {accuracy:.4f}")
            else:
                # Use cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                results[name] = float(cv_scores.mean())
                print(f"{name} CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model
        self.best_model_name = max(results, key=results.get)
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nBest model: {self.best_model_name} with accuracy: {results[self.best_model_name]:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                            model_name: str = 'RandomForest') -> Any:
        """Perform hyperparameter tuning for the best model"""
        
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return self.models[model_name]
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        self.models[model_name] = self.best_model
        
        # Convert best_params_ to regular dict to avoid numpy types
        best_params = {}
        for key, value in grid_search.best_params_.items():
            if hasattr(value, 'item') and hasattr(value, 'size') and value.size == 1:
                best_params[key] = value.item()
            elif isinstance(value, np.ndarray):
                best_params[key] = value.tolist()
            else:
                best_params[key] = value
        
        return best_params
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = None) -> Dict[str, Any]:
        """Evaluate the model on test data"""
        
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models[model_name]
        
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get actual class names from the data
        unique_classes = np.unique(y_test)
        actual_class_names = [f'Class_{i}' for i in unique_classes]
        
        # Always use the default report to avoid class name mismatches
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                if obj.size == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            elif hasattr(obj, 'item') and hasattr(obj, 'size') and obj.size == 1:
                return obj.item()
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj
        
        results = {
            'accuracy': float(accuracy),
            'classification_report': convert_numpy_types(report),
            'confusion_matrix': convert_numpy_types(cm),
            'predictions': convert_numpy_types(y_pred),
            'probabilities': convert_numpy_types(y_pred_proba)
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return results
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance from the model"""
        
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            print("Model doesn't support feature importance")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict_range(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict hand range classification"""
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        return predictions, probabilities
    
    def predict_single_hand(self, features):
        """Predict the class for a single hand"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # New working model format
                prediction = self.model.predict([features])[0]
                probabilities = self.model.predict_proba([features])[0]
                
                result = {
                    'predicted_class': self.class_names[prediction],
                    'class_id': int(prediction),
                    'confidence': float(probabilities[prediction]),
                    'probabilities': {
                        self.class_names[i]: float(prob) for i, prob in enumerate(probabilities)
                    }
                }
                return result
            elif hasattr(self, 'best_model') and self.best_model is not None:
                # Old model format
                prediction = self.best_model.predict([features])[0]
                probabilities = self.best_model.predict_proba([features])[0]
                
                result = {
                    'predicted_class': self.class_names[prediction],
                    'class_id': int(prediction),
                    'confidence': float(probabilities[prediction]),
                    'probabilities': {
                        self.class_names[i]: float(prob) for i, prob in enumerate(probabilities)
                    }
                }
                return result
            else:
                return {'error': 'No model loaded'}
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'models': self.models,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, model_path):
        """Load a trained model from file"""
        try:
            if model_path.endswith('_working.pkl'):
                # Load the new working model format
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.feature_engineer = model_data['feature_engineer']
                self.class_names = model_data['class_names']
                print(f"Model loaded from {model_path}")
                print("✓ Working model loaded successfully")
            else:
                # Load the old model format
                self.model = joblib.load(model_path)
                print(f"Model loaded from {model_path}")
                print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str = "Confusion Matrix"):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        # Don't call plt.show() on headless servers
        # plt.show()
        plt.close()
    
    def plot_feature_importance(self, top_n: int = 20) -> str:
        """Plot feature importance (returns base64 encoded image or error message)"""
        try:
            if not hasattr(self.best_model, 'feature_importances_'):
                return "Feature importance not available for this model type"
            
            # Get feature importance
            importance_df = self.get_feature_importance()
            if importance_df is None:
                return "No feature importance data available"
            
            # Plot top features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(top_n)
            
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            # Save to base64 string instead of showing
            import io
            import base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            plt.close()  # Ensure plot is closed
            return f"Plotting failed: {str(e)}"
    
    def safe_plot(self, plot_type: str = 'importance') -> str:
        """Safely create plots without crashing"""
        try:
            if plot_type == 'importance':
                return self.plot_feature_importance()
            else:
                return "Unknown plot type"
        except Exception as e:
            return f"Plotting failed: {str(e)}"
    
    def plot_class_distribution(self, y: np.ndarray, title: str = "Class Distribution"):
        """Plot class distribution"""
        plt.figure(figsize=(8, 6))
        class_counts = pd.Series(y).value_counts().sort_index()
        
        plt.bar(self.class_names, class_counts.values)
        plt.title(title)
        plt.xlabel('Hand Strength')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(class_counts.values):
            plt.text(i, count + max(class_counts.values) * 0.01, str(count), 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        # Don't call plt.show() on headless servers
        # plt.show()
        plt.close()

def train_range_classifier(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                          test_size: float = 0.2, random_state: int = 42) -> PokerRangeClassifier:
    """Complete training pipeline for range classification"""
    
    print("Starting Poker Range Classification Training Pipeline")
    print("=" * 60)
    
    # Split data - use non-stratified split to avoid class balance issues
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize classifier
    classifier = PokerRangeClassifier()
    classifier.feature_names = feature_names
    
    # Train models
    results = classifier.train_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning for best model
    classifier.hyperparameter_tuning(X_train, y_train, classifier.best_model_name)
    
    # Final evaluation
    evaluation_results = classifier.evaluate_model(X_test, y_test)
    
    # Plot results (commented out for headless server compatibility)
    # classifier.plot_confusion_matrix(evaluation_results['confusion_matrix'])
    # classifier.plot_feature_importance()
    # classifier.plot_class_distribution(y, "Training Data Class Distribution")
    
    return classifier, evaluation_results

if __name__ == "__main__":
    # Test the classifier
    from data_generator import PokerDataGenerator
    from feature_engineering import PokerFeatureEngineer
    
    # Generate and prepare data
    print("Generating training data...")
    generator = PokerDataGenerator(num_hands=5000)
    df = generator.generate_dataset()
    
    print("Engineering features...")
    engineer = PokerFeatureEngineer()
    df_engineered = engineer.engineer_all_features(df)
    
    # Prepare features
    X, y = engineer.prepare_features(df_engineered)
    feature_names = engineer.get_feature_names(df_engineered)
    
    # Train classifier
    classifier, results = train_range_classifier(X, y, feature_names)
    
    # Save model
    classifier.save_model('poker_range_classifier.pkl')
