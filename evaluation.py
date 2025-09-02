import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class PokerRangeEvaluator:
    def __init__(self, classifier, class_names=None):
        self.classifier = classifier
        self.class_names = class_names or ['Nuts', 'Strong', 'Marginal', 'Bluff']
        
    def comprehensive_evaluation(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the model"""
        
        print("Performing Comprehensive Model Evaluation")
        print("=" * 50)
        
        # Get predictions
        y_pred = self.classifier.best_model.predict(X_test)
        y_pred_proba = self.classifier.best_model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC and PR curves for each class
        roc_data = {}
        pr_data = {}
        
        for i, class_name in enumerate(self.class_names):
            # ROC curve
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
            auc = roc_auc_score((y_test == i).astype(int), y_pred_proba[:, i])
            roc_data[class_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
            
            # PR curve
            precision_curve, recall_curve, _ = precision_recall_curve(
                (y_test == i).astype(int), y_pred_proba[:, i]
            )
            ap = average_precision_score((y_test == i).astype(int), y_pred_proba[:, i])
            pr_data[class_name] = {'precision': precision_curve, 'recall': recall_curve, 'ap': ap}
        
        results = {
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_data': roc_data,
            'pr_data': pr_data
        }
        
        return results
    
    def plot_comprehensive_results(self, results: Dict[str, Any]):
        """Plot comprehensive evaluation results"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'Per-Class Metrics', 'ROC Curves', 'Precision-Recall Curves'),
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Confusion Matrix
        cm = results['confusion_matrix']
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=self.class_names,
                y=self.class_names,
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                showscale=True
            ),
            row=1, col=1
        )
        
        # 2. Per-Class Metrics
        metrics = results['class_metrics']
        precision_values = [metrics[cls]['precision'] for cls in self.class_names]
        recall_values = [metrics[cls]['recall'] for cls in self.class_names]
        f1_values = [metrics[cls]['f1_score'] for cls in self.class_names]
        
        fig.add_trace(
            go.Bar(name='Precision', x=self.class_names, y=precision_values, marker_color='blue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Recall', x=self.class_names, y=recall_values, marker_color='red'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='F1-Score', x=self.class_names, y=f1_values, marker_color='green'),
            row=1, col=2
        )
        
        # 3. ROC Curves
        for class_name, roc_info in results['roc_data'].items():
            fig.add_trace(
                go.Scatter(
                    x=roc_info['fpr'],
                    y=roc_info['tpr'],
                    mode='lines',
                    name=f'{class_name} (AUC={roc_info["auc"]:.3f})',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                      line=dict(dash='dash', color='gray')),
            row=2, col=1
        )
        
        # 4. Precision-Recall Curves
        for class_name, pr_info in results['pr_data'].items():
            fig.add_trace(
                go.Scatter(
                    x=pr_info['recall'],
                    y=pr_info['precision'],
                    mode='lines',
                    name=f'{class_name} (AP={pr_info["ap"]:.3f})',
                    line=dict(width=2)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Comprehensive Model Evaluation",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="Actual", row=1, col=1)
        fig.update_xaxes(title_text="Class", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)
        fig.update_xaxes(title_text="Recall", row=2, col=2)
        fig.update_yaxes(title_text="Precision", row=2, col=2)
        
        fig.show()
    
    def analyze_prediction_confidence(self, results: Dict[str, Any]):
        """Analyze prediction confidence distribution"""
        
        probabilities = results['probabilities']
        predictions = results['predictions']
        
        # Get confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        # Create confidence analysis
        confidence_data = pd.DataFrame({
            'predicted_class': [self.class_names[pred] for pred in predictions],
            'confidence': confidence_scores,
            'correct': predictions == results.get('true_labels', predictions)  # Assuming we have true labels
        })
        
        # Plot confidence distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall confidence distribution
        axes[0, 0].hist(confidence_scores, bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Overall Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Confidence by class
        for i, class_name in enumerate(self.class_names):
            class_confidences = confidence_scores[predictions == i]
            axes[0, 1].hist(class_confidences, bins=20, alpha=0.7, label=class_name)
        axes[0, 1].set_title('Confidence Distribution by Class')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Confidence vs accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
            if mask.sum() > 0:
                accuracies.append(confidence_data.loc[mask, 'correct'].mean())
            else:
                accuracies.append(0)
        
        axes[1, 0].plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Confidence vs Accuracy')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average confidence by class
        avg_confidence = [confidence_scores[predictions == i].mean() for i in range(len(self.class_names))]
        axes[1, 1].bar(self.class_names, avg_confidence, color=['red', 'orange', 'yellow', 'green'])
        axes[1, 1].set_title('Average Confidence by Class')
        axes[1, 1].set_ylabel('Average Confidence')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return confidence_data
    
    def feature_importance_analysis(self, top_n: int = 20):
        """Analyze and visualize feature importance"""
        
        importance_df = self.classifier.get_feature_importance()
        
        if importance_df is None:
            print("Model doesn't support feature importance analysis")
            return None
        
        # Create interactive feature importance plot
        top_features = importance_df.head(top_n)
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='lightblue',
                text=top_features['importance'].round(4),
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title=f'Top {top_n} Most Important Features',
            xaxis_title='Feature Importance',
            yaxis_title='Feature Name',
            height=600
        )
        
        fig.show()
        
        # Print top features
        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 50)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        return importance_df
    
    def error_analysis(self, X_test: np.ndarray, y_test: np.ndarray, 
                      feature_names: List[str], sample_size: int = 100):
        """Analyze prediction errors"""
        
        y_pred = self.classifier.best_model.predict(X_test)
        y_pred_proba = self.classifier.best_model.predict_proba(X_test)
        
        # Find misclassified samples
        misclassified_mask = y_pred != y_test
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassified samples found!")
            return
        
        # Sample misclassified cases
        sample_indices = np.random.choice(
            misclassified_indices, 
            size=min(sample_size, len(misclassified_indices)), 
            replace=False
        )
        
        print(f"\nError Analysis - Sample of {len(sample_indices)} Misclassified Cases:")
        print("=" * 80)
        
        error_summary = {}
        
        for i, idx in enumerate(sample_indices):
            true_class = self.class_names[y_test[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = np.max(y_pred_proba[idx])
            
            error_type = f"{true_class} → {pred_class}"
            if error_type not in error_summary:
                error_summary[error_type] = []
            
            error_summary[error_type].append({
                'index': idx,
                'confidence': confidence,
                'probabilities': y_pred_proba[idx]
            })
            
            print(f"Case {i+1:3d}: {true_class:>8} → {pred_class:<8} (conf: {confidence:.3f})")
        
        # Analyze error patterns
        print(f"\nError Pattern Analysis:")
        print("-" * 40)
        for error_type, cases in error_summary.items():
            avg_confidence = np.mean([case['confidence'] for case in cases])
            print(f"{error_type}: {len(cases)} cases, avg confidence: {avg_confidence:.3f}")
        
        return error_summary
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 save_path: str = "evaluation_report.txt"):
        """Generate a comprehensive evaluation report"""
        
        with open(save_path, 'w') as f:
            f.write("POKER RANGE CLASSIFICATION - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall performance
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n\n")
            
            # Per-class performance
            f.write("Per-Class Performance:\n")
            f.write("-" * 30 + "\n")
            for class_name, metrics in results['class_metrics'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n\n")
            
            # ROC AUC scores
            f.write("ROC AUC Scores:\n")
            f.write("-" * 20 + "\n")
            for class_name, roc_info in results['roc_data'].items():
                f.write(f"{class_name}: {roc_info['auc']:.4f}\n")
            f.write("\n")
            
            # Average Precision scores
            f.write("Average Precision Scores:\n")
            f.write("-" * 30 + "\n")
            for class_name, pr_info in results['pr_data'].items():
                f.write(f"{class_name}: {pr_info['ap']:.4f}\n")
            f.write("\n")
            
            # Confusion matrix
            f.write("Confusion Matrix:\n")
            f.write("-" * 20 + "\n")
            cm = results['confusion_matrix']
            f.write("Predicted →\n")
            f.write("Actual ↓\n")
            f.write("     " + " ".join(f"{cls:>8}" for cls in self.class_names) + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name:>8} " + " ".join(f"{val:>8}" for val in cm[i]) + "\n")
        
        print(f"Evaluation report saved to {save_path}")

if __name__ == "__main__":
    # Test the evaluator
    from data_generator import PokerDataGenerator
    from feature_engineering import PokerFeatureEngineer
    from range_classifier import PokerRangeClassifier
    
    # Load or train a model
    try:
        classifier = PokerRangeClassifier()
        classifier.load_model('poker_range_classifier.pkl')
        print("Loaded existing model")
    except:
        print("Training new model...")
        # Generate data and train
        generator = PokerDataGenerator(num_hands=2000)
        df = generator.generate_dataset()
        
        engineer = PokerFeatureEngineer()
        df_engineered = engineer.engineer_all_features(df)
        
        X, y = engineer.prepare_features(df_engineered)
        feature_names = engineer.get_feature_names(df_engineered)
        
        # Split for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train classifier
        classifier = PokerRangeClassifier()
        classifier.feature_names = feature_names
        classifier.train_models(X_train, y_train, X_test, y_test)
    
    # Create evaluator
    evaluator = PokerRangeEvaluator(classifier)
    
    # Perform evaluation
    results = evaluator.comprehensive_evaluation(X_test, y_test)
    
    # Generate plots and analysis
    evaluator.plot_comprehensive_results(results)
    evaluator.analyze_prediction_confidence(results)
    evaluator.feature_importance_analysis()
    evaluator.error_analysis(X_test, y_test, classifier.feature_names)
    evaluator.generate_evaluation_report(results)
