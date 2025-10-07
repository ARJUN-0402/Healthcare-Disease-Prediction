import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """Class to interpret trained ML models using SHAP and feature importance"""

    def __init__(self, data_dir='data', models_dir='models', output_dir='notebooks/interpretability'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_engineered_data(self, dataset_name):
        """Load engineered dataset"""
        file_path = os.path.join(self.data_dir, f'{dataset_name}_engineered.csv')
        df = pd.read_csv(file_path)
        print(f"Loaded {dataset_name} dataset: {df.shape}")
        return df

    def load_best_model(self, dataset_name):
        """Load the best performing model for each dataset"""
        # Based on our previous results, load the best model for each dataset
        best_models = {
            'diabetes': 'xgboost',
            'heart_disease': 'random_forest',
            'kidney_disease': 'svm'
        }

        model_type = best_models.get(dataset_name)
        model_filename = os.path.join(self.models_dir, f'{dataset_name}_{model_type}.pkl')

        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            print(f"Loaded {model_type} model for {dataset_name}")
            return model
        else:
            print(f"Model file not found: {model_filename}")
            return None

    def plot_feature_importance(self, model, feature_names, dataset_name, model_name):
        """Plot feature importance"""
        print(f"\nCreating feature importance plot for {dataset_name}...")

        try:
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importance = model.feature_importances_
                indices = np.argsort(importance)[::-1]

                plt.figure(figsize=(12, 8))
                plt.title(f'Feature Importance - {dataset_name.upper()} ({model_name})')
                plt.bar(range(len(importance)), importance[indices])
                plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()

                output_file = os.path.join(self.output_dir, f'{dataset_name}_{model_name}_feature_importance.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.show()

                # Print top 10 features
                print("\nTop 10 Most Important Features:")
                for i in range(min(10, len(indices))):
                    print(f"{i+1}. {feature_names[indices[i]]}: {importance[indices[i]]:.4f}")

                return importance

        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
            return None

    def create_shap_analysis(self, model, X_train, X_test, dataset_name, model_name):
        """Create SHAP analysis"""
        print(f"\nCreating SHAP analysis for {dataset_name}...")

        try:
            # Use a smaller sample for SHAP analysis to avoid long computation times
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)

            # Choose appropriate explainer based on model type
            if model_name.lower() in ['random_forest', 'xgboost', 'decision_tree']:
                # Use TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                # For binary classification, use only positive class
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values_plot = shap_values[1]
                else:
                    shap_values_plot = shap_values

            else:
                # Use KernelExplainer for other models (slower but more general)
                def predict_fn(X):
                    return model.predict_proba(X)[:, 1]

                # Use a background dataset for KernelExplainer
                background_size = min(50, len(X_train))
                background = X_train.sample(n=background_size, random_state=42)

                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values_plot = explainer.shap_values(X_sample)

            # Create SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_plot, X_sample, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {dataset_name.upper()} ({model_name})')

            output_file = os.path.join(self.output_dir, f'{dataset_name}_{model_name}_shap_summary.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()

            # Create SHAP waterfall plot for first prediction
            if len(X_sample) > 0:
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(explainer.expected_value, shap_values_plot[0],
                                   X_sample.iloc[0], show=False)
                plt.title(f'SHAP Waterfall Plot - {dataset_name.upper()} (First Prediction)')

                output_file = os.path.join(self.output_dir, f'{dataset_name}_{model_name}_shap_waterfall.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.show()

            print(f"SHAP analysis completed for {dataset_name}")

        except Exception as e:
            print(f"Error in SHAP analysis: {e}")

    def interpret_model(self, dataset_name):
        """Complete model interpretation for a dataset"""
        print(f"\n{'='*60}")
        print(f"MODEL INTERPRETABILITY - {dataset_name.upper()}")
        print(f"{'='*60}")

        # Load data
        df = self.load_engineered_data(dataset_name)

        # Separate features and target
        target_col = 'class' if 'class' in df.columns else 'target'
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Load best model
        model = self.load_best_model(dataset_name)
        if model is None:
            print(f"Could not load model for {dataset_name}")
            return

        model_name = type(model).__name__

        # Feature importance analysis
        feature_importance = self.plot_feature_importance(model, X.columns, dataset_name, model_name)

        # SHAP analysis (using train/test split for proper evaluation)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.create_shap_analysis(model, X_train, X_test, dataset_name, model_name)

        # Create feature importance ranking
        if feature_importance is not None:
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            # Save feature importance to CSV
            importance_file = os.path.join(self.output_dir, f'{dataset_name}_feature_importance.csv')
            importance_df.to_csv(importance_file, index=False)
            print(f"Feature importance saved to: {importance_file}")

    def interpret_all_models(self):
        """Interpret all best models"""
        datasets = ['diabetes', 'heart_disease', 'kidney_disease']

        for dataset in datasets:
            try:
                self.interpret_model(dataset)
            except Exception as e:
                print(f"Error interpreting {dataset}: {e}")
                continue

        print(f"\n{'='*80}")
        print("MODEL INTERPRETABILITY COMPLETED FOR ALL DATASETS!")
        print(f"{'='*80}")
        print(f"Interpretability results saved in: {self.output_dir}")

def main():
    """Main function"""
    interpreter = ModelInterpreter()
    interpreter.interpret_all_models()

if __name__ == "__main__":
    main()