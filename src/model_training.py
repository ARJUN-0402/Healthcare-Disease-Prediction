import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Class to train and evaluate multiple ML models on healthcare datasets"""

    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Define models with their hyperparameters
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            }
        }

    def load_engineered_data(self, dataset_name):
        """Load engineered dataset"""
        file_path = os.path.join(self.data_dir, f'{dataset_name}_engineered.csv')
        df = pd.read_csv(file_path)
        print(f"Loaded {dataset_name} dataset: {df.shape}")
        return df

    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        target_col = 'class' if 'class' in df.columns else 'target'
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_model_with_tuning(self, model_name, model_info, X_train, y_train):
        """Train model with hyperparameter tuning"""
        print(f"\nTraining {model_name}...")

        model = model_info['model']
        params = model_info['params']

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            model, params, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation F1 score: {best_score:.4f}")

        return best_model, best_params, best_score

    def evaluate_model(self, model, X_test, y_test, model_name, dataset_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n{model_name} Results for {dataset_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

    def train_all_models(self, dataset_name):
        """Train all models on a dataset"""
        print(f"\n{'='*60}")
        print(f"TRAINING MODELS FOR {dataset_name.upper()} DATASET")
        print(f"{'='*60}")

        # Load data
        df = self.load_engineered_data(dataset_name)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)

        # Store results
        results = {}
        trained_models = {}

        # Train each model
        for model_name, model_info in self.models.items():
            try:
                # Train model with hyperparameter tuning
                best_model, best_params, best_cv_score = self.train_model_with_tuning(
                    model_name, model_info, X_train, y_train
                )

                # Evaluate model
                metrics = self.evaluate_model(best_model, X_test, y_test, model_name, dataset_name)

                # Store results
                results[model_name] = {
                    'best_params': best_params,
                    'cv_f1_score': best_cv_score,
                    **metrics
                }

                trained_models[model_name] = best_model

                # Save model
                model_filename = os.path.join(self.models_dir, f'{dataset_name}_{model_name.lower().replace(" ", "_")}.pkl')
                joblib.dump(best_model, model_filename)
                print(f"Model saved to: {model_filename}")

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

        # Create results summary
        self.create_results_summary(results, dataset_name)

        return results, trained_models

    def create_results_summary(self, results, dataset_name):
        """Create a summary of all model results"""
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON SUMMARY - {dataset_name.upper()}")
        print(f"{'='*60}")

        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'CV F1 Score': f"{metrics['cv_f1_score']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}"
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by F1 score
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)

        print(comparison_df.to_string(index=False))

        # Save results to CSV
        results_file = os.path.join(self.models_dir, f'{dataset_name}_model_results.csv')
        comparison_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

        # Find best model
        best_model = comparison_df.iloc[0]['Model']
        best_f1 = comparison_df.iloc[0]['F1 Score']
        print(f"\nBest performing model: {best_model} (F1 Score: {best_f1})")

    def train_on_all_datasets(self):
        """Train models on all datasets"""
        datasets = ['diabetes', 'heart_disease', 'kidney_disease']

        all_results = {}

        for dataset in datasets:
            try:
                results, trained_models = self.train_all_models(dataset)
                all_results[dataset] = results

            except Exception as e:
                print(f"Error processing {dataset}: {e}")
                continue

        print(f"\n{'='*80}")
        print("MODEL TRAINING COMPLETED FOR ALL DATASETS!")
        print(f"{'='*80}")
        print(f"Trained models saved in: {self.models_dir}")

        return all_results

def main():
    """Main function"""
    trainer = ModelTrainer()
    all_results = trainer.train_on_all_datasets()
    return all_results

if __name__ == "__main__":
    main()