import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Class to perform feature engineering on healthcare datasets"""

    def __init__(self, data_dir='data', output_dir='data'):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def load_processed_data(self, dataset_name):
        """Load processed dataset"""
        file_path = os.path.join(self.data_dir, f'{dataset_name}_processed.csv')
        df = pd.read_csv(file_path)
        print(f"Loaded {dataset_name} dataset: {df.shape}")
        return df

    def create_bmi_categories(self, df):
        """Create BMI categories"""
        if 'bmi' in df.columns:
            df['bmi_category'] = pd.cut(df['bmi'],
                                      bins=[-np.inf, 18.5, 25, 30, np.inf],
                                      labels=['underweight', 'normal', 'overweight', 'obese'])
            # Convert to numeric
            df['bmi_underweight'] = (df['bmi_category'] == 'underweight').astype(int)
            df['bmi_normal'] = (df['bmi_category'] == 'normal').astype(int)
            df['bmi_overweight'] = (df['bmi_category'] == 'overweight').astype(int)
            df['bmi_obese'] = (df['bmi_category'] == 'obese').astype(int)

            print("+ Created BMI categories")
        return df

    def create_age_groups(self, df):
        """Create age groups"""
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                   bins=[-np.inf, 30, 45, 60, np.inf],
                                   labels=['young', 'middle_aged', 'senior', 'elderly'])
            # Convert to numeric
            df['age_young'] = (df['age_group'] == 'young').astype(int)
            df['age_middle_aged'] = (df['age_group'] == 'middle_aged').astype(int)
            df['age_senior'] = (df['age_group'] == 'senior').astype(int)
            df['age_elderly'] = (df['age_group'] == 'elderly').astype(int)

            print("+ Created age groups")
        return df

    def create_risk_factors(self, df, dataset_name):
        """Create risk factor combinations"""
        if dataset_name == 'diabetes':
            # Diabetes-specific risk factors
            df['glucose_risk'] = (df['glucose_concentration'] > 140).astype(int)
            df['bp_risk'] = (df['blood_pressure'] > 80).astype(int)
            df['bmi_risk'] = (df['bmi'] > 30).astype(int)
            df['age_risk'] = (df['age'] > 50).astype(int)

            # Combined risk score
            df['diabetes_risk_score'] = df['glucose_risk'] + df['bp_risk'] + df['bmi_risk'] + df['age_risk']

        elif dataset_name == 'heart_disease':
            # Heart disease-specific risk factors
            df['age_risk'] = (df['age'] > 55).astype(int)
            df['chol_risk'] = (df['chol'] > 240).astype(int)
            df['bp_risk'] = (df['trestbps'] > 130).astype(int)
            df['thalach_risk'] = (df['thalach'] < 150).astype(int)

            # Combined risk score
            df['heart_risk_score'] = df['age_risk'] + df['chol_risk'] + df['bp_risk'] + df['thalach_risk']

        elif dataset_name == 'kidney_disease':
            # Kidney disease-specific risk factors
            df['age_risk'] = (df['age'] > 60).astype(int)
            df['bp_risk'] = (df['blood_pressure'] > 120).astype(int)
            df['urea_risk'] = (df['blood_urea'] > 40).astype(int)
            df['creatinine_risk'] = (df['serum_creatinine'] > 1.2).astype(int)

            # Combined risk score
            df['kidney_risk_score'] = df['age_risk'] + df['bp_risk'] + df['urea_risk'] + df['creatinine_risk']

        print(f"+ Created risk factors for {dataset_name}")
        return df

    def create_interaction_features(self, df, dataset_name):
        """Create interaction features"""
        if dataset_name == 'diabetes':
            # Key interactions for diabetes
            df['glucose_bmi_interaction'] = df['glucose_concentration'] * df['bmi']
            df['age_glucose_interaction'] = df['age'] * df['glucose_concentration']
            df['insulin_bmi_interaction'] = df['insulin'] * df['bmi']

        elif dataset_name == 'heart_disease':
            # Key interactions for heart disease
            df['age_chol_interaction'] = df['age'] * df['chol']
            df['bp_thalach_interaction'] = df['trestbps'] * df['thalach']
            df['oldpeak_slope_interaction'] = df['oldpeak'] * df['slope']

        elif dataset_name == 'kidney_disease':
            # Key interactions for kidney disease
            df['age_bp_interaction'] = df['age'] * df['blood_pressure']
            df['urea_creatinine_interaction'] = df['blood_urea'] * df['serum_creatinine']

        print(f"+ Created interaction features for {dataset_name}")
        return df

    def remove_multicollinear_features(self, df, threshold=0.8):
        """Remove highly correlated features"""
        # Separate features and target
        target_col = 'class' if 'class' in df.columns else 'target'
        feature_cols = [col for col in df.columns if col != target_col]

        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()

        # Find highly correlated features
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Get features to drop
        to_drop = []
        for col in upper_triangle.columns:
            if any(upper_triangle[col] > threshold):
                # Get the feature with highest correlation to the target
                correlated_features = upper_triangle[col][upper_triangle[col] > threshold].index.tolist()
                target_corr = {}

                for feat in correlated_features + [col]:
                    if target_col in df.columns:
                        target_corr[feat] = abs(df[feat].corr(df[target_col]))

                # Keep the feature most correlated with target
                best_feature = max(target_corr.keys(), key=lambda x: target_corr[x])
                to_drop.extend([f for f in correlated_features if f != best_feature])

        # Remove duplicates from to_drop
        to_drop = list(set(to_drop))

        if to_drop:
            df = df.drop(columns=to_drop)
            print(f"+ Removed {len(to_drop)} highly correlated features: {to_drop}")
        else:
            print("+ No highly correlated features found to remove")

        return df

    def select_k_best_features(self, df, k=10):
        """Select K best features using statistical tests"""
        target_col = 'class' if 'class' in df.columns else 'target'
        feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols]
        y = df[target_col]

        # Use f_classif for classification
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

        # Create new dataframe with selected features
        df_selected = pd.DataFrame(X_selected, columns=selected_features)
        df_selected[target_col] = y

        print(f"+ Selected top {k} features: {selected_features}")

        return df_selected

    def engineer_features(self, df, dataset_name):
        """Complete feature engineering pipeline"""
        print(f"\n{'='*50}")
        print(f"FEATURE ENGINEERING - {dataset_name.upper()}")
        print(f"{'='*50}")

        initial_shape = df.shape
        print(f"Initial shape: {initial_shape}")

        # 1. Create BMI categories (if applicable)
        df = self.create_bmi_categories(df)

        # 2. Create age groups
        df = self.create_age_groups(df)

        # 3. Create risk factors
        df = self.create_risk_factors(df, dataset_name)

        # 4. Create interaction features
        df = self.create_interaction_features(df, dataset_name)

        print(f"After feature creation: {df.shape}")

        # Convert categorical features to numeric
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != 'class' and col != 'target':  # Don't encode target
                df[col] = pd.Categorical(df[col]).codes

        print(f"After encoding categorical features: {df.shape}")

        # 5. Remove multicollinear features
        df = self.remove_multicollinear_features(df)

        print(f"After removing multicollinear features: {df.shape}")

        # 6. Select K best features
        n_features = min(15, len(df.columns) - 1)  # Keep target column
        df = self.select_k_best_features(df, k=n_features)

        print(f"Final shape after feature selection: {df.shape}")
        print(f"Features added: {df.shape[1] - initial_shape[1]}")
        print(f"Samples: {df.shape[0]}")

        return df

    def save_engineered_data(self, df, dataset_name):
        """Save engineered dataset"""
        output_file = os.path.join(self.output_dir, f'{dataset_name}_engineered.csv')
        df.to_csv(output_file, index=False)
        print(f"+ Engineered dataset saved to: {output_file}\n")

    def process_all_datasets(self):
        """Process all datasets"""
        datasets = ['diabetes', 'heart_disease', 'kidney_disease']

        for dataset in datasets:
            try:
                # Load data
                df = self.load_processed_data(dataset)

                # Engineer features
                df_engineered = self.engineer_features(df, dataset)

                # Save results
                self.save_engineered_data(df_engineered, dataset)

            except Exception as e:
                print(f"Error processing {dataset}: {e}")
                continue

        print(f"{'='*60}")
        print("FEATURE ENGINEERING COMPLETED FOR ALL DATASETS!")
        print(f"{'='*60}")

def main():
    """Main function"""
    engineer = FeatureEngineer()
    engineer.process_all_datasets()

if __name__ == "__main__":
    main()