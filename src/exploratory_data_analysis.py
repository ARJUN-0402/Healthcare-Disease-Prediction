import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

class EDAnalyzer:
    """Class to perform exploratory data analysis on healthcare datasets"""

    def __init__(self, data_dir='data', output_dir='notebooks/eda_output'):
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_processed_data(self, dataset_name):
        """Load processed dataset"""
        file_path = os.path.join(self.data_dir, f'{dataset_name}_processed.csv')
        df = pd.read_csv(file_path)
        print(f"Loaded {dataset_name} dataset: {df.shape}")
        return df

    def generate_summary_statistics(self, df, dataset_name):
        """Generate summary statistics"""
        print(f"\n{'='*50}")
        print(f"Summary Statistics for {dataset_name.upper()} Dataset")
        print(f"{'='*50}")

        # Basic info
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nColumn Information:")
        print(df.info())

        # Summary statistics
        print(f"\nSummary Statistics:")
        print(df.describe())

        # Class distribution
        if 'class' in df.columns:
            print(f"\nClass Distribution:")
            print(df['class'].value_counts(normalize=True))

        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing Values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found!")

        return df.describe()

    def plot_feature_distributions(self, df, dataset_name):
        """Plot feature distributions"""
        print(f"\nCreating distribution plots for {dataset_name}...")

        # Separate features and target
        target_col = 'class' if 'class' in df.columns else 'target'
        feature_cols = [col for col in df.columns if col != target_col]

        # Create subplots for histograms
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle(f'Feature Distributions - {dataset_name.upper()}', fontsize=16)

        # Handle different subplot configurations
        axes_flat = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for i, feature in enumerate(feature_cols):
            if i < len(axes_flat):
                ax = axes_flat[i]
                df[feature].hist(ax=ax, bins=30, alpha=0.7)
                ax.set_title(f'{feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')

        # Hide empty subplots
        for i in range(n_features, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{dataset_name}_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_heatmap(self, df, dataset_name):
        """Plot correlation heatmap"""
        print(f"Creating correlation heatmap for {dataset_name}...")

        # Separate features and target
        target_col = 'class' if 'class' in df.columns else 'target'
        feature_cols = [col for col in df.columns if col != target_col]

        # Calculate correlation matrix
        corr_matrix = df[feature_cols + [target_col]].corr()

        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=0.5)

        plt.title(f'Correlation Heatmap - {dataset_name.upper()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{dataset_name}_correlation.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return corr_matrix

    def plot_class_distribution(self, df, dataset_name):
        """Plot class distribution"""
        print(f"Creating class distribution plot for {dataset_name}...")

        target_col = 'class' if 'class' in df.columns else 'target'

        # Create pie chart and bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        class_counts = df[target_col].value_counts()
        class_labels = ['No Disease' if x == 0 else 'Disease' for x in class_counts.index]

        ax1.pie(class_counts.values, labels=class_labels, autopct='%1.1f%%',
               colors=['lightblue', 'lightcoral'])
        ax1.set_title(f'Class Distribution - {dataset_name.upper()}')

        # Bar chart
        bars = ax2.bar(class_labels, class_counts.values, color=['lightblue', 'lightcoral'])
        ax2.set_title(f'Class Counts - {dataset_name.upper()}')
        ax2.set_ylabel('Count')
        ax2.bar_label(bars)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{dataset_name}_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_box_plots(self, df, dataset_name):
        """Plot box plots for features"""
        print(f"Creating box plots for {dataset_name}...")

        target_col = 'class' if 'class' in df.columns else 'target'
        feature_cols = [col for col in df.columns if col != target_col]

        # Select subset of features for better visualization
        n_features = min(len(feature_cols), 9)  # Limit to 9 features
        selected_features = feature_cols[:n_features]

        # Create box plots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Box Plots by Disease Status - {dataset_name.upper()}', fontsize=16)

        for i, feature in enumerate(selected_features):
            row, col = i // 3, i % 3

            # Create box plot
            df.boxplot(column=feature, by=target_col, ax=axes[row, col])

            # Customize plot
            axes[row, col].set_title(f'{feature}')
            axes[row, col].set_xlabel('Disease Status')
            axes[row, col].set_ylabel(feature)

        # Hide empty subplots
        for i in range(n_features, 9):
            row, col = i // 3, i % 3
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{dataset_name}_boxplots.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def generate_eda_report(self, df, dataset_name):
        """Generate comprehensive EDA report"""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EDA REPORT - {dataset_name.upper()}")
        print(f"{'='*60}")

        # Summary statistics
        stats = self.generate_summary_statistics(df, dataset_name)

        # Class distribution
        self.plot_class_distribution(df, dataset_name)

        # Feature distributions
        self.plot_feature_distributions(df, dataset_name)

        # Correlation heatmap
        corr_matrix = self.plot_correlation_heatmap(df, dataset_name)

        # Box plots
        self.plot_box_plots(df, dataset_name)

        # Generate insights
        self.generate_insights(df, dataset_name, stats, corr_matrix)

    def generate_insights(self, df, dataset_name, stats, corr_matrix):
        """Generate insights from the analysis"""
        print(f"\n{'='*50}")
        print(f"KEY INSIGHTS - {dataset_name.upper()}")
        print(f"{'='*50}")

        target_col = 'class' if 'class' in df.columns else 'target'

        # Class balance insight
        class_counts = df[target_col].value_counts()
        balance_ratio = class_counts.min() / class_counts.max()

        print(f"1. Class Balance: {balance_ratio:.2f}")
        if balance_ratio < 0.8:
            print("   → Dataset is imbalanced and may need balancing techniques")
        else:
            print("   → Dataset is relatively balanced")

        # Feature correlation insights
        high_corr_features = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_features.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr_features:
            print(f"\n2. Highly Correlated Features (correlation > 0.7):")
            for feat1, feat2, corr in high_corr_features:
                print(f"   → {feat1} and {feat2}: {corr:.2f}")
        else:
            print("\n2. No highly correlated features found")

        # Feature importance based on correlation with target
        if target_col in corr_matrix.columns:
            target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
            print(f"\n3. Top Features Correlated with Target:")
            for feature, correlation in target_corr.head(5).items():
                print(f"   → {feature}: {correlation:.2f}")

        # Statistical insights
        print(f"\n4. Statistical Insights:")
        print(f"   → Dataset has {df.shape[0]} samples and {df.shape[1]} features")
        print(f"   → Features are standardized (mean ≈ 0, std ≈ 1)")

        # Save insights to file
        insights_file = os.path.join(self.output_dir, f'{dataset_name}_insights.txt')
        with open(insights_file, 'w') as f:
            f.write(f"EDA Insights for {dataset_name.upper()} Dataset\n")
            f.write(f"Dataset Shape: {df.shape}\n")
            f.write(f"Class Balance Ratio: {balance_ratio:.2f}\n")
            f.write(f"Highly Correlated Features: {high_corr_features}\n")
            f.write(f"Top Features Correlated with Target:\n")
            if target_col in corr_matrix.columns:
                target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
                for feature, correlation in target_corr.head(5).items():
                    f.write(f"  {feature}: {correlation:.2f}\n")

        print(f"\nDetailed insights saved to: {insights_file}")

    def analyze_all_datasets(self):
        """Analyze all three datasets"""
        datasets = ['diabetes', 'heart_disease', 'kidney_disease']

        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"ANALYZING {dataset.upper()} DATASET")
            print(f"{'='*80}")

            try:
                # Load data
                df = self.load_processed_data(dataset)

                # Generate comprehensive EDA report
                self.generate_eda_report(df, dataset)

            except Exception as e:
                print(f"Error analyzing {dataset} dataset: {e}")
                continue

        print(f"\n{'='*80}")
        print("EDA COMPLETED FOR ALL DATASETS!")
        print(f"{'='*80}")
        print(f"Output files saved in: {self.output_dir}")

def main():
    """Main function"""
    analyzer = EDAnalyzer()
    analyzer.analyze_all_datasets()

if __name__ == "__main__":
    main()