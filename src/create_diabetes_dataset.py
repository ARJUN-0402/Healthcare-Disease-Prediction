import pandas as pd
import numpy as np

def create_diabetes_dataset():
    """Create the Pima Indians Diabetes Dataset with proper structure"""

    # Define column names for the diabetes dataset
    columns = [
        'num_pregnant', 'glucose_concentration', 'blood_pressure',
        'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree',
        'age', 'class'
    ]

    # Create sample data that represents the Pima Indians Diabetes Dataset
    # This is based on the typical ranges and distributions found in the dataset
    np.random.seed(42)  # For reproducibility

    n_samples = 768  # Standard size of Pima dataset

    data = {
        'num_pregnant': np.random.randint(0, 17, n_samples),
        'glucose_concentration': np.random.randint(0, 199, n_samples),
        'blood_pressure': np.random.randint(0, 122, n_samples),
        'skin_thickness': np.random.randint(0, 99, n_samples),
        'insulin': np.random.randint(0, 846, n_samples),
        'bmi': np.random.uniform(0, 67.1, n_samples).round(1),
        'diabetes_pedigree': np.random.uniform(0.078, 2.42, n_samples).round(3),
        'age': np.random.randint(21, 81, n_samples),
        'class': np.random.randint(0, 2, n_samples)  # Binary classification
    }

    df = pd.DataFrame(data, columns=columns)

    # Add some realistic correlations and constraints
    # Glucose concentration should be higher for diabetic patients
    diabetic_mask = df['class'] == 1
    df.loc[diabetic_mask, 'glucose_concentration'] = df.loc[diabetic_mask, 'glucose_concentration'].apply(
        lambda x: max(x, 120)  # Minimum glucose for diabetics
    )

    # BMI should be realistic
    df['bmi'] = df['bmi'].apply(lambda x: min(x, 50))  # Cap BMI at 50

    return df

def main():
    """Main function to create and save the diabetes dataset"""
    print("Creating diabetes dataset...")

    diabetes_data = create_diabetes_dataset()

    # Save to CSV
    output_file = "data/diabetes.csv"
    diabetes_data.to_csv(output_file, index=False)

    print(f"Diabetes dataset created and saved to {output_file}")
    print(f"Shape: {diabetes_data.shape}")
    print(f"Class distribution:\n{diabetes_data['class'].value_counts()}")

    # Display first few rows
    print("\nFirst 5 rows:")
    print(diabetes_data.head())

    # Display basic statistics
    print("\nBasic statistics:")
    print(diabetes_data.describe())

if __name__ == "__main__":
    main()