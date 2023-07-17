import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def create_new_features(df):
    """Create new features from existing data."""
    # Example: Let's create a new feature by taking the square root of a numerical column.
    df['sqrt_feature'] = np.sqrt(df['numerical_feature'])

    # You can add more feature engineering steps here based on your domain knowledge.

def evaluate_new_features(df, target_column):
    """Evaluate the effectiveness of new features using statistical methods."""
    # Example: Evaluate the correlation between new features and the target variable.
    new_features = df.select_dtypes(include='number').columns.drop(target_column)
    for feature in new_features:
        correlation, _ = pearsonr(df[feature], df[target_column])
        print(f"Correlation between {feature} and {target_column}: {correlation}")

    # Example: Evaluate the p-value of the new features using linear regression.
    X = df[new_features]
    X = sm.add_constant(X)  # Add a constant term for the intercept in linear regression
    y = df[target_column]
    model = sm.OLS(y, X).fit()
    print("\nP-values of new features:")
    print(model.pvalues)

if __name__ == "__main__":
    # Provide the file path of your dataset CSV
    data_file_path = "data.csv"
    target_column = "target"  # Replace with the actual name of your target variable
    df = load_data(data_file_path)

    # Feature engineering: Create new features
    create_new_features(df)

    # Evaluate the effectiveness of new features
    evaluate_new_features(df, target_column)
