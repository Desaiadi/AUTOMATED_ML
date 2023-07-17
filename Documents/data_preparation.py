import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Clean the data by handling missing values and duplicates."""
    # Handle missing values (you can customize this step as needed)
    df.fillna(df.mean(), inplace=True)

    # Drop duplicate rows, if any
    df.drop_duplicates(inplace=True)

def remove_outliers(df, columns):
    """Remove outliers from the specified columns using z-score method."""
    from scipy import stats

    for col in columns:
        z_scores = stats.zscore(df[col])
        df = df[(z_scores < 3) & (z_scores > -3)]

    return df

def transform_data(df):
    """Transform the data as needed for machine learning algorithms."""
    # You can add more data transformations here as per your specific needs.
    pass

def generate_descriptive_statistics(df):
    """Generate descriptive statistics for the data."""
    statistics = df.describe()
    print(statistics)

def generate_visualizations(df):
    """Generate graphical visualizations of the data."""
    # You can create different visualizations based on the nature of your data and problem.
    # Here, we'll create a histogram for each numerical column.
    for col in df.select_dtypes(include='number'):
        plt.hist(df[col], bins=20)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {col}')
        plt.show()

if __name__ == "__main__":
    # Provide the file path of your dataset CSV
    data_file_path = "data.csv"
    df = load_data(data_file_path)

    # Clean the data
    clean_data(df)

    # Remove outliers from numerical columns (you can customize this step as needed)
    numerical_columns = df.select_dtypes(include='number').columns
    df = remove_outliers(df, numerical_columns)

    # Transform the data (you can customize this step as needed)
    transform_data(df)

    # Generate descriptive statistics
    generate_descriptive_statistics(df)

    # Generate graphical visualizations
    generate_visualizations(df)
