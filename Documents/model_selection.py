import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Perform data preprocessing as needed for model selection."""
    # Implement data preprocessing steps here (if any).
    pass

def select_model(X, y):
    """Select the best machine learning model based on cross-validation."""
    # Create a list of models to evaluate
    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('SVM', SVC())
    ]

    # Evaluate each model using cross-validation
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"{name}: Accuracy = {scores.mean():.2f} ± {scores.std():.2f}")

def train_best_model(X_train, y_train, model_name):
    """Train the best machine learning model on the entire training set."""
    # Choose the best model based on the results from select_model function
    if model_name == 'Logistic Regression':
        model = LogisticRegression()
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_name == 'Random Forest':
        model = RandomForestClassifier()
    elif model_name == 'SVM':
        model = SVC()

    # Train the selected model on the entire training set
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the performance of the trained model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

if __name__ == "__main__":
    # Provide the file path of your dataset CSV
    data_file_path = "data.csv"
    df = load_data(data_file_path)

    # Preprocess the data
    preprocess_data(df)

    # Split the data into features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Model selection: Evaluate different models using cross-validation
    select_model(X, y)

    # Choose the best model based on the cross-validation results
    best_model_name = 'Random Forest'  # Replace with the best model from the cross-validation

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the best model on the entire training set
    best_model = train_best_model(X_train, y_train, best_model_name)

    # Evaluate the performance of the best model on the test set
    evaluate_model(best_model, X_test, y_test)
