# test_dummy.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier

# Path to your CSV dataset
CSV_PATH = 'semiconductor_data_with_available_features.csv'

# List of numerical features to use
NUMERICAL_FEATURES = [
    'Band Gap (eV)',
    'Density (g/cm³)',
    'Volume (Å³)',
    'Number of Sites',
    'Formation Energy per Atom (eV)',
    'Energy Above Hull (eV)',
    'Fermi Energy (eV)'
]

# Load the CSV data
def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    return df

# Preprocess: split, SMOTE, scale
def preprocess_data(df):
    # Extract X and y
    X = df[NUMERICAL_FEATURES].copy()
    y = df['Is Metal'].astype(int)

    # Fill missing values
    X = X.fillna(X.mean())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    # Apply SMOTE on training data only
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_res, y_test


def main():
    # Load and preprocess
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Flatten (DummyClassifier expects 2D)
    # Already 2D: (n_samples, n_features)

    # Train DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)

    # Evaluate
    accuracy = dummy.score(X_test, y_test)
    print(f"Dummy accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()