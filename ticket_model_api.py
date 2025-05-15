# -*- coding: utf-8 -*-
"""
This module contains the core logic for training and using the ticket
prioritization model. It's designed to be imported by the Flask API (app.py).
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB # Using RandomForest instead
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from collections import Counter
# import matplotlib.pyplot as plt # Plotting removed for API context
# import seaborn as sns # Plotting removed for API context

# Define paths for saving/loading models and data within the container
MODEL_DIR = '/app/models'
DATA_DIR = '/app/data'
MODEL_PATH = os.path.join(MODEL_DIR, 'trained-model.pkl')
VECTORIZERS_PATH = os.path.join(MODEL_DIR, 'trained-vectorizers.pkl')
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, 'tickets_priorizados.csv')

# Ensure model and data directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def load_data(filename):
    """
    Loads data from a CSV file.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        pandas.DataFrame or None: A DataFrame with the loaded data, or None on error.
    """
    try:
        # Specify encoding to handle potential special characters
        df = pd.read_csv(filename, encoding='utf-8')
        print(f"Data loaded successfully from {filename}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find the file {filename}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def combine_text(df):
    """
    Combines title and description into a single 'combinedText' column.
    Handles potential missing columns gracefully.

    Args:
        df (pandas.DataFrame): The DataFrame with ticket data.

    Returns:
        pandas.DataFrame: The DataFrame with 'combinedText' added.
    """
    df['title'] = df['title'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str)
    df['combinedText'] = df['title'] + ' ' + df['description']
    print("Combined 'title' and 'description' into 'combinedText'.")
    return df

def preprocess_text_and_features(df):
    """
    Preprocesses text and extracts features like 'days_until_limit'.

    Args:
        df (pandas.DataFrame): The DataFrame with ticket data.

    Returns:
        pandas.DataFrame: The DataFrame with preprocessed text and added features.
    """
    def clean_text(text):
        # Keep alphanumeric and spaces
        text = ''.join(char for char in str(text).lower() if char.isalnum() or char.isspace())
        return text

    # Clean combined text
    if 'combinedText' in df.columns:
        df['combinedText'] = df['combinedText'].apply(clean_text)
    else:
        print("Warning: 'combinedText' column not found for cleaning.")

    # Process limit_date if exists
    if 'limit_date' in df.columns:
        # Attempt to parse date with common formats, handling errors
        df['limit_date'] = pd.to_datetime(df['limit_date'], errors='coerce', dayfirst=True)
        # Calculate days_until_limit only for valid dates
        now = pd.Timestamp.now().tz_localize(None) # Ensure timezone naive comparison
        df['days_until_limit'] = (df['limit_date'].dt.tz_localize(None) - now).dt.days
        # Fill NaT in limit_date with 0 days until limit and take absolute value
        df['days_until_limit'] = df['days_until_limit'].fillna(0).abs()
        print("Processed 'limit_date' and calculated 'days_until_limit'.")
    else:
        # If no limit_date, create days_until_limit with 0
        df['days_until_limit'] = 0
        print("Warning: 'limit_date' column not found. Setting 'days_until_limit' to 0.")

    # Clean other text columns if they exist
    for col in ['board_status_name', 'state_name']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).apply(clean_text)
            print(f"Cleaned '{col}'.")
        else:
            # Add the column with empty strings if it's missing
            df[col] = ''
            print(f"Warning: '{col}' column not found. Added as empty.")

    return df

def split_data_and_vectorize(df):
    """
    Vectorizes text features, combines with numerical features, splits data,
    and applies SMOTE.

    Args:
        df (pandas.DataFrame): The preprocessed DataFrame.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizers) or None on error.
    """
    if 'priority_level' not in df.columns:
        print("Error: Target column 'priority_level' not found in the training data.")
        return None

    if df['priority_level'].isnull().any():
        print("Warning: Found missing values in 'priority_level'. Dropping these rows for training.")
        df = df.dropna(subset=['priority_level'])
        if df.empty:
            print("Error: No valid data left after dropping rows with missing 'priority_level'.")
            return None

    try:
        # Initialize vectorizers
        text_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for efficiency
        board_vectorizer = TfidfVectorizer()
        state_vectorizer = TfidfVectorizer()

        # Fit and transform text features
        X_text = text_vectorizer.fit_transform(df['combinedText'])
        X_board_status = board_vectorizer.fit_transform(df['board_status_name'])
        X_state = state_vectorizer.fit_transform(df['state_name'])

        # Numerical features (ensure it's 2D)
        X_date = df[['days_until_limit']]

        # Combine features
        X = hstack([X_text, X_date, X_board_status, X_state]).tocsr() # Use CSR format

        # Target variable
        y = df['priority_level']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y # Stratify for imbalanced data
        )

        print(f"Original training distribution: {Counter(y_train)}")

        # Apply SMOTE only if there's more than one class and enough samples
        if len(Counter(y_train)) > 1:
             # Check minimum class size for SMOTE's k_neighbors
            min_class_count = min(Counter(y_train).values())
            k_neighbors = min(5, max(1, min_class_count - 1)) # Adjust k_neighbors

            if k_neighbors > 0:
                try:
                    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print(f"Distribution after SMOTE: {Counter(y_train)}")
                except ValueError as e:
                    print(f"Warning: Could not apply SMOTE due to insufficient class samples. {e}. Proceeding without oversampling.")
            else:
                 print("Warning: Not enough samples in the minority class to apply SMOTE. Proceeding without oversampling.")
        else:
            print("Warning: Only one class present in training data or SMOTE not applicable. Skipping SMOTE.")


        # Store vectorizers
        vectorizers = {
            'text': text_vectorizer,
            'board': board_vectorizer,
            'state': state_vectorizer
        }

        return X_train, X_test, y_train, y_test, vectorizers

    except Exception as e:
        print(f"Error during data splitting and vectorization: {e}")
        return None


def train_model_internal(X_train, y_train):
    """
    Trains a RandomForestClassifier model.

    Args:
        X_train (scipy.sparse matrix): Training feature data.
        y_train (pandas.Series): Training target labels.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained model.
    """
    print("Starting model training...")
    # Using RandomForestClassifier as per the original script logic
    model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced') # Added params
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns the classification report.

    Args:
        model: The trained model.
        X_test (scipy.sparse matrix): Test feature data.
        y_test (pandas.Series): Test target labels.

    Returns:
        str: The classification report.
    """
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    # Use zero_division=0 or 1 based on preference for warnings/output
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Model evaluation completed.")
    # print(report) # Print report here for server logs

    # Optional: Confusion Matrix (cannot display plot in API, but can log)
    # cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    # print("Confusion Matrix:\n", cm)

    return report

def save_model_and_vectorizers(model, vectorizers):
    """
    Saves the trained model and vectorizers to disk.

    Args:
        model: The trained model object.
        vectorizers (dict): Dictionary containing the fitted vectorizers.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved successfully to {MODEL_PATH}")
        joblib.dump(vectorizers, VECTORIZERS_PATH)
        print(f"Vectorizers saved successfully to {VECTORIZERS_PATH}")
        return True
    except Exception as e:
        print(f"Error saving model or vectorizers: {e}")
        return False

# --- Main API Functions ---

def perform_training(data_path):
    """
    Orchestrates the model training process using data from data_path.

    Args:
        data_path (str): Path to the training CSV file.

    Returns:
        tuple: (success (bool), message (str))
    """
    print(f"--- Starting Training Process for {data_path} ---")
    df = load_data(data_path)
    if df is None:
        return False, "Failed to load training data."

    df = combine_text(df)
    df = preprocess_text_and_features(df)
    if df is None:
         return False, "Failed during preprocessing."

    split_result = split_data_and_vectorize(df)
    if split_result is None:
        return False, "Failed during data splitting/vectorization."

    X_train, X_test, y_train, y_test, vectorizers = split_result

    if X_train.shape[0] == 0 or y_train.empty:
         return False, "No training data available after preprocessing and splitting."

    model = train_model_internal(X_train, y_train)
    if model is None:
        return False, "Model training failed."

    # Evaluate (optional, good for logging)
    try:
        if X_test.shape[0] > 0 and not y_test.empty:
            report = evaluate_model(model, X_test, y_test)
            print("Evaluation Report on Test Set:\n", report)
        else:
            print("Skipping evaluation due to empty test set.")
    except Exception as e:
        print(f"Warning: Error during evaluation: {e}")


    if not save_model_and_vectorizers(model, vectorizers):
        return False, "Failed to save the trained model or vectorizers."

    print("--- Training Process Completed Successfully ---")
    return True, "Model trained and saved successfully."


def perform_prediction(data_path):
    """
    Loads the trained model and vectorizers to perform predictions on new data.

    Args:
        data_path (str): Path to the CSV file with data for prediction.

    Returns:
        tuple: (success (bool), result_message_or_path (str))
               If successful, returns True and the path to the output CSV.
               If failed, returns False and an error message.
    """
    print(f"--- Starting Prediction Process for {data_path} ---")

    # Load data for prediction
    df_predict = load_data(data_path)
    if df_predict is None:
        return False, f"Failed to load prediction data from {data_path}"

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        return False, f"Error: Model file not found at {MODEL_PATH}. Please train the model first."
    except Exception as e:
        return False, f"Error loading model: {e}"

    # Load vectorizers
    try:
        vectorizers = joblib.load(VECTORIZERS_PATH)
        print(f"Vectorizers loaded successfully from {VECTORIZERS_PATH}")
    except FileNotFoundError:
        return False, f"Error: Vectorizers file not found at {VECTORIZERS_PATH}. Please train the model first."
    except Exception as e:
        return False, f"Error loading vectorizers: {e}"

    try:
        # Preprocess the prediction data using the same steps as training
        df_predict = combine_text(df_predict)
        df_predict = preprocess_text_and_features(df_predict) # Ensure 'days_until_limit' etc. are handled

        # Vectorize using loaded vectorizers (use transform, not fit_transform)
        X_text = vectorizers['text'].transform(df_predict['combinedText'])
        X_board_status = vectorizers['board'].transform(df_predict['board_status_name'])
        X_state = vectorizers['state'].transform(df_predict['state_name'])

        # Numerical features (ensure it's 2D)
        X_date = df_predict[['days_until_limit']]

        # Combine features in the same order as training
        X_combined = hstack([X_text, X_date, X_board_status, X_state]).tocsr()

        # Make predictions
        print("Making predictions...")
        predictions = model.predict(X_combined)
        df_predict['predicted_priority_level'] = predictions
        print("Predictions completed.")

        # Save results to CSV
        df_predict.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"Predictions saved successfully to {OUTPUT_CSV_PATH}")

        print("--- Prediction Process Completed Successfully ---")
        return True, OUTPUT_CSV_PATH

    except KeyError as e:
         return False, f"Error during prediction: Missing expected column or vectorizer key: {e}. Ensure input CSV has required columns (title, description, optional: limit_date, board_status_name, state_name) and the model was trained correctly."
    except Exception as e:
        return False, f"Error during prediction processing: {e}"

# Example of how to potentially call these functions (for testing, not used by Flask app directly)
# if __name__ == '__main__':
#     # Create dummy data for testing if needed
#     # train_success, train_msg = perform_training('path/to/your/train.csv')
#     # print(train_msg)
#     # if train_success:
#     #     predict_success, predict_msg_or_path = perform_prediction('path/to/your/predict.csv')
#     #     print(predict_msg_or_path)
#     pass
