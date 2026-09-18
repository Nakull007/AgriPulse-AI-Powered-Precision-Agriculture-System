# =============================================================================
# TRAIN_MODEL_FROM_PRESPLIT.PY
#
# Author: [Your Name/Team Name]
# Date: [Date]
#
# Description:
# This script trains a RandomForestClassifier model using pre-split and
# preprocessed data files. It loads the training and testing sets directly,
# trains the model, evaluates it, and saves the final artifact.
# =============================================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
# Update these paths to the exact location of your files
DATA_PATH = r"D:\ML projects\sprinkler_system\Dataset\\"
X_TRAIN_PATH = DATA_PATH + "X_train.csv"
Y_TRAIN_PATH = DATA_PATH + "y_train.csv"
X_TEST_PATH = DATA_PATH + "X_test.csv"
Y_TEST_PATH = DATA_PATH + "y_test.csv"

MODEL_SAVE_PATH = 'sprinkler_model.pkl'
RANDOM_STATE_SEED = 42

def train_from_preprocessed_data():
    """
    The main function to train a model from pre-split data files.
    """
    print("--- Starting AI Model Training from Pre-split Data ---")

    # 1. Load Pre-split Data
    try:
        print("[1/4] Loading pre-split training and testing data...")
        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).squeeze() # .squeeze() converts a single-column DataFrame to a Series
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()
        
        print("      Data loaded successfully:")
        print(f"      - X_train shape: {X_train.shape}")
        print(f"      - y_train shape: {y_train.shape}")
        print(f"      - X_test shape:  {X_test.shape}")
        print(f"      - y_test shape:  {y_test.shape}")

    except FileNotFoundError as e:
        print(f"      ERROR: Could not find a data file. Please check your paths.")
        print(f"      File not found: {e.filename}")
        return

    # 2. Define and Train the Model
    # Since preprocessing is already done, we don't need a Pipeline.
    # We can use the RandomForestClassifier directly.
    print("[2/4] Defining the RandomForestClassifier model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE_SEED,
        oob_score=True, # Out-of-bag score is a nice way to get a performance estimate
        n_jobs=-1       # Use all available CPU cores for faster training
    )
    
    print("[3/4] Training the model on the training data...")
    model.fit(X_train, y_train)
    print(f"      Model training complete. Out-of-Bag Score: {model.oob_score_:.4f}")

    # 4. Evaluate and Save the Final Model
    print("[4/4] Evaluating model and saving the final artifact...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy on unseen test data: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    target_names = [f'State {i}' for i in sorted(y_train.unique())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("---------------------------------")
    
    # Save the trained model to a file
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n✅ SUCCESS: Model has been saved to '{MODEL_SAVE_PATH}'")
    print("--- Training Pipeline Finished ---")


if __name__ == '__main__':
    train_from_preprocessed_data()