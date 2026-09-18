import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# --- Configuration ---
DATASET_PATH = os.path.join(os.getcwd(), 'Dataset', 'NE_India_tea_garden_data.csv')
OUTPUT_FEATURES_PATH = os.path.join(os.getcwd(), 'processed_features.csv')
OUTPUT_LABELS_PATH = os.path.join(os.getcwd(), 'processed_labels.csv')


def preprocess_data_for_training():
    """
    Loads and preprocesses the tea garden dataset,
    applies one-hot encoding, and saves processed features and labels.
    """

    # 1. Load the Dataset
    print(f"📥 Step 1: Loading Dataset from '{DATASET_PATH}'")
    if not os.path.isfile(DATASET_PATH):
        print(f"\n❌ Error: Dataset file not found at '{DATASET_PATH}'.")

        print("\n🔍 Available CSV files in the current directory:")
        for file in os.listdir(os.getcwd()):
            if file.endswith('.csv'):
                print(f" - {file}")
        return None, None

    df = pd.read_csv(DATASET_PATH)

    print("✅ Dataset Loaded. Sample:")
    print(df.head())
    print("\n📊 Data Types:")
    print(df.dtypes)

    # 2. Separate Features and Labels
    print("\n✂️ Step 2: Separating Features (X) and Label (y)")
    X_raw = df.drop('action_state', axis=1)
    y = df['action_state']

    print("\n🧾 Features before preprocessing:")
    print(X_raw.head())
    print("\n🎯 Labels:")
    print(y.head())

    # 3. Preprocess Features
    print("\n🧪 Step 3: One-Hot Encoding 'season' Column")
    categorical_features = ['season']
    numerical_features = X_raw.columns.drop(categorical_features)

    print(f"\n🔤 Categorical Feature: {categorical_features}")
    print(f"🔢 Numerical Features: {list(numerical_features)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X_raw)
    new_cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_columns = list(new_cat_columns) + list(numerical_features)

    X_processed_df = pd.DataFrame(X_processed, columns=all_columns)

    print("\n✅ Features after One-Hot Encoding:")
    print(X_processed_df.head())
    print("\n📝 'season' column converted to binary features. Data is ready for ML training.")

    # 4. Save Preprocessed Files
    print("\n💾 Step 4: Saving Preprocessed Data")
    X_processed_df.to_csv(OUTPUT_FEATURES_PATH, index=False)
    y.to_csv(OUTPUT_LABELS_PATH, index=False)

    print(f"📁 Features saved to: {OUTPUT_FEATURES_PATH}")
    print(f"📁 Labels saved to: {OUTPUT_LABELS_PATH}")

    return X_processed_df, y

# --- Entry Point ---
if __name__ == '__main__':
    X_final, y_final = preprocess_data_for_training()

    if X_final is not None:
        print("\n✅ Preprocessing Complete")
        print("📐 Shape of features (X):", X_final.shape)
        print("🎯 Shape of labels (y):", y_final.shape)
