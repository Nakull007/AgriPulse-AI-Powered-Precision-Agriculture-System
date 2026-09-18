import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_PATH = r"D:\ML projects\sprinkler_system\Dataset\\"
MODEL_PATH = 'sprinkler_model.pkl'

# Define a consistent color palette and labels for all plots
ACTION_STATE_PALETTE = {
    0: "green", 1: "blue", 2: "red", 3: "orange", 4: "purple"
}
ACTION_STATE_LABELS = {
    0: 'Do Nothing', 1: 'Sprinkle (Normal)', 2: 'Sprinkle (High EC Alert)',
    3: 'Sprinkle (Low EC Warn)', 4: 'Fertigate Alert'
}

# --- Load Model and Test Data ---
try:
    print("Loading model and test data...")
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(DATA_PATH + "X_test.csv")
    y_test = pd.read_csv(DATA_PATH + "y_test.csv").squeeze()
    print("Model and data loaded successfully.")

except FileNotFoundError as e:
    print(f"ERROR: Could not find a necessary file. Please check your paths.")
    print(f"File not found: {e.filename}")
    exit()

# --- Make Predictions ---
print("Making predictions on the test data...")
y_pred = model.predict(X_test)
print("Predictions complete.")

# --- Prepare DataFrame for Plotting ---
# Create a new DataFrame from the test features
plot_df = X_test.copy()
# Add a column for the TRUE labels
plot_df['actual_state'] = y_test
# Add a column for the PREDICTED labels
plot_df['predicted_state'] = y_pred

# Find the misclassified points to highlight them
misclassified_points = plot_df[plot_df['actual_state'] != plot_df['predicted_state']]
print(f"\nFound {len(misclassified_points)} misclassified points out of {len(plot_df)}.")

print("\nGenerating 'Actual vs. Predicted' comparison plot...")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharex=True, sharey=True)
fig.suptitle('Model Predictions vs. Actual Labels on Test Data', fontsize=20)

# --- Plot 1: Ground Truth (Actual Labels) ---
sns.scatterplot(
    ax=axes[0],
    data=plot_df,
    x='soil_moisture',
    y='soil_ec',
    hue='actual_state',  # Color by the TRUE state
    palette=ACTION_STATE_PALETTE,
    alpha=0.7,
    s=50 # Point size
)
axes[0].set_title('Ground Truth (Actual Labels)', fontsize=16)
axes[0].set_xlabel('Soil Moisture (Higher is Drier)', fontsize=12)
axes[0].set_ylabel('Soil EC (mS/cm)', fontsize=12)
axes[0].legend_.set_visible(False) # Hide legend for clarity

# --- Plot 2: Model's Predictions ---
sns.scatterplot(
    ax=axes[1],
    data=plot_df,
    x='soil_moisture',
    y='soil_ec',
    hue='predicted_state', # Color by the PREDICTED state
    palette=ACTION_STATE_PALETTE,
    alpha=0.7,
    s=50
)
axes[1].set_title("Model's Predictions", fontsize=16)
axes[1].set_xlabel('Soil Moisture (Higher is Drier)', fontsize=12)
axes[1].set_ylabel('')

# --- Highlight Misclassified Points on the Prediction Plot ---
# We can circle the points where the model made a mistake
if not misclassified_points.empty:
    axes[1].scatter(
        misclassified_points['soil_moisture'],
        misclassified_points['soil_ec'],
        s=200,            # Make the circle larger
        facecolors='none', # Make it an empty circle
        edgecolors='yellow', # Use a bright color
        linewidth=2,
        label='Misclassified'
    )

# Create a single, shared legend
handles, _ = axes[1].get_legend_handles_labels()
# Manually create legend labels to match the full set of states
legend_labels = [ACTION_STATE_LABELS[i] for i in sorted(ACTION_STATE_PALETTE.keys())]
if not misclassified_points.empty:
    legend_labels.append('Misclassified') # Add our custom label if needed

fig.legend(handles, legend_labels, title='Action State', loc='upper right', bbox_to_anchor=(0.99, 0.85))
axes[1].legend_.set_visible(False)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# If there were any errors, print them out for detailed analysis
if not misclassified_points.empty:
    print("\n--- Details of Misclassified Points ---")
    # Add the actual and predicted labels as text for easier reading
    misclassified_points['actual_label'] = misclassified_points['actual_state'].map(ACTION_STATE_LABELS)
    misclassified_points['predicted_label'] = misclassified_points['predicted_state'].map(ACTION_STATE_LABELS)
    print(misclassified_points[['soil_moisture', 'soil_ec', 'actual_label', 'predicted_label']])