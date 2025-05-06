import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
import joblib

# GPU check and setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s): {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth enabled on all GPUs")
else:
    print("No GPUs detected")
tf.config.experimental.enable_tensor_float_32_execution(True)
print("TF32 precision enabled for faster computation")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Feature and label configuration
FEATURES = [
    "Density (g/cm³)", "Formation Energy per Atom (eV)", "Fermi Energy (eV)",
    "Is Metal", "Direct Band Gap", "Conduction Band Minimum (eV)", "Valence Band Maximum (eV)"
]
LABEL_COLS = [
    "Band Gap (eV)", "Energy Above Hull (eV)", "Fermi Energy (eV)", "Is Metal"
]

cols_to_load = list(set(FEATURES + LABEL_COLS + ["Symmetry", "Is Stable"]))
print(f"Columns to load: {cols_to_load}")

# Load data
df = pd.read_csv("semiconductor_data_with_available_features.csv", usecols=cols_to_load, engine='pyarrow')
print(f"Initial shape after loading specific columns: {df.shape}")

# Check for extreme values and outliers before preprocessing
for col in df.select_dtypes(include=['number']).columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr
    outliers = df[(df[col] > upper_limit) | (df[col] < lower_limit)][col]
    if not outliers.empty:
        print(f"Found {len(outliers)} outliers in {col}")

# Labeling logic
symmetry_str = df["Symmetry"].astype(str)
condition_is_metal = (df["Is Metal"] == 1)
condition_band_gap_invalid = ~((df["Band Gap (eV)"] > 0.1) & (df["Band Gap (eV)"] <= 4.0))
condition_above_hull = (df["Energy Above Hull (eV)"] > 0.1)
condition_not_stable = (df["Is Stable"] == False)
condition_fermi_invalid = (df["Fermi Energy (eV)"] < -5) | (df["Fermi Energy (eV)"] > 5)
condition_invalid_symmetry = ~(symmetry_str.str.contains("F-43m", case=False) | symmetry_str.str.contains("Fd-3m", case=False))

is_not_semiconductor = (condition_is_metal | condition_band_gap_invalid | condition_above_hull |
                        condition_not_stable | condition_fermi_invalid | condition_invalid_symmetry)

df["is_semiconductor"] = (~is_not_semiconductor).astype("float32")
print("Vectorized labeling completed.")
print(f"Class distribution before balancing: {df['is_semiconductor'].value_counts()}")

# Prepare features and labels
y = df["is_semiconductor"]

# Available features based on the input dataset
AVAILABLE_FEATURES = [c for c in FEATURES if c in df.columns]
X_df = df[AVAILABLE_FEATURES].copy()
print(f"Using features: {AVAILABLE_FEATURES}")

# Drop fully missing columns
cols_to_drop = [col for col in AVAILABLE_FEATURES if X_df[col].isna().all()]
if cols_to_drop:
    print(f"Columns dropped due to 100% NaN: {cols_to_drop}")
X_df = X_df.drop(columns=cols_to_drop)
AVAILABLE_FEATURES = [c for c in AVAILABLE_FEATURES if c not in cols_to_drop]

# More robust outlier handling - clip extreme values
for col in X_df.select_dtypes(include=['number']).columns:
    q1 = X_df[col].quantile(0.01)  # More conservative percentiles
    q3 = X_df[col].quantile(0.99)
    X_df[col] = X_df[col].clip(q1, q3)

# Impute missing values with median (more robust than mean)
numeric_cols = X_df.select_dtypes(include="number").columns
cat_cols = X_df.select_dtypes(exclude="number").columns
fill_values = {}
if not numeric_cols.empty:
    fill_values.update(X_df[numeric_cols].median().to_dict())
if not cat_cols.empty:
    fill_values.update(X_df[cat_cols].mode().iloc[0].to_dict())
X_df.fillna(fill_values, inplace=True)

# Convert categorical to int (if any)
for col in cat_cols:
    if col in X_df.columns:
        X_df[col] = X_df[col].astype(int)
print(f"Shape after preprocessing X: {X_df.shape}")
print(f"Class balance:\n{y.value_counts(normalize=True)}")

# More efficient approach to one-hot encoding for Symmetry
# Limit to top N most common categories to prevent dimension explosion
top_n = 10  # Only keep top 10 most common symmetry types
top_symmetry = symmetry_str.value_counts().nlargest(top_n).index
symmetry_filtered = symmetry_str.apply(lambda x: x if x in top_symmetry else 'Other')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
symmetry_one_hot = encoder.fit_transform(symmetry_filtered.values.reshape(-1, 1))
symmetry_one_hot_df = pd.DataFrame(
    symmetry_one_hot, 
    columns=[f"Symmetry_{cat}" for cat in encoder.categories_[0]]
)
joblib.dump(encoder, "symmetry_encoder.pkl")

# Add the one-hot encoded Symmetry features to the feature set
X_df = pd.concat([X_df, symmetry_one_hot_df], axis=1)
print(f"Updated feature shape after adding Symmetry: {X_df.shape}")

# In MLP.py, after filling NaNs and before splitting/scaling
numeric_cols = X_df.select_dtypes(include="number").columns # Ensure this matches EXPECTED_NUMERICAL_FEATURES if possible
median_values_to_save = X_df[numeric_cols].median()
joblib.dump(median_values_to_save, 'median_values.pkl')
print("Saved median values for imputation to median_values.pkl")

# Split data BEFORE applying SMOTE to prevent data leakage
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X_df.values, y.to_numpy(), test_size=0.2, random_state=42, stratify=y.to_numpy()
)

# Create a validation set from the training data
X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(
    X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw
)

# X_train should have all 18 features here (after symmetry one-hot)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw.astype("float32"))

# Save the scaler
import joblib
joblib.dump(scaler, "scaler.pkl")

X_val_scaled = scaler.transform(X_val_raw.astype("float32"))  # Transform validation data
X_test_scaled = scaler.transform(X_test_raw.astype("float32"))  # Transform test data
print("Features standardized")

# Apply moderate SMOTE only to training data
print("Applying SMOTE for class imbalance (training data only)...")
smote = SMOTE(sampling_strategy=0.7, random_state=42)  # Less aggressive oversampling
X_train, y_train = smote.fit_resample(X_train_scaled, y_train_raw)
print(f"After SMOTE, training class distribution: {np.bincount(y_train.astype('int'))}")
print(f"Validation class distribution: {np.bincount(y_val.astype('int'))}")
print(f"Test class distribution: {np.bincount(y_test.astype('int'))}")

# tf.data pipeline
BATCH_SIZE = 256  # Smaller batch size
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_features(x, y):
    return x, y

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=len(X_train))
    .batch(BATCH_SIZE)
    .map(preprocess_features, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val))
    .batch(BATCH_SIZE)
    .map(preprocess_features, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test))
    .batch(BATCH_SIZE)
    .map(preprocess_features, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

print("tf.data pipelines built.")

# Updated Model: Simpler and more regularized
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Lower initial learning rate
initial_learning_rate = 0.0005

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# Callbacks — tuned for anti-overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,            # Earlier stopping
    restore_best_weights=True,
    min_delta=0.0005       # Stricter improvement requirement
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

import datetime
import os
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Train the model — with fewer max epochs
history = model.fit(
    train_ds,
    epochs=30,             # Reduced maximum epochs
    validation_data=val_ds,
    callbacks=[tensorboard_callback, early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model on test set
test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(test_ds, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

model_save_path = f"semiconductor_classifier_tf32_batch{BATCH_SIZE}.keras"
model.save(model_save_path)
print(f"Saved model to {model_save_path}")

# Predictions and additional metrics
# Evaluate on test set again (with manual predictions)
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype("int32").flatten()

y_true = y_test.astype("int32")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Semiconductor", "Semiconductor"])
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved to confusion_matrix.png")

# F1 and MCC
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

loss, acc, auc, precision, recall = model.evaluate(val_ds, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

# Helper to pad history up to 50 epochs
def pad_history(metric_name, history_dict, target_len=50):
    values = history_dict.get(metric_name, [])
    if len(values) < target_len:
        last_val = values[-1] if values else 0
        values += [last_val] * (target_len - len(values))
    return values[:target_len]

# Epoch range
epochs_range = range(50)

# 2. Learning Rate Plot — only available if you use a scheduler or callback that changes it
# If lr not tracked in history, skip this
if 'lr' in history.history:
    lr_history = pad_history('lr', history.history)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, lr_history, label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate During Training')
    plt.legend()
    plt.savefig("learning_rate_plot.png")
    plt.close()
    print("Learning rate plot saved to learning_rate_plot.png")

# 3. Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, pad_history('accuracy', history.history), label='Train Accuracy', color='blue')
plt.plot(epochs_range, pad_history('val_accuracy', history.history), label='Test Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy During Training')
plt.legend()
plt.savefig("accuracy_plot.png")
plt.close()
print("Model accuracy plot saved to accuracy_plot.png")

# 4. Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, pad_history('loss', history.history), label='Train Loss', color='green')
plt.plot(epochs_range, pad_history('val_loss', history.history), label='Test Loss', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss During Training')
plt.legend()
plt.savefig("loss_plot.png")
plt.close()
print("Model loss plot saved to loss_plot.png")

# 5. AUC Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, pad_history('auc', history.history), label='Train AUC', color='cyan')
plt.plot(epochs_range, pad_history('val_auc', history.history), label='Test AUC', color='magenta')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Model AUC During Training')
plt.legend()
plt.savefig("auc_plot.png")
plt.close()
print("Model AUC plot saved to auc_plot.png")

# 6. Precision and Recall Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, pad_history('precision', history.history), label='Train Precision', color='yellow')
plt.plot(epochs_range, pad_history('val_precision', history.history), label='Test Precision', color='green')
plt.plot(epochs_range, pad_history('recall', history.history), label='Train Recall', color='blue')
plt.plot(epochs_range, pad_history('val_recall', history.history), label='Test Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Precision / Recall')
plt.title('Model Precision and Recall During Training')
plt.legend()
plt.savefig("precision_recall_plot.png")
plt.close()
print("Precision and recall plot saved to precision_recall_plot.png")


