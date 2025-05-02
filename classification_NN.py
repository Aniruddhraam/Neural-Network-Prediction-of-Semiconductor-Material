import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time
import matplotlib.pyplot as plt
import joblib
import pyarrow
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef

print("TensorFlow version:", tf.__version__)

# --- GPU + performance setup --- (Keep this section as is)
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


# --- Optimized Data loading & preprocessing ---
start_time = time.time()

# Define features needed for the model AND columns needed for labeling
FEATURES = [
    "Density (g/cm³)",
    "Formation Energy per Atom (eV)",
    "Fermi Energy (eV)", # Also needed for labeling
    "Is Metal", # Also needed for labeling
    "Direct Band Gap",
    "Conduction Band Minimum (eV)",
    "Valence Band Maximum (eV)"
]
# Add other columns explicitly needed only for labeling
LABEL_COLS = [
    "Band Gap (eV)",
    "Energy Above Hull (eV)",
    # Add Fermi Energy and Is Metal if they weren't already in FEATURES
    "Fermi Energy (eV)",
    "Is Metal"
]
# Combine and find unique columns to load
cols_to_load = list(set(FEATURES + LABEL_COLS))
print(f"Columns to load: {cols_to_load}")

# Load only necessary columns using pyarrow engine directly
# Removed the try-except fallback block
print("Using 'pyarrow' engine for pd.read_csv")
df = pd.read_csv(
        "semiconductor_data.csv",
        usecols=cols_to_load,
        engine='pyarrow' # Directly specify pyarrow
        )
print(f"Initial shape after loading specific columns: {df.shape}")

# --- Vectorized Labeling ---
# Define conditions for being NOT a semiconductor (label 0)
# Using vectorized operations instead of df.apply for speed
condition_is_metal = (df["Is Metal"] == 1)
condition_band_gap_low = (df["Band Gap (eV)"] <= 0.1)
condition_band_gap_high = (df["Band Gap (eV)"] > 4.0)
condition_above_hull = (df["Energy Above Hull (eV)"] > 0.1)
condition_fermi_low = (df["Fermi Energy (eV)"] < -5)
condition_fermi_high = (df["Fermi Energy (eV)"] > 5)

# Combine conditions using logical OR (|). If ANY are true, it's not a semiconductor.
# Handle potential NaNs in condition columns before comparison
# Fill NaNs in a way that they won't wrongly trigger the conditions
# (e.g., fill NaNs in 'Is Metal' with 0, 'Band Gap' with a value between 0.1 and 4.0 etc.)
# A simpler approach for now: just check the conditions, NaNs will yield False in comparisons.
is_not_semiconductor = (
    condition_is_metal |
    condition_band_gap_low |
    condition_band_gap_high |
    condition_above_hull |
    condition_fermi_low |
    condition_fermi_high
)

# The label 'y' is 1 if NONE of the 'is_not_semiconductor' conditions are met.
# So, y is the negation (~) of the combined condition.
y = (~is_not_semiconductor).astype("float32")
print("Vectorized labeling completed.")

# --- Feature Preprocessing ---
# Select only the feature columns for X
# Check which FEATURES actually exist in the loaded df (robustness)
AVAILABLE_FEATURES = [c for c in FEATURES if c in df.columns]
print(f"Using features: {AVAILABLE_FEATURES}")
X_df = df[AVAILABLE_FEATURES].copy() # Create X_df from the available features

# Drop columns that are 100% NaN *within the selected features*
cols_to_drop = [col for col in AVAILABLE_FEATURES if X_df[col].isna().all()]
if cols_to_drop:
    print(f"Columns dropped due to 100% NaN: {cols_to_drop}")
    X_df = X_df.drop(columns=cols_to_drop)
    # Update AVAILABLE_FEATURES list
    AVAILABLE_FEATURES = [c for c in AVAILABLE_FEATURES if c not in cols_to_drop]

# Optimized Imputation
# 1. Calculate medians/modes for relevant columns
numeric_cols = X_df.select_dtypes(include="number").columns
cat_cols = X_df.select_dtypes(exclude="number").columns

fill_values = {}
# Numeric imputation
if not numeric_cols.empty:
    medians = X_df[numeric_cols].median()
    fill_values.update(medians.to_dict())

# Categorical imputation (assuming they should be int after filling)
if not cat_cols.empty:
    # Calculate mode for each categorical column
    # mode() can return multiple values if they have the same frequency, so take the first [0]
    modes = X_df[cat_cols].mode().iloc[0]
    fill_values.update(modes.to_dict())

# 2. Apply fillna once using the dictionary
X_df.fillna(fill_values, inplace=True)
print("NaN imputation completed.")

# 3. Ensure correct types (especially for categorical after fillna)
for col in cat_cols:
   if col in X_df.columns: # Check if column wasn't dropped
        X_df[col] = X_df[col].astype(int)

print(f"Shape after preprocessing X: {X_df.shape}")
print(f"Class balance:\n{y.value_counts(normalize=True)}") # Use normalize for large datasets
del df # Free up memory from the original loaded dataframe
print("Original dataframe memory released.")

print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

# --- Feature standardization ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.astype("float32")) # Ensure float32 for TF
print("Features standardized")
del X_df # Free up memory

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=42, stratify=y # Added stratify for potentially imbalanced data
)
print("Train/Test split completed.")
del X_scaled, y # Free up memory

# --- Build optimized tf.data pipeline with parallelism ---
# Increased batch size for faster training
BATCH_SIZE = 4096 # Keep or adjust based on GPU memory
AUTOTUNE = tf.data.AUTOTUNE

# Function for parallel mapping (can remain simple)
def preprocess_features(x, y):
    return x, y

# Optimized training pipeline
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=len(X_train))
    .cache()
    .batch(BATCH_SIZE)
    .map(preprocess_features, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# Optimized validation pipeline
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .cache()
    .batch(BATCH_SIZE)
    .map(preprocess_features, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
print("tf.data pipelines built.")

# --- Learning rate scheduler ---
initial_learning_rate = 0.002
EPOCHS = 30
def get_lr_schedule(): # Keep your custom schedule
    steps_per_epoch = len(X_train) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    def lr_schedule(epoch):
        step = epoch * steps_per_epoch
        if step < warmup_steps:
            return initial_learning_rate + (step / warmup_steps) * (0.02 - initial_learning_rate)
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_steps))
        return 0.02 * cosine_decay + 0.0001 * (1 - cosine_decay)
    return lr_schedule

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(get_lr_schedule())
class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LRLogger, self).__init__()
        self.lrs = []
    def on_epoch_begin(self, epoch, logs=None):
        # Ensure model and optimizer are available
        if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'lr'):
           lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
           self.lrs.append(lr)
           # Only print if verbose
           # print(f"\nEpoch {epoch + 1}: Learning rate: {lr:.6f}")
        else:
           # Handle case where optimizer might not be set yet (e.g., before compilation)
           pass # Or append a placeholder like None or np.nan
lr_logger = LRLogger()


# --- Model definition ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(AVAILABLE_FEATURES),)), # Use updated feature count
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate) # LR is controlled by scheduler
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
model.summary()


# --- Callbacks ---
log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True
)
checkpoint_path = f"checkpoints/model_tf32_batch{BATCH_SIZE}_" + "{epoch:02d}_{val_accuracy:.4f}.keras" # Use .keras format
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', save_format="keras" # Save in new format
)
class MemoryTracker(tf.keras.callbacks.Callback): # Keep your tracker
     def on_epoch_end(self, epoch, logs=None):
         try:
             if gpus:
                 import subprocess
                 result = subprocess.check_output(
                     ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
                 memory_used = int(result.decode('utf-8').strip())
                 print(f"\nGPU Memory used: {memory_used} MB")
         except Exception as e:
             # print(f"Could not get GPU memory: {e}") # Optional: print error if needed
             pass
memory_tracker = MemoryTracker()

callbacks = [
    tensorboard_callback,
    early_stopping,
    model_checkpoint,
    lr_scheduler,
    lr_logger,
    memory_tracker
]

# --- Train ---
print(f"\nStarting training with TF32 precision and batch size {BATCH_SIZE}...")
start_train_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - start_train_time
print(f"\nTraining completed in {training_time:.2f} seconds")
if history.epoch: # Avoid division by zero if training stops early
    print(f"Average time per epoch: {training_time / len(history.epoch):.2f} seconds")


# --- Evaluate & save ---
print("\nEvaluating on test set:")
loss, acc, auc, precision, recall = model.evaluate(val_ds, verbose=0) # Use val_ds (test set)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

model_save_path = f"semiconductor_classifier_tf32_batch{BATCH_SIZE}.keras" # Use .keras format
model.save(model_save_path)
print(f"Saved model to {model_save_path}")

scaler_save_path = "feature_scaler_tf32.pkl"
joblib.dump(scaler, scaler_save_path)
print(f"Saved feature scaler to {scaler_save_path}")

# --- Plotting ---
def plot_history(history, batch_size, lr_log): # Pass lr_log
    try:
        plt.figure(figsize=(15, 10))
        # Accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy (TF32, Batch Size: {batch_size})')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        # Loss
        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss (TF32, Batch Size: {batch_size})')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        # AUC
        plt.subplot(2, 2, 3)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title(f'Model AUC (TF32, Batch Size: {batch_size})')
        plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
        # Precision/Recall
        plt.subplot(2, 2, 4)
        plt.plot(history.history['precision'], label='Train Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.plot(history.history['recall'], label='Train Recall')
        plt.plot(history.history['val_recall'], label='Validation Recall')
        plt.title(f'Precision & Recall (TF32, Batch Size: {batch_size})')
        plt.xlabel('Epoch'); plt.ylabel('Value'); plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'training_history_tf32_batch{batch_size}.png')
        plt.close()

        # Plot learning rate if data exists
        if lr_log and lr_log.lrs:
             plt.figure(figsize=(10, 4))
             plt.plot(range(1, len(lr_log.lrs) + 1), lr_log.lrs) # Use epoch number (1-based)
             plt.title(f'Learning Rate Schedule (TF32, Batch Size: {batch_size})')
             plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.grid(True)
             plt.savefig(f'learning_rate_schedule_tf32_batch{batch_size}.png')
             plt.close()

        print(f"Training plots saved as training_history_tf32_batch{batch_size}.png and learning_rate_schedule_tf32_batch{batch_size}.png")
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

plot_history(history, BATCH_SIZE, lr_logger)

loss, acc, auc, precision, recall = model.evaluate(val_ds, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

# --- Get raw predictions and true labels ---
y_pred_prob = model.predict(val_ds)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

# Pull out the true labels from the dataset
y_true = np.concatenate([y for _, y in val_ds], axis=0).astype("int32")

# --- F1 Score ---
f1 = f1_score(y_true, y_pred)
print(f"Test F1-Score: {f1:.4f}")

# --- Matthews Correlation Coefficient ---
mcc = matthews_corrcoef(y_true, y_pred)
print(f"Test MCC: {mcc:.4f}")

# --- Confusion Matrix & Plot ---
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Not Semiconductor", "Semiconductor"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (TF32, Batch Size: {BATCH_SIZE})")
plt.savefig(f'confusion_matrix_tf32_batch{BATCH_SIZE}.png')
plt.close()
print(f"Confusion matrix saved as confusion_matrix_tf32_batch{BATCH_SIZE}.png")
