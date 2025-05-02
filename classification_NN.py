import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef

print("TensorFlow version:", tf.__version__)

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

start_time = time.time()

FEATURES = [
    "Density (g/cm³)",
    "Formation Energy per Atom (eV)",
    "Fermi Energy (eV)",
    "Is Metal",
    "Direct Band Gap",
    "Conduction Band Minimum (eV)",
    "Valence Band Maximum (eV)"
]
LABEL_COLS = [
    "Band Gap (eV)",
    "Energy Above Hull (eV)",
    "Fermi Energy (eV)",
    "Is Metal"
]
cols_to_load = list(set(FEATURES + LABEL_COLS))
print(f"Columns to load: {cols_to_load}")

print("Using 'pyarrow' engine for pd.read_csv")
df = pd.read_csv(
    "semiconductor_data.csv",
    usecols=cols_to_load,
    engine='pyarrow'
)
print(f"Initial shape after loading specific columns: {df.shape}")

condition_is_metal = (df["Is Metal"] == 1)
condition_band_gap_low = (df["Band Gap (eV)"] <= 0.1)
condition_band_gap_high = (df["Band Gap (eV)"] > 4.0)
condition_above_hull = (df["Energy Above Hull (eV)"] > 0.1)
condition_fermi_low = (df["Fermi Energy (eV)"] < -5)
condition_fermi_high = (df["Fermi Energy (eV)"] > 5)

is_not_semiconductor = (
    condition_is_metal |
    condition_band_gap_low |
    condition_band_gap_high |
    condition_above_hull |
    condition_fermi_low |
    condition_fermi_high
)

y = (~is_not_semiconductor).astype("float32")
print("Vectorized labeling completed.")

AVAILABLE_FEATURES = [c for c in FEATURES if c in df.columns]
print(f"Using features: {AVAILABLE_FEATURES}")
X_df = df[AVAILABLE_FEATURES].copy()

cols_to_drop = [col for col in AVAILABLE_FEATURES if X_df[col].isna().all()]
if cols_to_drop:
    print(f"Columns dropped due to 100% NaN: {cols_to_drop}")
    X_df = X_df.drop(columns=cols_to_drop)
    AVAILABLE_FEATURES = [c for c in AVAILABLE_FEATURES if c not in cols_to_drop]

numeric_cols = X_df.select_dtypes(include="number").columns
cat_cols = X_df.select_dtypes(exclude="number").columns

fill_values = {}
if not numeric_cols.empty:
    medians = X_df[numeric_cols].median()
    fill_values.update(medians.to_dict())
if not cat_cols.empty:
    modes = X_df[cat_cols].mode().iloc[0]
    fill_values.update(modes.to_dict())

X_df.fillna(fill_values, inplace=True)
print("NaN imputation completed.")

for col in cat_cols:
    if col in X_df.columns:
        X_df[col] = X_df[col].astype(int)

print(f"Shape after preprocessing X: {X_df.shape}")
print(f"Class balance:\n{y.value_counts(normalize=True)}")
del df
print("Original dataframe memory released.")

print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.astype("float32"))
print("Features standardized")
del X_df

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
)
print("Train/Test split completed.")
del X_scaled, y

BATCH_SIZE = 4096
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_features(x, y):
    return x, y

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=len(X_train))
    .cache()
    .batch(BATCH_SIZE)
    .map(preprocess_features, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .cache()
    .batch(BATCH_SIZE)
    .map(preprocess_features, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
print("tf.data pipelines built.")

initial_learning_rate = 0.002
EPOCHS = 30

def get_lr_schedule():
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
        if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'lr'):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            self.lrs.append(lr)

lr_logger = LRLogger()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(AVAILABLE_FEATURES),)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
model.summary()

log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True
)
checkpoint_path = f"checkpoints/model_tf32_batch{BATCH_SIZE}_" + "{epoch:02d}_{val_accuracy:.4f}.keras"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', save_format="keras"
)

class MemoryTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            if gpus:
                import subprocess
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
                memory_used = int(result.decode('utf-8').strip())
                print(f"\nGPU Memory used: {memory_used} MB")
        except:
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
if history.epoch:
    print(f"Average time per epoch: {training_time / len(history.epoch):.2f} seconds")

print("\nEvaluating on test set:")
loss, acc, auc, precision, recall = model.evaluate(val_ds, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

model_save_path = f"semiconductor_classifier_tf32_batch{BATCH_SIZE}.keras"
model.save(model_save_path)
print(f"Saved model to {model_save_path}")

scaler_save_path = "feature_scaler_tf32.pkl"
joblib.dump(scaler, scaler_save_path)
print(f"Saved feature scaler to {scaler_save_path}")

def plot_history(history, batch_size, lr_log):
    try:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy (TF32, Batch Size: {batch_size})')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss (TF32, Batch Size: {batch_size})')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title(f'Model AUC (TF32, Batch Size: {batch_size})')
        plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
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
        if lr_log and lr_log.lrs:
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, len(lr_log.lrs) + 1), lr_log.lrs)
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

y_pred_prob = model.predict(val_ds)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
y_true = np.concatenate([y for _, y in val_ds], axis=0).astype("int32")

f1 = f1_score(y_true, y_pred)
print(f"Test F1-Score: {f1:.4f}")

mcc = matthews_corrcoef(y_true, y_pred)
print(f"Test MCC: {mcc:.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Semiconductor", "Semiconductor"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (TF32, Batch Size: {BATCH_SIZE})")
plt.savefig(f'confusion_matrix_tf32_batch{BATCH_SIZE}.png')
plt.close()
print(f"Confusion matrix saved as confusion_matrix_tf32_batch{BATCH_SIZE}.png")
