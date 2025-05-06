import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import time

# Simple timer utility
class Timer:
    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print(f"{self.name} completed in {self.end - self.start:.2f} seconds")

# Configure GPU - avoid problematic XLA settings
print("Setting up GPU configuration for RTX 4070...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {gpus}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found. Using CPU.")

# Load and preprocess data
def load_and_preprocess_data(file_path):
    with Timer("Data loading and preprocessing"):
        df = pd.read_csv(file_path)
        print(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")

        df['Is_Semiconductor'] = (~df['Is Metal']).astype(int)
        features = [
            'Band Gap (eV)',
            'Density (g/cm³)',
            'Formation Energy per Atom (eV)',
            'Fermi Energy (eV)'
        ]
        df_clean = df.dropna(subset=['Is Metal'] + features)
        print(f"Clean samples: {df_clean.shape[0]}")
        print("Class distribution:")
        print(df_clean['Is_Semiconductor'].value_counts())

        X = df_clean[features].values
        y = df_clean['Is_Semiconductor'].values

        noise_idx = np.random.choice(len(y), size=int(len(y) * 0.05), replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        print(f"Added noise to {len(noise_idx)} samples")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        return X_reshaped, y, scaler

# Build model
def build_1d_cnn_model(input_shape):
    model = Sequential([
        Conv1D(8, 2, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# Helper to save and show plots
def save_and_show(fig_name):
    plt.savefig(fig_name, bbox_inches='tight')
    print(f"Saved {fig_name}")
    plt.show()

# Prediction helper
def predict_semiconductor(model, features, scaler):
    f = scaler.transform([features])
    f = f.reshape(1, f.shape[1], 1)
    pred = model.predict(f, verbose=0)[0][0]
    return bool(pred > 0.5), pred

# Main pipeline
def main(file_path='semiconductor_data_with_available_features.csv'):
    # Load data
    X_data, y_data, scaler = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.25, random_state=42
    )
    input_shape = (X_train.shape[1], 1)

    # Build and train model
    model = build_1d_cnn_model(input_shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Plot and save Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_and_show('loss_curve.png')

    # Plot and save Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    save_and_show('accuracy_curve.png')

    # Plot and save AUC
    plt.figure()
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    save_and_show('auc_curve.png')

    # Evaluate on test set
    loss, acc, test_auc = model.evaluate(X_test, y_test, batch_size=64)
    print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}, AUC: {test_auc:.4f}")
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    # Plot and save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    save_and_show('roc_curve.png')

    # Plot and save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Metal', 'Semiconductor'])
    plt.yticks(tick_marks, ['Metal', 'Semiconductor'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.tight_layout()
    save_and_show('confusion_matrix.png')

    # Save model
    model.save('semiconductor_1dcnn_classifier.h5')
    print("Model saved as 'semiconductor_1dcnn_classifier.h5'")

if __name__ == '__main__':
    main()