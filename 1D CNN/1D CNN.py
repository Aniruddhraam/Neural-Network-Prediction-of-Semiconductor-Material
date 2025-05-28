import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import time
import joblib  # <-- Added for PKL export

# --- Timer Utility ------------------------------------------------------------
class Timer:
    def __init__(self, name="Operation"):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        print(f"{self.name} completed in {time.time() - self.start:.2f} seconds")

# --- Custom Callback ---------------------------------------------------------
class ValAUC(Callback):
    """
    At end of each epoch, computes ROC‑AUC for semiconductor (class=1)
    on the held‑out validation data.
    """
    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        sem_probs = y_pred[:, 1]   # index 1 = Semiconductor
        sem_auc = roc_auc_score((self.y_val == 1).astype(int), sem_probs)
        logs = logs or {}
        logs['val_sem_auc'] = sem_auc
        self.history.append(sem_auc)
        print(f" — val_sem_auc: {sem_auc:.4f}")

# --- Data Loading & Preprocessing --------------------------------------------
def load_and_preprocess_data(path):
    with Timer("Data load & prep"):
        df = pd.read_csv(path)
        features = ['Band Gap (eV)', 'Density (g/cm³)',
                    'Formation Energy per Atom (eV)', 'Fermi Energy (eV)']
        df = df.dropna(subset=['Is Metal'] + features)

        def mat_class(r):
            if r['Is Metal']:
                return 0
            return 1 if r['Band Gap (eV)'] < 3.0 else 2

        df['Material_Class'] = df.apply(mat_class, axis=1)

        X = df[features].values
        y = df['Material_Class'].values

        # Split FIRST before any preprocessing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Scale using training data statistics only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reshape for Conv1D after splitting and scaling
        X_train_ready = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
        X_test_ready = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

        return X_train_ready, X_test_ready, y_train, y_test, scaler

# --- Model Definition ---------------------------------------------------------
def build_model(input_shape):
    model = Sequential([
        Conv1D(16, 2, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Plotting Helper ----------------------------------------------------------
def save_and_show(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved {filename}")

# --- Main Pipeline ------------------------------------------------------------
def main(data_path='semiconductor_data_with_available_features.csv'):
    # Load & preprocess with proper splitting
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)

    # Split validation set from training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Build & compile
    model = build_model(input_shape=X_train.shape[1:])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    val_auc_cb = ValAUC(validation_data=(X_val, y_val))

    # Train
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop, val_auc_cb],
        verbose=1
    )

    # 1) Plot loss & accuracy
    for m in ['loss', 'accuracy']:
        fig = plt.figure()
        plt.plot(history.history[m],      label=f"train_{m}")
        plt.plot(history.history[f"val_{m}"], label=f"val_{m}")
        plt.title(m.capitalize())
        plt.xlabel('Epoch'); plt.ylabel(m)
        plt.legend()
        save_and_show(fig, f"{m}_curve.png")

    # 2) Plot validation semiconductor AUC
    fig = plt.figure()
    plt.plot(val_auc_cb.history, label='val_sem_auc')
    plt.title("Validation Semiconductor AUC")
    plt.xlabel("Epoch"); plt.ylabel("AUC")
    plt.legend()
    save_and_show(fig, "val_sem_auc_curve.png")

    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Detailed classification report
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Metal','Semiconductor','Insulator'], digits=3
    ))

    # ROC curve for semiconductor vs rest
    fpr, tpr, _ = roc_curve((y_test==1).astype(int), y_prob[:,1])
    sem_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"Semiconductor AUC = {sem_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC: Semiconductor vs Others")
    plt.legend()
    save_and_show(fig, "roc_semiconductor.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.xticks([0,1,2], ['Metal','Semi','Insulator'])
    plt.yticks([0,1,2], ['Metal','Semi','Insulator'])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i,j], ha='center', va='center')
    plt.title("Confusion Matrix")
    save_and_show(fig, "confusion_matrix_3way.png")

    # Save artifacts
    model.save('material_classifier_3way.h5')
    joblib.dump(scaler, 'scaler.pkl')  # <-- Added PKL export
    print("\nSaved artifacts:")
    print("- material_classifier_3way.h5 (Keras model)")
    print("- scaler.pkl (StandardScaler for preprocessing)")

    # --- Feature Importance via Permutation ----------------------------------
    # flatten X_test back to (n_samples, n_features)
    X_test_2d = X_test.reshape(X_test.shape[0], X_test.shape[1])

    def permutation_importance(model, X, y, n_repeats=5):
        """Return mean decrease in accuracy when permuting each feature."""
        y_base = np.argmax(model.predict(X, verbose=0), axis=1)
        base_acc = accuracy_score(y, y_base)
        importances = []
        X_perm = X.copy()
        for feat in range(X.shape[1]):
            drops = []
            for _ in range(n_repeats):
                col = X_perm[:, feat, 0]
                np.random.shuffle(col)
                X_perm[:, feat, 0] = col
                y_pred = np.argmax(model.predict(X_perm, verbose=0), axis=1)
                drops.append(base_acc - accuracy_score(y, y_pred))
                X_perm[:, feat, 0] = X[:, feat, 0]
            importances.append(np.mean(drops))
        return importances

    feat_names = ['Band Gap (eV)', 'Density (g/cm³)',
                  'Formation Energy per Atom (eV)', 'Fermi Energy (eV)']
    importances = permutation_importance(model, X_test, y_test)

    # plot feature importances
    fig = plt.figure()
    plt.bar(feat_names, importances)
    plt.ylabel("Mean ↓ in Test Accuracy")
    plt.title("Permutation Feature Importance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_and_show(fig, "feature_importance.png")

if __name__ == '__main__':
    main()