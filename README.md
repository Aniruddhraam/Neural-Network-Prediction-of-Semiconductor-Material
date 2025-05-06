# 🧠 Neural-Network-Prediction-of-Semiconductor-Material (With Frontend using Python)

This project builds and evaluates a deep learning model to classify materials as **semiconductors** or **non-semiconductors** based on physical and electronic properties. The model is implemented in TensorFlow and optimized for TF32 execution on modern NVIDIA GPUs, making it fast and scalable.

---

## 🚀 Features

- Deep Neural Network (DNN) model with `tf.keras`
- GPU acceleration with TensorFloat-32 (TF32) precision
- Custom learning rate scheduler with warmup and cosine decay
- Class balancing, missing value imputation, and feature scaling
- Early stopping, model checkpointing, and TensorBoard logging
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1-Score, MCC
- Visualization of training curves and confusion matrix

---

## 📁 Project Structure

```
.
├── semiconductor_data.csv               # Input dataset
├── semiconductor_classifier_tf32_batch4096.keras  # Saved trained model
├── feature_scaler_tf32.pkl              # Saved feature scaler
├── training_history_tf32_batch4096.png  # Training curves
├── learning_rate_schedule_tf32_batch4096.png # LR schedule
├── confusion_matrix_tf32_batch4096.png  # Confusion matrix
├── checkpoints/                         # Best model checkpoints
├── logs/                                # TensorBoard logs
└── main.py                              # Full training and evaluation script
```

---

## 📊 Dataset

The dataset should include the following **features** and **labels**:

### Input Features:
- Density (g/cm³)
- Formation Energy per Atom (eV)
- Fermi Energy (eV)
- Is Metal
- Direct Band Gap
- Conduction Band Minimum (eV)
- Valence Band Maximum (eV)

### Labels (used for filtering/validation only):
- Band Gap (eV)
- Energy Above Hull (eV)
- Fermi Energy (eV)
- Is Metal

The dataset is filtered to identify likely semiconductors based on physical rules:
- Band Gap between 0.1 and 4.0 eV
- Fermi Energy between -5 and 5 eV
- Stable compounds (Energy Above Hull ≤ 0.1)
- Non-metals

---

## 🧪 How to Run

1. Place `semiconductor_data.csv` in the root directory.
2. Run the main training and evaluation script:

```bash
python main.py
```

3. Results and plots will be saved automatically in the working directory.

---

## 📈 Output and Metrics

After training, the model will output:

- **Test Accuracy, AUC, Precision, Recall**
- **F1 Score**: balances precision and recall
- **Matthews Correlation Coefficient (MCC)**: robust metric for imbalanced data
- **Confusion Matrix**: saved as PNG
- **Training Curves**: accuracy, loss, AUC, precision & recall
- **Learning Rate Schedule**: visualized over epochs

---

## 📦 Model and Scaler Saving

- Trained model: `semiconductor_classifier_tf32_batch4096.keras`
- Feature scaler (for inference): `feature_scaler_tf32.pkl`

These can be reloaded for inference or fine-tuning on new data.

---

## 📌 Notes

- TF32 is enabled for improved GPU performance on NVIDIA Ampere+ hardware.
- Memory growth is enabled to handle large batches efficiently.
- Stratified train-test split ensures class distribution consistency.

---

## 📜 License

This project is provided under the MIT License.

---

## ✍️ Acknowledgment

This project was developed to explore deep learning applications in **materials informatics**, with an emphasis on classifying semiconducting behavior from computed descriptors.
