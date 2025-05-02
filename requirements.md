# Hardware and Software Setup Guide

## ===========================
## Hardware Recommendations
## ===========================
- **GPU:** NVIDIA RTX 3000 series or newer (e.g., RTX 3060, 3080, 4090)  
  - *Minimum CUDA Compute Capability:* 8.6  
- **CPU:** Modern multi-core processor (e.g., AMD Ryzen 5+, Intel i5 10th Gen+)  
- **RAM:** Minimum 16 GB  
- **Storage:** SSD with at least 50 GB free space  
- **OS:** Windows 10/11 64-bit, Ubuntu 20.04+, or macOS (limited GPU support)  

---

## ===========================
## Python Environment Setup
## ===========================
- **Recommended Python Version:** 3.8 – 3.10  
- **Create virtual environment:**  
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```

---

## ===========================
## Core Python Packages
## ===========================
```plaintext
tensorflow==2.13.0  
tensorflow-addons==0.22.0  
keras==2.13.1  
numpy==1.24.3  
pandas==1.5.3  
matplotlib==3.7.1  
scikit-learn==1.2.2  
scipy==1.10.1  
```

---

## ===========================
## Jupyter & Development Tools
## ===========================
```plaintext
jupyterlab==3.6.3  
tensorboard==2.13.0  
ipykernel==6.22.0  
notebook==6.5.4  
tqdm==4.65.0  
```

---

## ===========================
## Visualization (Optional)
## ===========================
```plaintext
seaborn==0.12.2  
plotly==5.14.1  
```

---

## ===========================
## GPU Monitoring Tools (Optional)
## ===========================
- **Linux/Windows only**
```plaintext
nvidia-ml-py3==7.352.0
```

---

## ===========================
## System Requirements
## ===========================
- **CUDA Toolkit:** 11.8  
- **cuDNN:** v8.6.0 (for CUDA 11.8)  
- **Drivers:** NVIDIA 515+ for CUDA 11.8 support  

---

> ⚠️ **Note:**  
Ensure your GPU drivers and CUDA/cuDNN versions are correctly installed.  
Refer to TensorFlow's compatibility guide:  
[https://www.tensorflow.org/install/source#gpu](https://www.tensorflow.org/install/source#gpu")
