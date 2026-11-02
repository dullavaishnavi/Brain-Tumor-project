# 🧠 Brain Tumor MRI Image Classification

## Overview
This project provides an end-to-end Deep Learning pipeline for classifying brain MRI images into multiple categories according to tumor type. It covers building a custom Convolutional Neural Network (CNN) from scratch, leveraging state-of-the-art Transfer Learning models (ResNet50, MobileNetV2), comprehensive model evaluation, and deployment via an interactive Streamlit web application.

---

## 🚀 Key Features & Workflow
1. **Data Preprocessing & Augmentation**: Normalization to [0, 1], resizing to 224x224, and data augmentation (rotation, zoom, flips, shifts) for robust generalization.
2. **Custom CNN Architecture**: Designed from scratch with convolutional, pooling, batch normalization, and dropout layers.
3. **Transfer Learning**: Fine-tuned pretrained architectures (`ResNet50`, `MobileNetV2`) with ImageNet weights.
4. **Comprehensive Evaluation**: Metrics including Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Training History Plots.
5. **Streamlit Web Application**: Real-time tumor prediction interface with confidence scores and interactive visualizations.

---

## 🛠️ Tech Stack
- **Language**: Python 3.8+
- **Deep Learning**: TensorFlow / Keras
- **Web App**: Streamlit
- **Data Analysis & Visualization**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn

---

## 📦 Project Structure
```
brain_tumor_project/
│
├── app.py                      # Streamlit Web Application
├── train.py                    # Model Training Script (Custom CNN & Transfer Learning)
├── requirements.txt            # Python Dependencies
├── brain_tumor_classification.ipynb # Google Colab Notebook
└── README.md                   # Project Documentation
```

---

## 🚦 Getting Started & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## 📌 Business Use Cases
- **AI-Assisted Medical Diagnosis**: Reduce radiologist turnaround time and improve diagnostic accuracy.
- **Early Detection & Triage**: Automatically flag high-risk scans for immediate specialist review.
- **Research & Clinical Trials**: Segment patient datasets by tumor type for clinical studies.
- **Second-Opinion Systems**: Assist remote or under-resourced healthcare facilities with AI-driven preliminary diagnostics.
