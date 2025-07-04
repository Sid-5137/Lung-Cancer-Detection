# Lung Cancer Detection using Deep Learning

A deep learning-based image classification model designed to detect lung cancer from medical images. This project leverages a dual CNN architecture combining **ResNet50** and **MobileNet**, trained on a 3-class dataset to classify lung conditions as **Benign**, **Malignant**, or **Normal**.

---

## 📌 Project Overview

- **Goal**: Accurately classify lung cancer images into three categories using a robust deep learning pipeline.
- **Model**: Dual-branch architecture using ResNet50 and MobileNet as feature extractors, combined via concatenation and fully connected layers.
- **Training Epochs**: 20
- **Final Validation Accuracy**: **93.50%**
- **Test Accuracy**: **93.66%**

---

## 🧠 Model Architecture

- **Base Models**: ResNet50 + MobileNet (pretrained on ImageNet, frozen)
- **Fusion Layer**: GlobalAveragePooling2D + Concatenate
- **Classification Head**:
  - Dense(128, ReLU)
  - Dense(64, ReLU)
  - Dense(3, Softmax)

---

## 📁 Dataset Structure

Images are structured into the following directories:

```

data/
├── train/
│   ├── Benign/
│   ├── Malignant/
│   └── Normal/
├── val/
│   ├── Benign/
│   ├── Malignant/
│   └── Normal/
└── test/
├── Benign/
├── Malignant/
└── Normal/

````

> 📌 **Note**: Download Dataset from: [https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset]
---

## 🔧 Installation & Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
````

---

## 🚀 How to Run

1. Ensure the dataset is placed in the `data/` directory with the correct structure (as shown above).
2. Run the training script:

```bash
python train_model.py
```

This project is for academic and research purposes.
