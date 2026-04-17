# 🧠 SDT-OCR: Spiking Transformer for Text Transcription

## 🧩 Spike-Driven Transformer (SDT) Architecture

![SDT Architecture](sdt_architecture.png)

---

This project focuses on Optical Character Recognition (OCR) using a **Spike-Driven Transformer (SDT)** based on **Spiking Neural Networks (SNNs)**.

---

## 📌 Description

The goal of this project is to transcribe text from images using a biologically inspired neural network architecture.

The model is trained on a **synthetically generated dataset**, making it suitable for controlled OCR experiments.

---

## 🧠 Model Architecture

* Spike-Driven Transformer (SDT)
* Spiking Neural Networks (SNN)
* Sequence modeling for text transcription

---

## 📂 Project Structure

```
models/                # Model architecture (SpikeFormer, SDT)
module/                # Core modules and layers
dataset.py             # Dataset loading and preprocessing
tokenizer.py           # Text encoding / decoding
train.py               # Training script
utils.py               # Utility functions
outputs/               # Training results (models, losses, etc.)
Transcription_de_texte.ipynb  # Visualization (loss curves & test results)
```

---

## 📊 Notebook (Visualization)

The file **Transcription_de_texte.ipynb** is used to:

* Visualize training and validation loss curves
* Evaluate model predictions on test data
* Analyze model performance

---

## ▶️ How to Run

```
python train.py
```

---

## 📈 Results

The model learns to transcribe text from synthetic data using a spiking-based transformer architecture.

---

## 👤 Author

**Soidad Soule Ahamada**  
MSc Student in Image and Artificial Intelligence
