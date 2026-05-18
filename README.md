# 🧠 Alzheimer's Disease Detection & Classification using Deep Learning

[![IEEE Published](https://img.shields.io/badge/IEEE-Published-blue?logo=ieee)](https://doi.org/10.1109/ICCAMS65118.2025.11233895)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLO-v5%20%7C%20v8%20%7C%20v9-green)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Published in IEEE ICCAMS 2025** — DOI: [10.1109/ICCAMS65118.2025.11233895](https://doi.org/10.1109/ICCAMS65118.2025.11233895)

---

## 📌 Overview

This project presents a dual-module deep learning system for **Alzheimer's Disease (AD) detection and classification** from MRI brain scans. It combines:

- **Classification Module** — Identifies the stage of Alzheimer's (Mild, Moderate, Non-Demented, Very Mild) using an ensemble of CNN-based transfer learning models.
- **Detection Module** — Localizes affected regions in MRI scans using YOLO object detection models (v5, v8, v9).
- **Web Application** — A Flask-based interface allowing clinicians and researchers to upload MRI images and receive real-time predictions.

The system was trained and evaluated on a dataset of **6,400 MRI images across 4 classes**, achieving a peak classification accuracy of **99.2%** with the ensemble model and a detection mAP of **92.6%** with YOLOv8.

---

## 📄 Publication

This work has been published in **IEEE**. If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{rakeshreddy2025alzheimer,
  title     = {An Innovative Deep Learning Framework for Multiclass Categorization and Alzheimer's Disease Level Detection},
  author    = {Rakesh Reddy and Co-authors},
  booktitle = {2025 International Conference on Computing, Analytics, Modeling and Simulation (ICCAMS)},
  year      = {2025},
  doi       = {10.1109/ICCAMS65118.2025.11233895},
  url       = {https://doi.org/10.1109/ICCAMS65118.2025.11233895},
  publisher = {IEEE}
}

## 🗂️ Project Structure

```
alzheimers-detection/
│
├── app.py                          # Flask web application
├── best.pt                         # Best YOLO detection model weights
├── extension.h5                    # Ensemble classification model weights
├── cnn1.h5                         # CNN (Adam) model
├── cnn2.h5                         # CNN (SGD) model
├── cnn3.h5                         # CNN (Nadam) model
├── nasnet.h5                       # NASNetMobile model
│
├── static/
│   └── uploads/                    # Uploaded MRI images
│
├── templates/
│   ├── home.html
│   ├── signin.html
│   ├── signup.html
│   ├── index.html                  # Classification interface
│   ├── index1.html                 # Detection interface
│   ├── result.html                 # Classification results
│   ├── Classification.html         # Classification notebook viewer
│   ├── Detection.html              # Detection notebook viewer
│   └── about.html
│
├── signup.db                       # SQLite user database
│
├── notebooks/
│   ├── Classification.ipynb        # Classification experiments
│   └── Detection.ipynb             # Detection experiments
│
└── README.md
```

---

## 🧪 Modules

### 1. Classification Module

Trained on **4,479 training** and **1,921 validation** MRI images across 4 Alzheimer's stages:

| Class | Description |
|---|---|
| Non-Demented | No signs of Alzheimer's |
| Very Mild Demented | Earliest stage |
| Mild Demented | Early stage with cognitive decline |
| Moderate Demented | Significant memory loss |

**17 models were benchmarked**, including transfer learning architectures and custom CNNs:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Extension (Ensemble)** | **0.992** | **0.993** | **0.990** | **0.991** |
| Xception | 0.992 | 0.992 | 0.992 | 0.992 |
| CNN – SGD | 0.984 | 0.984 | 0.984 | 0.984 |
| CNN – Nadam | 0.969 | 0.969 | 0.969 | 0.969 |
| CNN – Adam | 0.956 | 0.956 | 0.956 | 0.956 |
| NASNetMobile | 0.933 | 0.934 | 0.932 | 0.933 |
| MobileNetV2 | 0.753 | 0.770 | 0.736 | 0.747 |
| DenseNet169 | 0.623 | 0.626 | 0.467 | 0.520 |
| ResNet152 | 0.585 | 0.595 | 0.473 | 0.514 |
| InceptionResNetV2 | 0.554 | 0.530 | 0.392 | 0.438 |
| DenseNet121 | 0.549 | 0.551 | 0.476 | 0.501 |
| InceptionV3 | 0.527 | 0.538 | 0.432 | 0.467 |
| ResNet50 | 0.505 | 0.444 | 0.300 | 0.348 |
| VGG19 | 0.500 | 0.500 | 0.500 | 0.500 |

The **Extension model** is a simple-average ensemble of CNN-Adam, CNN-SGD, CNN-Nadam, and NASNetMobile, trained end-to-end with SGD.

---

### 2. Detection Module

YOLO-based object detection for localizing Alzheimer's-affected regions in MRI scans. Trained for 50 epochs at 416×416 resolution on an NVIDIA RTX 4090.

| Model | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| **YOLOv8n** | **0.893** | **0.895** | **0.926** |
| YOLOv5s6u | 0.882 | 0.829 | 0.924 |
| YOLOv5x6u | 0.856 | 0.819 | 0.918 |
| YOLOv9c | 0.812 | 0.863 | 0.900 |

**YOLOv8n** was selected as the production detection model (`best.pt`) due to its best balance of precision, recall, and mAP.

---

### 3. Web Application

A Flask web app (`app.py`) with:

- **User authentication** — OTP-based email verification on signup, SQLite-backed login
- **Classification** — Upload an MRI image → get Alzheimer's stage prediction
- **Detection** — Upload an MRI image → get bounding box detection output
- **Notebook viewer** — Embedded HTML views of classification and detection experiments

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/alzheimers-detection.git
cd alzheimers-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Core dependencies</summary>

```
flask
tensorflow
torch
torchvision
ultralytics
opencv-python
Pillow
pandas
numpy
scikit-learn
efficientnet
sqlite3
```

</details>

### 4. Download model weights

Place the following model files in the project root:

| File | Description |
|---|---|
| `best.pt` | YOLOv8 detection model |
| `extension.h5` | Ensemble classification model |
| `cnn1.h5` | CNN-Adam model |
| `cnn2.h5` | CNN-SGD model |
| `cnn3.h5` | CNN-Nadam model |
| `nasnet.h5` | NASNetMobile model |

> Model weights can be downloaded from [Releases](https://github.com/yourusername/alzheimers-detection/releases) or trained from scratch using the notebooks.

### 5. Set up the database

```bash
python -c "import sqlite3; con=sqlite3.connect('signup.db'); con.execute('CREATE TABLE IF NOT EXISTS info (user TEXT, email TEXT, password TEXT, mobile TEXT, name TEXT)'); con.commit()"
```

### 6. Configure email (for OTP signup)

In `app.py`, update the sender credentials:

```python
msg['From'] = "your-email@gmail.com"
s.login("your-email@gmail.com", "your-app-password")
```

> Use a [Gmail App Password](https://support.google.com/accounts/answer/185833) — do not use your main password.

### 7. Run the app

```bash
python app.py
```

Open your browser at `http://localhost:5000`

---

## 🖥️ Usage

### Classification

1. Navigate to the **Classification** page
2. Upload an MRI brain scan (`.jpg`, `.jpeg`, `.png`)
3. The model returns one of: **Mild Demented**, **Moderate Demented**, **Non-Demented**, or **Very Mild Demented**

### Detection

1. Navigate to the **Detection** page
2. Upload an MRI brain scan (`.jpg`)
3. YOLOv8 returns the image with bounding boxes drawn around detected regions

---

## 📊 Dataset

- **Source:** [Kaggle — Alzheimer's Dataset (4 class of Images)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
- **Total images:** 6,400
- **Training set:** 4,479 images
- **Validation set:** 1,921 images
- **Classes:** Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented
- **Image size:** 128×128 (classification), 416×416 (detection)

---

## 🏗️ Model Architecture

### Ensemble (Extension) Model

```
Input (128×128×3)
    │
    ├── CNN-Adam ──┐
    ├── CNN-SGD ───┤
    ├── CNN-Nadam ─┤──► Average Layer ──► Output (4 classes, softmax)
    └── NASNet ────┘
```

Each CNN branch:
```
Conv2D(32, 5×5, same) → MaxPool → Conv2D(48, 5×5) → MaxPool → Flatten → Dense(256) → Dense(84) → Dense(4, softmax)
```

### YOLO Detection Pipeline

```
MRI Image (416×416) → YOLOv8n Backbone → Neck (PAN+FPN) → Detection Head → Bounding Boxes + Class Labels
```

---

## 📈 Results

**Classification — Ensemble model (Extension):**

| Metric | Score |
|---|---|
| Accuracy | 99.2% |
| Precision | 99.3% |
| Recall | 99.0% |
| F1-Score | 99.1% |

**Detection — YOLOv8n:**

| Metric | Score |
|---|---|
| Precision | 89.3% |
| Recall | 89.5% |
| mAP@0.5 | 92.6% |

---

## 🔒 Security Notes

- Passwords are stored in plain text in the current implementation. For production, use `bcrypt` or `werkzeug.security` for password hashing.
- The SMTP credentials in `app.py` should be stored in environment variables, not hardcoded.
- SQL queries use parameterized statements to prevent injection.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Rakesh Reddy**  
M.Tech in Data Science, Presidency University, Bengaluru  
📧 rakeshreddy.da@gmail.com  
🔗 [GitHub](https://github.com/rakeshreddyda-ops) · [LinkedIn](https://linkedin.com/in/rakeshreddy-da)

---

## 🙏 Acknowledgements

- [Ultralytics](https://ultralytics.com) for the YOLO framework
- [TensorFlow / Keras](https://tensorflow.org) for the deep learning backbone
- [Kaggle](https://kaggle.com) for the Alzheimer's MRI dataset
- IEEE for publishing this research

---

*⭐ If this project helped you, please consider starring the repository!*
