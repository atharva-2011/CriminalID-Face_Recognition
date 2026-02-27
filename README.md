# 🔍 CriminalID — AI Face Recognition System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-97.85%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**An AI-powered criminal face recognition web app built with Transfer Learning and deployed via Streamlit.**

[🚀 Live Demo](https://YOUR_APP_NAME.streamlit.app) • [📓 Training Notebook](Criminal_Face_Recognition_Using_Transfer_Learning.ipynb) • [📊 Dataset Notebook](Dataset_Creation.ipynb)

</div>

---

## 📸 Demo

> Upload a photo or use your webcam — the model identifies the individual and displays their criminal profile in real time.

| Upload Image | Webcam Capture | Criminal Database |
|:---:|:---:|:---:|
| Upload any photo | Live camera snap | Search & filter records |
| Face detected via OpenCV | Auto-runs on capture | Name, crime, status, age |

---

## 🧠 Model Performance

| Metric | Score |
|--------|-------|
| ✅ Overall Validation Accuracy | **97.85%** |
| 📊 Macro Avg Precision | **0.98** |
| 📊 Macro Avg Recall | **0.98** |
| 📊 Macro Avg F1-Score | **0.98** |
| 🖼️ Total Training Samples | **4,420** (13 classes × 340 each) |
| 🔁 Training Strategy | Frozen base → Fine-tuning (Phase 2 from epoch 13) |

### Per-Class Results

| Individual | Precision | Recall | F1-Score |
|---|---|---|---|
| Aniket Kakad | 1.00 | 1.00 | 1.00 |
| Atharva Ayachit | 0.96 | 0.97 | 0.97 |
| Chandrakant Kshirsagar | 1.00 | 0.90 | 0.95 |
| Dhananjay Patil | 1.00 | 1.00 | 1.00 |
| Ishika Mehre | 1.00 | 1.00 | 1.00 |
| Japneet Singh | 1.00 | 1.00 | 1.00 |
| Jay Kumar | 0.99 | 1.00 | 0.99 |
| Jyoti Ayachit | 1.00 | 1.00 | 1.00 |
| Kajal Yerone | 0.88 | 0.95 | 0.92 |
| Sakshi Lahekar | 0.95 | 0.92 | 0.93 |
| Sayali Chidrawar | 1.00 | 0.99 | 1.00 |
| Shreyas Jadhav | 1.00 | 0.99 | 0.99 |
| Tushar Patil | 0.97 | 1.00 | 0.98 |

---

## 🏗️ Project Structure

```
CriminalID-Face-Recognition/
├── CriminalID.py                                          ← Streamlit web app
├── criminal_recognition_model.keras                      ← Trained Transfer Learning model
├── criminals_info.csv                                     ← Criminal profile database
├── class_names.txt                                        ← 13 class names (alphabetical order)
├── requirements.txt                                       ← Python dependencies
├── Criminal_Face_Recognition_Using_Transfer_Learning.ipynb  ← Model training notebook
├── Dataset_Creation.ipynb                                 ← Dataset preparation notebook
└── criminal_dataset/                                      ← Training images (13 classes)
    ├── Aniket_Kakad/
    ├── Atharva_Ayachit/
    └── ...
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/CriminalID-Face-Recognition.git
cd CriminalID-Face-Recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install streamlit tensorflow opencv-python pillow pandas numpy
```

### 3. Run the App
```bash
streamlit run CriminalID.py
```

Then open **http://localhost:8501** in your browser.

---

## 🖥️ How to Use

### 📤 Upload Image Tab
1. Upload any photo (`.jpg` / `.png` / `.webp`)
2. Click **RUN IDENTIFICATION**
3. Face is detected → model classifies → result shown with criminal profile

### 📷 Webcam Tab
1. Click the camera widget → allow browser camera access
2. Take a photo → identification runs automatically

### 🗄️ Database Tab
- View all criminal records
- Search and filter by name or crime type

---

## ⚙️ Sidebar Settings

| Setting | Description |
|---|---|
| Model Path | Path to `criminal_recognition_model.keras` |
| Criminal Info CSV | Path to `criminals_info.csv` |
| Confidence Threshold | Min % confidence to make a positive ID (default **80%**) |
| Class Names File | Optional `.txt` with class names in alphabetical order |

---

## 📋 criminals_info.csv Format

```csv
name,crime,status,age,nationality,last_seen,description
Aniket_Kakad,Armed Robbery,Wanted,22,Indian,Pune 2024,Considered dangerous
Atharva_Ayachit,Fraud,Arrested,21,Indian,Mumbai 2024,Financial crimes
```

> ⚠️ `name` must **exactly** match the folder names in your dataset.
> 
> `status` accepted values: `Wanted` / `Arrested` / `Imprisoned`

---

## 🤖 Model Architecture

- **Base Model:** Transfer Learning (MobileNetV2 / EfficientNet)
- **Training Phase 1:** Base layers frozen → only top layers trained (epochs 1–12)
- **Training Phase 2:** Fine-tuning — top layers of base model unfrozen (epoch 13 onwards)
- **Input Size:** 224 × 224 × 3
- **Output:** Softmax over 13 classes
- **Optimizer:** Adam with learning rate scheduling
- **Data Augmentation:** Rotation, flip, zoom, brightness shifts

---

## 🚀 Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub (must be **public**)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `CriminalID.py`
4. Click **Deploy** — your app goes live in ~5 minutes

🔗 Live at: `https://YOUR_APP_NAME.streamlit.app`

---

## 👥 Recognized Individuals

The model is trained to identify **13 individuals**:

`Aniket_Kakad` · `Atharva_Ayachit` · `Chandrakant_Kshirsagar` · `Dhananjay_Patil` · `Ishika_Mehre` · `Japneet_Singh` · `Jay_Kumar` · `Jyoti_Ayachit` · `Kajal_Yerone` · `Sakshi_Lahekar` · `Sayali_Chidrawar` · `Shreyas_Jadhav` · `Tushar_Patil`

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.15 / Keras |
| Face Detection | OpenCV (Haar Cascades) |
| Web App | Streamlit |
| Data Handling | Pandas, NumPy |
| Image Processing | Pillow |
| Deployment | Streamlit Community Cloud |

---

## 📦 Requirements

```
streamlit==1.32.0
tensorflow==2.15.0
opencv-python-headless==4.9.0.80
pillow==10.2.0
pandas==2.2.0
numpy==1.26.4
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [TensorFlow / Keras](https://www.tensorflow.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the rapid web app deployment
- [OpenCV](https://opencv.org/) for face detection
- Transfer Learning base architectures: MobileNetV2 / EfficientNet

---

<div align="center">
  Made with ❤️ using TensorFlow · OpenCV · Streamlit
</div>
