# 🧠 Brain Tumor Detection using CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat&logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-~90%25-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

A deep learning project that detects brain tumors from MRI scan images using a Convolutional Neural Network (CNN). The model classifies MRI images to identify the presence and type of brain tumor with approximately **90% accuracy**.

---

## 📌 Features

- MRI image classification using CNN
- Trained on labeled brain MRI scan dataset
- Confusion matrix and accuracy visualization
- Streamlit web app for real-time predictions
- Supports multiple tumor categories

---

## 🗂️ Project Structure

```
Brain_Tumor/
├── MRI Scans/                  # Dataset directory
├── .ipynb_checkpoints/         # Jupyter checkpoints
├── my_model_categorical/       # Saved model directory
├── mainTrain.py                # Model training script
├── accuracy.py                 # Accuracy evaluation script
├── confusion matrix.py         # Confusion matrix visualization
├── streamlit(app).py           # Streamlit web app
├── demo.py                     # Demo/inference script
├── learning.py                 # Learning curve plots
├── brain_tumor_model.h5        # Trained model (Keras H5)
├── BrainTumor10Epochs.h5       # Model trained for 10 epochs
├── my_model.keras              # Saved Keras model
├── main.test.py                # Test script
├── index.html                  # Frontend HTML
└── README.md
```

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Notebook | Jupyter |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/abhik3373/Brain_Tumor.git
cd Brain_Tumor
```

### 2. Install dependencies
```bash
pip install tensorflow opencv-python matplotlib seaborn streamlit numpy scikit-learn
```

### 3. Train the model
```bash
python mainTrain.py
```

### 4. Evaluate accuracy
```bash
python accuracy.py
```

### 5. Run the Streamlit app
```bash
streamlit run "streamlit(app).py"
```

---

## 📊 Results

- **Model Accuracy:** ~90%
- **Training Epochs:** 10
- Confusion matrix and learning curves are generated via `confusion matrix.py` and `learning.py`

---

## 📁 Dataset & Models Download

Large files (MRI Scans dataset + trained model files) are not included in this repository due to size constraints.

**👉 [Download Full Project Assets (Google Drive)](https://drive.google.com/file/d/1EZ_2TxGCrGO4bISD1M983vR_wro3XuQs/view?usp=sharing)**

The zip contains:
- `MRI Scans/` — labeled MRI scan images for training & testing
- `brain_tumor_model.h5` — trained CNN model
- `my_model.keras` — saved Keras model
- `my_model_categorical/` — categorical model directory

After downloading, extract and place the contents in the root of the cloned repository before running any scripts.

---

## 👤 Author

**Abhishek Kolpe**
- GitHub: [@abhik3373](https://github.com/abhik3373)
- LinkedIn: [linkedin.com/in/abhishek-kolpe](https://linkedin.com/in/abhishek-kolpe)
- Email: abhishekkolpe2004@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
