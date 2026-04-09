# 🦷 DeepDent: AI-Powered Dental Decay Detection

DeepDent is a Deep Learning application designed to assist in identifying dental caries (decay) from Orthopantomogram (OPG) X-rays. By leveraging Transfer Learning with the MobileNetV2 architecture, the model provides high-accuracy screening to aid dental professionals in diagnostics.

---

## 🚀 Execution Flow

The project follows a structured Machine Learning Lifecycle (MLLC):

1. **Data Engineering**: Automated streaming of OPG X-ray datasets from Kaggle, followed by a controlled 80/20 split for training and validation.
2. **Preprocessing**: Real-time Data Augmentation (rotation, zoom, flips) using `ImageDataGenerator` to improve model generalization.
3. **Model Development**: Implementation of **Transfer Learning**. We utilized a pre-trained **MobileNetV2** base for feature extraction and added a custom Classification Head (GlobalAveragePooling, Dense, and Dropout layers).
4. **Training**: Optimized using the **Adam** optimizer and **Binary Cross-Entropy** loss over 10 epochs.
5. **Deployment**: Integration of the saved `.h5` model into a responsive **Streamlit** web interface for real-time inference.

---

## 🛠️ Tech Stack

| Technology | Role |
| :--- | :--- |
| **Python 3.13** | Core Programming Language |
| **TensorFlow / Keras** | Model Building & Training |
| **MobileNetV2** | Pre-trained CNN Architecture |
| **Streamlit** | Web UI & Frontend Deployment |
| **Kaggle API** | Dataset Acquisition |
| **Matplotlib** | Training Visualization |

---

## 📂 File Structure

- `DeepDent_Train.ipynb`: Jupyter Notebook containing the end-to-end training pipeline.
- `app.py`: Streamlit application file for the web interface.
- `deepdent_model.h5`: The serialized weights of the trained model.
- `requirements.txt`: List of dependencies for environment reproduction.
- `CONTRIBUTOR_GUIDE.txt`: Step-by-step instructions for setup and testing.

---

## 📊 Data Profile

- **Source**: [Dental OPG X-Ray Dataset](https://www.kaggle.com/datasets/arfintanim/dental-opg-xray)
- **Type**: Panoramic X-rays (OPGs)
- **Classes**: 
  - **Healthy**: Structurally sound teeth.
  - **Decay**: Visible dental caries/erosion.

---

## 💻 Installation & Usage

### 1. Setup Environment
```bash
git clone [https://github.com/RRonium/DeepDent.git](https://github.com/RRonium/DeepDent.git)
cd DeepDent
pip install -r requirements.txt