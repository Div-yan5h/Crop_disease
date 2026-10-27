# 🌾 Crop Disease Detection Using Machine Learning

**Course**: CSET485 — AI & Society Mini Project  
**Institution**: Department of Computer Science & Engineering, Bennett University  
**Team**:
- Lavanya Budhiraja (Roll No: S24CSEU0064)
- Nandini Singh (Roll No: S24CSEU2408)
- Divyansh Singh (Roll No: S24CSU2535)

---

## 📌 Project Overview
This project delivers a complete, lightweight, and academically rigorous classical machine learning pipeline for automated crop leaf disease classification. The solution classifies leaf images across **5 major agricultural staple crops** (**Corn**, **Potato**, **Rice**, **Wheat**, **Sugarcane**) into **17 disease and healthy classes**.

Instead of compute-intensive deep learning networks requiring dedicated GPUs, this system extracts **17 hand-crafted numerical color moments and Gray-Level Co-occurrence Matrix (GLCM) texture descriptors**, comparing three benchmark classical classifiers:
1. **K-Nearest Neighbors (KNN)**
2. **Random Forest (RF)**
3. **Support Vector Machine (SVM)**

All models are trained and evaluated under strictly identical conditions with an **80:20 stratified train/test split**. Preprocessing scalers and cross-validation hyperparameter tuning are fitted exclusively on training data, preventing data leakage.

---

## 🔍 Problem Statement
Crop diseases pose severe threats to global food security, crop yield, and agricultural farm livelihoods. Fungal, bacterial, and viral infections disrupt crop growth and cause devastating economic losses when detected late. 

The traditional inspection process relies on manual visual examination by agricultural extension workers, which is slow, labor-intensive, error-prone, and cannot scale across expansive fields or diverse crop varieties. Automating disease diagnosis through computational image analysis offers timely, accessible interventions. This project investigates whether lightweight, feature-engineered machine learning can achieve practical classification accuracy on consumer hardware without deep learning complexity.

---

## 📦 Dataset Source & Structure
- **Dataset**: [Five Crop Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset) (`shubham2703/five-crop-diseases-dataset`)
- **Total Archive Size**: ~4.3 GB (13,324 images)
- **Crops & Classes (17 total)**:
  - **Corn (Maize)**: *Common Rust*, *Gray Leaf Spot*, *Northern Leaf Blight*, *Healthy*
  - **Potato**: *Early Blight*, *Late Blight*, *Healthy*
  - **Rice**: *Brown Spot*, *Leaf Blast*, *Neck Blast*, *Healthy*
  - **Wheat**: *Brown Rust*, *Yellow Rust*, *Healthy*
  - **Sugarcane**: *Bacterial Blight*, *Red Rot*, *Healthy*
- **Experimental Subset**: The dataset has 17 classes across 5 crops. With a cap of 150 images per class, the theoretical maximum would be $17 \times 150 = 2,550$ images if every class had at least 150 images. In this dataset, the 3 Sugarcane classes contain exactly 100 images each (300 images total, all 100% retained), while the other 14 classes are sampled at 150 images each ($14 \times 150 = 2,100$ images). This produces an exact representative stratified subset of **2,400 images**, partitioned 80:20 into **1,920 training samples** and **480 held-out test samples**.

---

## 🔬 Features & Machine Learning Models

### 1. Preprocessing
- Image loading, channel verification, and conversion to RGB color space.
- Uniform resizing to $64 \times 64$ pixels to standardize variations across different camera resolutions.

### 2. Hand-Crafted Feature Vector (17 Dimensions per Image)
- **Color Moments (12 features)**:
  - **RGB**: Mean and standard deviation of Red, Green, and Blue channels (captures leaf pigment shifts and necrosis).
  - **HSV**: Mean and standard deviation of Hue, Saturation, and Value channels (captures chlorosis and rust discoloration while remaining resilient to lighting variations).
- **GLCM Texture Descriptors (5 features)**:
  - Grayscale quantization (64 levels) with Gray-Level Co-occurrence Matrix (distance 1, angle 0).
  - **Contrast**: Measures local intensity variation and lesion edge prominence.
  - **Homogeneity**: Measures tissue uniformity (healthy leaves exhibit high homogeneity).
  - **Energy**: Measures textural orderliness.
  - **Correlation**: Measures gray-level linear dependency between neighboring pixels.
  - **Entropy**: Measures surface textural complexity and irregularity.

### 3. Classification Algorithms & Tuning (3-Fold CV on Training Split)
- **K-Nearest Neighbors (KNN)**: Tuned over neighbor count $k \in \{3, 5, 7, 9\}$ and weights $\in \{\text{'uniform'}, \text{'distance'}\}$.
- **Random Forest (RF)**: Tuned over estimator count $\in \{50, 100\}$ and max depth $\in \{10, 20, \text{None}\}$.
- **Support Vector Machine (SVM)**: Standardized features, tuned over regularization parameter $C \in \{1.0, 10.0\}$ and kernel $\in \{\text{'linear'}, \text{'rbf'}\}$.

---

## 📊 Actual Verified Results (Evaluated on 480 Held-Out Test Images)

All three models were evaluated on the exact same held-out test set ($N=480$ images across 17 classes):

| Model | Test Accuracy (%) | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | Training Time | Test Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** 🏆 | **84.38%** | **0.8501** | **0.8438** | **0.8427** | **0.8397** | **0.15s** | **38.0 ms** |
| **Random Forest (100 Trees)** | **77.92%** | **0.7786** | **0.7792** | **0.7736** | **0.7661** | **1.12s** | **8.7 ms** |
| **K-Nearest Neighbors (k=5)** | **75.42%** | **0.7518** | **0.7542** | **0.7466** | **0.7370** | **1.45s** | **58.8 ms** |

### Key Analytical Findings
- **Top Performer**: **Support Vector Machine (RBF kernel, $C=10.0$)** achieved the highest accuracy (**84.38%**) and macro F1-score (**0.8397**), demonstrating that non-linear kernel projection effectively separates disease symptoms in the standardized 17D feature space.
- **Fastest Inference**: **Random Forest** achieved the lowest prediction latency (**8.7 ms** across all 480 test queries).
- **Class Separation**: Healthy leaf classes across all 5 crops achieved near-100% precision and recall due to high green intensity and uniform GLCM homogeneity. Diseases with similar chlorotic yellowing (e.g., Corn Gray Leaf Spot vs. Corn Northern Leaf Blight) accounted for minor misclassifications.

---

## ⚠️ Limitations & Future Work
1. **Macroscopic Feature Summarization**: Hand-crafted color and texture moments summarize global image statistics. When different fungal diseases exhibit similar chlorotic yellowing on related cereal grains, global moments can encounter ambiguity.
2. **Controlled vs. Field Imagery**: Lab-captured datasets have relatively uniform backgrounds; real-world field images with soil, shadows, and multiple overlapping leaves may require lesion segmentation.
3. **Deep Learning Benchmarking**: When GPU hardware is available, comparing this classical baseline against lightweight transfer learning (e.g., MobileNetV2) provides a valuable trade-off analysis between compute cost and spatial discrimination.

---

## 📂 Repository File Structure
```
Ai society/
├── crop_disease_detection.ipynb  # Primary self-contained Jupyter notebook
├── TASKS.md                      # Development and verification task checklist
├── requirements.txt              # Minimal Python dependencies
├── .gitignore                    # Excludes datasets, venvs, caches, and model binaries
├── README.md                     # Complete project documentation
└── results/                      # Generated evaluation artifacts and figures
    ├── model_comparison.csv                 # Official benchmark table
    ├── experiment_summary.txt               # Text summary of experiment
    ├── class_mapping.json                   # Class indices & best hyperparameters
    ├── class_distribution.png               # Class distribution bar chart
    ├── sample_images.png                    # Sample leaf image grid
    ├── model_performance_comparison.png     # Performance comparison bar chart
    ├── confusion_matrices_combined.png      # 3-in-1 test confusion matrices
    └── confusion_matrix_*.png               # Individual confusion matrices
```

*(Note: Raw image datasets, intermediate `.npz` feature caches, and large `.joblib` model binaries are excluded from Git via `.gitignore` to keep the repository lightweight.)*

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- Python 3.9 or higher.

### 2. Environment Setup
Clone the repository and create a clean Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/crop-disease-detection.git
cd crop-disease-detection

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate       # On macOS / Linux
# .venv\Scripts\activate        # On Windows

# Install required dependencies
pip install -r requirements.txt
```

### 3. Dataset Download & Setup
1. Download the **Five Crop Diseases Dataset** from Kaggle:
   👉 [https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset](https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset)
2. Extract the archive into the `./data` directory in the project root:
   ```bash
   mkdir -p data
   unzip -q ~/Downloads/archive.zip -d data/
   ```
3. The expected folder path is:
   `./data/Crop Diseases Dataset/Crop Diseases/Crop___Disease`
   *(If your folder is located elsewhere, you can configure `DATASET_PATH` inside the notebook or set the `CROP_DATASET_PATH` environment variable).*

### 4. Running the Notebook
Launch Jupyter Notebook and open `crop_disease_detection.ipynb`:

```bash
jupyter notebook crop_disease_detection.ipynb
```
Select the `.venv` Python kernel and run all cells sequentially (**Kernel → Restart & Run All**).
All outputs, figures, and benchmark metrics will execute automatically and refresh the files in `./results/`.
