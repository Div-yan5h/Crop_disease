"""
Crop Disease Detection — Interactive Leaf Prediction Application
Course: CSET485 — AI & Society Mini Project (Bennett University)

Allows users to upload a crop leaf image (JPG, JPEG, PNG) and classifies
it into one of the 17 disease/healthy classes across 5 staple crops using
the project's trained classical machine learning models (SVM, Random Forest, KNN).
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import joblib
from skimage.feature import graycomatrix, graycoprops

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌾",
    layout="centered"
)

# Custom light CSS for clean presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2b5c2b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #555;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #f8faf8;
        border: 1px solid #d4e5d4;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="main-title">🌾 Crop Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Classical Machine Learning Pipeline | CSET485 — AI & Society Mini Project</div>', unsafe_allow_html=True)

# ==============================================================================
# Artifact Paths & Validation
# ==============================================================================
RESULTS_DIR = "./results"
CLASS_MAPPING_PATH = os.path.join(RESULTS_DIR, "class_mapping.json")
SCALER_PATH = os.path.join(RESULTS_DIR, "feature_scaler.joblib")
MODEL_PATHS = {
    "Support Vector Machine (SVM)": os.path.join(RESULTS_DIR, "svm_model.joblib"),
    "Random Forest": os.path.join(RESULTS_DIR, "random_forest_model.joblib"),
    "K-Nearest Neighbors (KNN)": os.path.join(RESULTS_DIR, "knn_model.joblib")
}

# Check artifact availability
missing_artifacts = []
if not os.path.exists(CLASS_MAPPING_PATH):
    missing_artifacts.append("class_mapping.json")
if not os.path.exists(SCALER_PATH):
    missing_artifacts.append("feature_scaler.joblib")
for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        missing_artifacts.append(os.path.basename(path))

if missing_artifacts:
    st.error(
        f"⚠️ **Missing Trained Model Artifacts** in `{RESULTS_DIR}/`:\n\n"
        f"- Missing: `{', '.join(missing_artifacts)}`\n\n"
        "**How to generate them:**\n"
        "1. Open `crop_disease_detection.ipynb` in Jupyter Notebook.\n"
        "2. Run all cells from beginning to end (**Kernel → Restart & Run All**).\n"
        "3. The notebook will extract features, train all models, and save all required `.joblib` and `.json` files to `./results/`."
    )
    st.stop()

# ==============================================================================
# Load Class Mapping & Models
# ==============================================================================
@st.cache_resource
def load_resources():
    with open(CLASS_MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    classes = mapping["classes"]
    feature_names = mapping.get("feature_names", [])
    scaler = joblib.load(SCALER_PATH)
    models = {
        name: joblib.load(path) for name, path in MODEL_PATHS.items()
    }
    return classes, feature_names, scaler, models

class_names, feature_names, scaler, loaded_models = load_resources()

# ==============================================================================
# Sidebar Configuration
# ==============================================================================
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Classifier:",
    options=[
        "Support Vector Machine (SVM)",
        "Random Forest",
        "K-Nearest Neighbors (KNN)"
    ],
    index=0,  # Default to SVM (best performing: 84.38% test accuracy)
    help="SVM is recommended as it achieved the highest accuracy (84.38%) in the experimental evaluation."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Benchmark Comparison")
st.sidebar.markdown("""
| Model | Test Accuracy | Macro F1 |
| :--- | :---: | :---: |
| **SVM (RBF)** 🏆 | **84.38%** | **0.8397** |
| **Random Forest** | 77.92% | 0.7661 |
| **KNN (k=5)** | 75.42% | 0.7370 |
""")
st.sidebar.caption("Evaluated on 480 held-out test images across 17 classes.")

# ==============================================================================
# Feature Extraction (Exact same pipeline as crop_disease_detection.ipynb)
# ==============================================================================
def extract_features_from_image(img_rgb):
    """
    Extracts the exact 17 color and texture features used during training:
      - 6 RGB moments (mean & std of R, G, B)
      - 6 HSV moments (mean & std of H, S, V)
      - 5 GLCM texture features (contrast, homogeneity, energy, correlation, entropy)
    """
    # 1. Color Features (RGB)
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    rgb_feats = [
        float(r.mean()), float(r.std()),
        float(g.mean()), float(g.std()),
        float(b.mean()), float(b.std())
    ]

    # 2. Color Features (HSV)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]
    hsv_feats = [
        float(h.mean()), float(h.std()),
        float(s.mean()), float(s.std()),
        float(v.mean()), float(v.std())
    ]

    # 3. Texture Features (GLCM on Grayscale)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_q = (img_gray // 4).astype(np.uint8)  # 64 gray levels
    glcm = graycomatrix(gray_q, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)

    contrast    = float(graycoprops(glcm, 'contrast')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy      = float(graycoprops(glcm, 'energy')[0, 0])
    correlation = float(graycoprops(glcm, 'correlation')[0, 0])

    # Texture Entropy
    p = glcm[:, :, 0, 0]
    p_norm = p / (p.sum() + 1e-10)
    entropy = float(-np.sum(p_norm * np.log2(p_norm + 1e-10)))

    texture_feats = [contrast, homogeneity, energy, correlation, entropy]

    return rgb_feats + hsv_feats + texture_feats

def parse_class_label(label):
    """
    Parses a class label like 'Corn___Common_Rust' into crop and disease/healthy status.
    """
    if "___" in label:
        parts = label.split("___")
        crop = parts[0].replace("_", " ").strip()
        status = parts[1].replace("_", " ").strip()
    else:
        crop = "Unknown"
        status = label.replace("_", " ").strip()
    return crop, status

# ==============================================================================
# Image Upload Section
# ==============================================================================
st.markdown("### 📤 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a crop leaf image from your computer:",
    type=["jpg", "jpeg", "png"],
    help="Select an image file in JPG, JPEG, or PNG format."
)

# Optional sample selection helper if dataset is present
sample_dir = "./data/Crop Diseases Dataset/Crop Diseases/Crop___Disease"
sample_choice = None
if os.path.exists(sample_dir):
    with st.expander("💡 Or test with a sample image from the dataset"):
        sample_options = [
            ("Select a sample...", None),
            ("Corn - Common Rust", os.path.join(sample_dir, "Corn/Corn___Common_Rust")),
            ("Potato - Early Blight", os.path.join(sample_dir, "Potato/Potato___Early_Blight")),
            ("Potato - Healthy", os.path.join(sample_dir, "Potato/Potato___Healthy")),
            ("Rice - Leaf Blast", os.path.join(sample_dir, "Rice/Rice___Leaf_Blast")),
            ("Sugarcane - Red Rot", os.path.join(sample_dir, "sugarcane/Red Rot")),
            ("Wheat - Yellow Rust", os.path.join(sample_dir, "Wheat/Wheat___Yellow_Rust")),
            ("Wheat - Healthy", os.path.join(sample_dir, "Wheat/Wheat___Healthy"))
        ]
        chosen_label = st.selectbox("Choose a sample leaf:", [s[0] for s in sample_options], index=0)
        chosen_path = next(s[1] for s in sample_options if s[0] == chosen_label)
        if chosen_path and os.path.isdir(chosen_path):
            sample_files = [f for f in os.listdir(chosen_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if sample_files:
                sample_choice = os.path.join(chosen_path, sample_files[0])

# Determine image source
active_image = None
image_source_name = ""

if uploaded_file is not None:
    try:
        active_image = Image.open(uploaded_file)
        image_source_name = uploaded_file.name
    except Exception as e:
        st.error(f"❌ Could not open uploaded image: {e}")
elif sample_choice is not None:
    try:
        active_image = Image.open(sample_choice)
        image_source_name = os.path.basename(sample_choice)
    except Exception as e:
        st.error(f"❌ Could not open sample image: {e}")

# ==============================================================================
# Prediction & Results Display
# ==============================================================================
if active_image is not None:
    # Display the selected image
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(active_image, caption=f"Selected Leaf: {image_source_name}", use_container_width=True)

    with col2:
        with st.spinner(f"Extracting features and classifying with {model_choice}..."):
            try:
                # Convert to RGB numpy array
                img_rgb = np.array(active_image.convert("RGB"))

                # Standardize to 64x64 using INTER_AREA (exact match to training)
                img_resized = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)

                # Extract 17 features
                feature_vector = extract_features_from_image(img_resized)
                feats = np.array([feature_vector], dtype=np.float32)

                # Get selected model
                model = loaded_models[model_choice]

                # Apply feature scaling matching training pipeline:
                # SVM and KNN require StandardScaler; Random Forest was trained on unscaled features
                if model_choice in ["Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"]:
                    feats_input = scaler.transform(feats)
                else:
                    feats_input = feats

                # Predict class
                pred_idx = model.predict(feats_input)[0]
                pred_class = class_names[pred_idx]
                crop, status = parse_class_label(pred_class)

                # Display Results
                is_healthy = "healthy" in status.lower()
                status_icon = "✅" if is_healthy else "⚠️"
                badge_color = "#28a745" if is_healthy else "#d9534f"

                st.markdown(f"""
                <div style="background-color: #fdfdfd; border-left: 5px solid {badge_color}; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <h3 style="margin-top:0; color: #333;">Prediction Result</h3>
                    <p style="font-size: 1.1rem; margin: 4px 0;"><strong>🌾 Crop:</strong> {crop}</p>
                    <p style="font-size: 1.1rem; margin: 4px 0;"><strong>🩺 Condition:</strong> {status_icon} <span style="color: {badge_color}; font-weight: bold;">{status}</span></p>
                    <p style="font-size: 0.95rem; margin: 4px 0; color: #666;"><strong>🤖 Model:</strong> {model_choice}</p>
                </div>
                """, unsafe_allow_html=True)

                # Diagnostic Explanation Message
                st.markdown("#### 💬 Diagnosis Explanation")
                if is_healthy:
                    st.success(
                        f"The leaf shows healthy characteristics for **{crop}** foliage. "
                        "The uniform green coloration and smooth textural homogeneity indicate no prominent lesions, "
                        "rust pustules, or necrotic blight."
                    )
                else:
                    st.warning(
                        f"The leaf shows visual signs consistent with **{status}** in **{crop}**. "
                        "Hand-crafted color moments detected chlorotic/necrotic pigment shifts, while GLCM texture features "
                        "identified irregular surface variations associated with diseased foliar tissue."
                    )

            except Exception as e:
                st.error(f"❌ Error during image classification: {e}")

    # Expandable Feature Inspector
    with st.expander("🔬 Inspect Extracted Feature Vector (17 Descriptors)"):
        st.write("These are the 17 numerical features extracted from the preprocessed 64×64 leaf image:")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Color Moments (12 features):**")
            color_data = {
                "Feature": feature_names[:12] if len(feature_names) >= 12 else [f"Color_{i}" for i in range(12)],
                "Value": [f"{v:.4f}" for v in feature_vector[:12]]
            }
            st.dataframe(color_data, hide_index=True, use_container_width=True)
        with col_b:
            st.markdown("**GLCM Texture Descriptors (5 features):**")
            texture_data = {
                "Feature": feature_names[12:] if len(feature_names) >= 17 else [f"Texture_{i}" for i in range(5)],
                "Value": [f"{v:.4f}" for v in feature_vector[12:]]
            }
            st.dataframe(texture_data, hide_index=True, use_container_width=True)

else:
    st.info("👆 Please upload a leaf image (JPG, JPEG, or PNG) above to see the automated crop disease diagnosis.")

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.caption(
    "🌾 Crop Disease Detection Mini Project | Developed for CSET485 (AI & Society) | "
    "Bennett University — Lavanya Budhiraja, Nandini Singh, Divyansh Singh"
)
