import streamlit as st
import numpy as np
import cv2
import pickle
from skimage import feature
from skimage.color import rgb2gray
from scipy import stats
from PIL import Image
import io
import time

# ------------------- Page Config -------------------
st.set_page_config(page_title="GuavaSense - AI Plant Disease Detector", layout="wide")

# ------------------- Feature Extractor (same as your earlier implementation) -------------------
class MultiDomainFeatureExtractor:
    def __init__(self, img_size=128):
        self.img_size = img_size

    def extract_rgb_features(self, image):
        features = []
        for c in range(3):
            ch = image[:, :, c]
            features.extend([
                np.mean(ch), np.std(ch), np.median(ch),
                np.percentile(ch, 25), np.percentile(ch, 75),
                np.min(ch), np.max(ch), np.ptp(ch),
                stats.skew(ch.flatten()), stats.kurtosis(ch.flatten())
            ])
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        features.extend([
            np.mean(r / (g + 1e-8)), np.mean(g / (b + 1e-8)), np.mean(b / (r + 1e-8)),
            np.std(r / (g + 1e-8)), np.std(g / (b + 1e-8)), np.std(b / (r + 1e-8))
        ])
        total = np.sum(image, axis=2)
        features.extend([np.mean(total), np.std(total), stats.skew(total.flatten()),
                         stats.kurtosis(total.flatten())])
        return np.array(features)

    def extract_hsv_features(self, image):
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        features = []
        for c in range(3):
            ch = hsv[:, :, c]
            features.extend([
                np.mean(ch), np.std(ch), np.median(ch),
                np.percentile(ch, 10), np.percentile(ch, 90),
                np.min(ch), np.max(ch),
                stats.skew(ch.flatten()), stats.kurtosis(ch.flatten())
            ])
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = (s > 50)
        features.extend([
            np.mean(s * v), np.std(s * v),
            np.mean(h[mask]) if np.any(mask) else 0.0,
            np.std(h[mask]) if np.any(mask) else 0.0,
            np.sum(s > 100) / s.size, np.sum(v > 150) / v.size
        ])
        return np.array(features)

    def extract_texture_features(self, image):
        gray = rgb2gray(image)
        g8 = (gray * 255).astype(np.uint8)
        features = [
            np.mean(gray), np.std(gray), np.median(gray),
            stats.skew(gray.flatten()), stats.kurtosis(gray.flatten()),
            np.percentile(gray, 10), np.percentile(gray, 90)
        ]
        lbp = feature.local_binary_pattern(g8, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=10, density=True)
        features.extend(lbp_hist)
        try:
            glcm = feature.graycomatrix(g8, distances=[1], angles=[0], levels=256,
                                        symmetric=True, normed=True)
            features.extend([
                feature.graycoprops(glcm, 'contrast')[0, 0],
                feature.graycoprops(glcm, 'dissimilarity')[0, 0],
                feature.graycoprops(glcm, 'homogeneity')[0, 0],
                feature.graycoprops(glcm, 'energy')[0, 0],
                feature.graycoprops(glcm, 'correlation')[0, 0],
            ])
        except:
            features.extend([0, 0, 0, 0, 0])
        features.extend([
            np.mean(np.abs(np.diff(gray, axis=0))),
            np.mean(np.abs(np.diff(gray, axis=1))),
            np.std(np.abs(np.diff(gray, axis=0))),
            np.std(np.abs(np.diff(gray, axis=1)))
        ])
        return np.array(features)

    def extract_edge_features(self, image):
        gray = rgb2gray(image)
        g8 = (gray * 255).astype(np.uint8)
        feat = []
        edges = cv2.Canny(g8, 50, 150)
        feat.extend([np.sum(edges > 0) / edges.size, np.mean(edges), np.std(edges)])
        sx = cv2.Sobel(g8, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(g8, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        feat.extend([np.mean(mag), np.std(mag), np.mean(np.abs(sx)),
                     np.mean(np.abs(sy)), np.percentile(mag, 90), np.percentile(mag, 95)])
        lap = cv2.Laplacian(g8, cv2.CV_64F)
        feat.extend([np.mean(np.abs(lap)), np.std(lap),
                     np.sum(np.abs(lap) > np.std(lap)) / lap.size])
        orient = np.arctan2(sy, sx)
        hist, _ = np.histogram(orient.flatten(), bins=8, density=True)
        feat.extend(hist)
        return np.array(feat)

    def extract_histogram_features(self, image):
        gray = rgb2gray(image)
        g8 = (gray * 255).astype(np.uint8)
        feat = []
        hist, _ = np.histogram(g8, bins=32, range=(0, 256), density=True)
        feat.extend(hist)
        feat.extend([np.mean(hist), np.std(hist), stats.skew(hist), stats.kurtosis(hist),
                     int(np.argmax(hist)), np.sum(hist[:8]), np.sum(hist[-8:])])
        for c in range(3):
            ch_hist, _ = np.histogram(image[:, :, c], bins=16, range=(0, 1), density=True)
            feat.extend([np.mean(ch_hist), np.std(ch_hist), int(np.argmax(ch_hist))])
        p = hist[hist > 0]
        entropy = -np.sum(p * np.log2(p + 1e-8))
        feat.append(entropy)
        return np.array(feat)

    def extract_all_features(self, image):
        return np.concatenate([
            self.extract_rgb_features(image),
            self.extract_hsv_features(image),
            self.extract_texture_features(image),
            self.extract_edge_features(image),
            self.extract_histogram_features(image)
        ])

# ------------------- Model Loading -------------------
@st.cache_resource
def load_model(MODEL_PATH="guava_disease_model.pkl"):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        # Expected structure:
        # {
        #   'model': sklearn_model,
        #   'selected_features': array_of_indices,
        #   'label_map': {int_label: 'class_name'}
        # }
        required_keys = {'model', 'selected_features', 'label_map'}
        if not required_keys.issubset(set(model_data.keys())):
            st.error(f"Model file missing required keys. Found keys: {list(model_data.keys())}")
            return None
        return model_data
    except FileNotFoundError:
        st.error(f"Model file not found. Please put 'guava_disease_model.pkl' in the app directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ------------------- Preprocessing & Prediction Helpers -------------------
def preprocess_image(uploaded_file, target_size=(128, 128)):
    try:
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, target_size) / 255.0
        return img_resized
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def extract_features_for_model(image, model_data):
    try:
        extractor = MultiDomainFeatureExtractor()
        all_features = extractor.extract_all_features(image)
        sel_idx = model_data['selected_features']
        # Ensure indices exist
        if np.max(sel_idx) >= all_features.shape[0]:
            st.error("Selected feature indices go beyond extracted features size.")
            return None
        selected = all_features[sel_idx]
        return selected.reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_disease(image, model_data):
    try:
        X = extract_features_for_model(image, model_data)
        if X is None:
            return None
        model = model_data['model']
        label_map = model_data['label_map']
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba_all = model.predict_proba(X)[0]
            # If predict_proba returns shape (n_classes,), probability for predicted class:
            proba = float(proba_all[pred]) if pred < len(proba_all) else None
        else:
            proba = None
        class_name = label_map.get(pred, str(pred))
        return {"class": class_name, "confidence": proba}
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ------------------- UI CSS -------------------
st.markdown("""
    <style>
        .main-title { font-size: 2.2rem; font-weight: 700; color: #1b1b1b; margin-bottom: 0.3rem; }
        .sub-text { color: #555; font-size: 1rem; margin-bottom: 1.2rem; }
        .header { background-color:#4b8b5f; padding: 1rem; border-radius:6px; margin-bottom: 1.2rem; color: white; }
        .stButton>button { background-color: #FF6B35; color: white; border-radius: 8px; padding: 0.5rem 1rem; font-weight:600; }
        .stButton>button:hover { background-color: #e2572a; }
        .result-box { background:#f0f8f0; padding:0.8rem 1rem; border-left:4px solid #2E8B57; border-radius:6px; }
        .image-card { border-radius:12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); overflow:hidden; }
        .placeholder { color:#999; padding:1rem; text-align:center; }
    </style>
""", unsafe_allow_html=True)

# ------------------- Header & Description -------------------
st.markdown("<div class='header'><h3 style='margin:0'>GuavaSense</h3></div>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>Plants make a positive impact on your environment.</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Upload your guava plant leaf to identify diseases instantly with our AI-powered detection system.</div>", unsafe_allow_html=True)

# ------------------- Load Model -------------------
model_data = load_model()
if model_data is None:
    st.stop()  # stop further UI (user must fix model issue)

# ------------------- Main Layout -------------------
col_left, col_right = st.columns([2, 1], vertical_alignment="center")

with col_left:
    uploaded_file = st.file_uploader("Drag and drop your leaf image here", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.write(f"**File:** {uploaded_file.name} — {round(uploaded_file.size / 1024 / 1024, 2)} MB")
        # Buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            st.button("Upload Image", use_container_width=True, key="upload_btn_dummy")
        with btn_col2:
            if st.button("Predict", use_container_width=True, key="predict_action"):
                with st.spinner("Analyzing image..."):
                    time.sleep(0.3)  # small UX pause
                    img_for_model = preprocess_image(uploaded_file)
                    if img_for_model is not None:
                        res = predict_disease(img_for_model, model_data)
                        if res:
                            conf = res['confidence']
                            conf_text = f" ({conf*100:.2f}% confidence)" if conf is not None else ""
                            st.success(f"Prediction Result: {res['class']}{conf_text}")
    else:
        st.info("Please upload an image to begin.")

with col_right:
    # Use a container with a fixed min-height to keep image vertically aligned
    st.markdown("<div style='min-height:120px; display:flex; align-items:center; justify-content:center;'>", unsafe_allow_html=True)
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            st.image(img)
        except Exception as e:
            st.error(f"Cannot display uploaded image: {e}")
    else:
        st.markdown("<div class='placeholder image-card'>Your leaf image will appear here</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- About & Footer -------------------
st.markdown("---")
st.markdown("""
<div style="background:#f9f9f9; padding:1.2rem; border-radius:8px;">
  <h4 style="color:#2E8B57; margin-top:0;">About the Project</h4>
  <p>
          Hi this is Sreeja Bonthu(22MID0193) and C.Jahnavi(22MID0200) <br/>Together, we built this Guava Plant Disease Detection system to help farmers and plant enthusiasts quickly identify diseases in guava leaves.
Simply upload a guava leaf image, and our AI-powered system will detect whether it is Healthy or affected by diseases such as Canker, Leaf Spot, Mummification, or Rust, and show the confidence percentage for the prediction.
Our goal is to provide a simple, fast, and reliable tool that can help monitor plant health and support timely intervention to protect guava crops.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#888; margin-top:0.6rem;'>© 2025 GuavaSense</p>", unsafe_allow_html=True)