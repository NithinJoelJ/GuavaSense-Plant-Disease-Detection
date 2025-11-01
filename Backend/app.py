"""
Guava Plant Disease Prediction Backend API
Flask REST API for serving the trained SVM model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import pickle
import os
from skimage import feature
from skimage.color import rgb2gray
from scipy import stats
import base64
from io import BytesIO
from PIL import Image
import traceback

app = Flask(__name__)
CORS(app)


# ============================================
# Multi-Domain Feature Extractor
# ============================================
class MultiDomainFeatureExtractor:
    """Advanced multi-domain feature extraction for plant disease detection"""

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


# ============================================
# Load Model
# ============================================
MODEL_PATH = 'guava_disease_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        selected_features = model_data['selected_features']
        label_map = model_data['label_map']

    print("✅ Model loaded successfully!")
    print(f"Classes: {list(label_map.values())}")
    print(f"Selected features: {len(selected_features)}")
except Exception as e:
    print(f"⚠️ Warning: Model not found. Please train and save the model first.")
    print(f"Error: {e}")
    model = None
    selected_features = None
    label_map = None

extractor = MultiDomainFeatureExtractor()


# ============================================
# Helper Functions
# ============================================
def preprocess_image(image_data):
    try:
        if isinstance(image_data, str):
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_bytes))
        else:
            img = Image.open(image_data)

        img = img.convert('RGB')
        img_array = np.array(img)
        img_resized = cv2.resize(img_array, (128, 128)) / 255.0

        return img_resized
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def extract_features(image):
    try:
        all_features = extractor.extract_all_features(image)
        selected = all_features[selected_features]
        return selected.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Error extracting features: {str(e)}")


# ============================================
# API Endpoints
# ============================================
@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Guava Disease Prediction API',
        'model_loaded': model is not None,
        'version': '1.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        image_data = None
        if 'file' in request.files:
            image_data = request.files['file']
        else:
            try:
                json_data = request.get_json(force=True) or {}
                image_data = json_data.get('image', None)
            except:
                pass

        if image_data is None:
            return jsonify({'error': 'No image provided'}), 400

        img = preprocess_image(image_data)
        features = extract_features(img)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        result = {
            'success': True,
            'prediction': {
                'class': label_map[prediction],
                'class_id': int(prediction),
                'confidence': float(probabilities[prediction])
            },
            'probabilities': {label_map[i]: float(probabilities[i]) for i in range(len(label_map))},
            'top_3_predictions': [
                {'class': label_map[i], 'confidence': float(probabilities[i])}
                for i in np.argsort(probabilities)[-3:][::-1]
            ]
        }

        return jsonify(result)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"[ERROR] {traceback_str}")
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback_str}), 500


# ============================================
# Run Server
# ============================================
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)
