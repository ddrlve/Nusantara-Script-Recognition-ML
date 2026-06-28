import time
import json
from pathlib import Path
from math import pi
import numpy as np
import pandas as pd
import cv2
import joblib
import streamlit as st
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import skeletonize

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / 'models'

if not ARTIFACT_DIR.exists():
    ARTIFACT_DIR = APP_DIR

with open(ARTIFACT_DIR / 'inference_config.json', 'r') as _f:
    CFG = json.load(_f)

CLASS_NAMES = np.load(ARTIFACT_DIR / 'class_names.npy', allow_pickle=True)
N_CLASSES = len(CLASS_NAMES)
IMG_SIZE = int(CFG.get('img_size', 96))

CONF_THRESHOLD = float(CFG.get('guardrails', {}).get('conf_threshold', 0.45))
MARGIN_THRESHOLD = float(CFG.get('guardrails', {}).get('margin_threshold', 0.08))
MULTI_OBJECT_REJECT = bool(CFG.get('guardrails', {}).get('multi_object_reject', True))
LARGE_COMP_AREA_RATIO = float(CFG.get('guardrails', {}).get('large_component_area_ratio', 0.04))
MAX_LARGE_COMPONENTS = int(CFG.get('guardrails', {}).get('max_large_components', 1))

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def _auto_binary_foreground(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.fastNlMeansDenoising(g, None, h=8, templateWindowSize=7, searchWindowSize=21)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    
    _, b_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    candidates = []
    for cand in [b_inv, b]:
        ratio = (cand > 0).mean()
        penalty = abs(ratio - 0.18)
        if ratio < 0.01 or ratio > 0.80:
            penalty += 10
        candidates.append((penalty, cand))
        
    binary = min(candidates, key=lambda x: x[0])[1]
    
    k2 = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k2, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k2, iterations=1)
    return binary.astype(np.uint8)

def _crop_pad_center(binary: np.ndarray, out_size: int = IMG_SIZE, pad_ratio: float = 0.18) -> np.ndarray:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return cv2.resize(binary, (out_size, out_size), interpolation=cv2.INTER_AREA)
        
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = binary[y1:y2+1, x1:x2+1]
    
    h, w = crop.shape
    side = max(h, w)
    pad = int(side * pad_ratio)
    canvas_sz = side + 2 * pad
    canvas = np.zeros((canvas_sz, canvas_sz), dtype=np.uint8)
    
    yoff = (canvas_sz - h) // 2
    xoff = (canvas_sz - w) // 2
    canvas[yoff:yoff+h, xoff:xoff+w] = crop
    
    m = cv2.moments(canvas)
    if m['m00'] > 0:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        shift_x = canvas_sz // 2 - cx
        shift_y = canvas_sz // 2 - cy
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, M, (canvas_sz, canvas_sz), borderValue=0)
        
    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA).astype(np.uint8)

def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert('L'), dtype=np.uint8)
    binary = _auto_binary_foreground(arr)
    return _crop_pad_center(binary, out_size=IMG_SIZE)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _build_gabor_kernels(freqs, thetas):
    kernels = []
    for freq in freqs:
        sigma = 0.4 / freq
        lambd = 1.0 / freq
        ks = int(2 * np.ceil(3.0 * sigma) + 1)
        ks = min(ks | 1, 63)
        for theta in thetas:
            kr = cv2.getGaborKernel((ks, ks), sigma, theta, lambd, 0.5, 0, cv2.CV_32F)
            ki = cv2.getGaborKernel((ks, ks), sigma, theta, lambd, 0.5, pi / 2, cv2.CV_32F)
            kernels.append((kr, ki))
    return kernels

class FeatureExtractor:
    GABOR_FREQS = [0.08, 0.14, 0.22, 0.32]
    GABOR_THETAS = [0, pi/8, pi/4, 3*pi/8, pi/2, 5*pi/8, 3*pi/4, 7*pi/8]

    def __init__(self):
        self._gabor_kernels = _build_gabor_kernels(self.GABOR_FREQS, self.GABOR_THETAS)

    def _fg(self, img):
        return (img > 0).astype(np.uint8)

    def hog(self, img):
        return hog(
            img,
            orientations=12,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True,
        ).astype(np.float32)

    def lbp(self, img):
        pattern = local_binary_pattern(img, P=24, R=3, method='uniform')
        hist, _ = np.histogram(pattern.ravel(), bins=26, range=(0, 26), density=True)
        return hist.astype(np.float32)

    def gabor(self, img):
        img_f = img.astype(np.float32) / 255.0
        n = len(self._gabor_kernels)
        feats = np.empty(n * 4, dtype=np.float32)
        for i, (kr, ki) in enumerate(self._gabor_kernels):
            real = cv2.filter2D(img_f, cv2.CV_32F, kr)
            imag = cv2.filter2D(img_f, cv2.CV_32F, ki)
            mag = np.sqrt(real * real + imag * imag)
            b = i * 4
            feats[b], feats[b+1] = mag.mean(), mag.std()
            feats[b+2], feats[b+3] = np.percentile(mag, 75), np.percentile(mag, 90)
        return feats

    def zoning(self, img):
        fg = self._fg(img).astype(np.float32)
        feats = []
        for grid in [4, 8]:
            cell = IMG_SIZE // grid
            crop = fg[:grid*cell, :grid*cell]
            zones = crop.reshape(grid, cell, grid, cell).transpose(0, 2, 1, 3).reshape(grid * grid, cell * cell)
            feats.extend(zones.mean(axis=1))
            feats.extend(zones.std(axis=1))
            
        h = fg.sum(axis=1)
        v = fg.sum(axis=0)
        h = h / (h.max() + 1e-8)
        v = v / (v.max() + 1e-8)
        feats.extend(h)
        feats.extend(v)
        
        for bins in [8, 16]:
            feats.extend(h.reshape(bins, IMG_SIZE // bins).mean(axis=1))
            feats.extend(v.reshape(bins, IMG_SIZE // bins).mean(axis=1))
        return np.asarray(feats, dtype=np.float32)

    def shape_features(self, img):
        fg = self._fg(img)
        feats = []
        area = float(fg.sum())
        total = float(fg.size)
        feats.append(area / (total + 1e-8))
        
        moments = cv2.moments((fg * 255).astype(np.uint8))
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        feats.extend(hu)
        
        contours, _ = cv2.findContours((fg * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            c_area = float(cv2.contourArea(cnt))
            perim = float(cv2.arcLength(cnt, True))
            x, y, w, h = cv2.boundingRect(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            aspect = w / (h + 1e-8)
            extent = c_area / (w * h + 1e-8)
            solidity = c_area / (hull_area + 1e-8)
            circularity = 4 * np.pi * c_area / (perim * perim + 1e-8)
            feats.extend([c_area / total, perim / (IMG_SIZE * 4), aspect, extent, solidity, circularity])
        else:
            feats.extend([0, 0, 0, 0, 0, 0])
            
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        comp_count = max(0, num_labels - 1)
        comp_areas = stats[1:, cv2.CC_STAT_AREA] if comp_count else np.array([0])
        feats.extend([
            comp_count / 10.0,
            float(comp_areas.max()) / (area + 1e-8),
            float(comp_areas.mean()) / (area + 1e-8),
        ])
        
        try:
            skel = skeletonize(fg > 0).astype(np.uint8)
            skel_cnt = float(skel.sum())
            kernel = np.ones((3, 3), dtype=np.uint8)
            neigh = cv2.filter2D(skel, -1, kernel) - skel
            endpoints = int(((skel == 1) & (neigh == 1)).sum())
            junctions = int(((skel == 1) & (neigh >= 3)).sum())
            feats.extend([skel_cnt / (area + 1e-8), endpoints / 20.0, junctions / 20.0])
        except Exception:
            feats.extend([0, 0, 0])
            
        return np.asarray(feats, dtype=np.float32)

    def extract(self, img: np.ndarray) -> np.ndarray:
        return np.concatenate([
            self.hog(img),
            self.lbp(img),
            self.gabor(img),
            self.zoning(img),
            self.shape_features(img),
        ]).astype(np.float32)

# ---------------------------------------------------------------------------
# Model class definitions
# ---------------------------------------------------------------------------
class TwoStageAksaraClassifier:
    def __init__(self, script_model, script_encoder, char_models, char_encoders, label_to_script, n_classes):
        self.script_model = script_model
        self.script_encoder = script_encoder
        self.char_models = char_models
        self.char_encoders = char_encoders
        self.label_to_script = np.asarray(label_to_script)
        self.classes_ = np.arange(n_classes)
        self.n_classes = int(n_classes)

    def _align_script_proba(self, X):
        p = np.asarray(self.script_model.predict_proba(X), dtype=np.float32)
        out = np.zeros((X.shape[0], len(self.script_encoder.classes_)), dtype=np.float32)
        model_classes = np.asarray(self.script_model.classes_).astype(int)
        for src_i, cls in enumerate(model_classes):
            if cls < out.shape[1]:
                out[:, cls] = p[:, src_i]
        return out

    def predict_proba(self, X):
        script_p = self._align_script_proba(X)
        final = np.zeros((X.shape[0], self.n_classes), dtype=np.float32)
        for s_idx, script_name in enumerate(self.script_encoder.classes_):
            if script_name not in self.char_models:
                continue
            cm = self.char_models[script_name]
            ce = self.char_encoders[script_name]
            raw = np.asarray(cm.predict_proba(X), dtype=np.float32)
            local_classes = np.asarray(cm.classes_).astype(int)
            global_labels = ce.inverse_transform(local_classes)
            for src_j, glabel in enumerate(global_labels):
                final[:, int(glabel)] += script_p[:, s_idx] * raw[:, src_j]
        row_sum = final.sum(axis=1, keepdims=True)
        return final / (row_sum + 1e-8)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class SoftVoteEnsemble:
    def __init__(self, models, model_keys=None, weights=None):
        self.models = models
        self.model_keys = model_keys if model_keys is not None else list(models.keys())
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        self.classes_ = np.arange(N_CLASSES)

    def _align_proba_one(self, model, X):
        raw_proba = np.asarray(model.predict_proba(X))
        if hasattr(model, 'classes_'):
            model_classes = np.asarray(model.classes_).astype(int)
            aligned = np.zeros((raw_proba.shape[0], len(self.classes_)), dtype=np.float32)
            label_to_pos = {int(lbl): i for i, lbl in enumerate(self.classes_)}
            for src_idx, cls in enumerate(model_classes):
                cls = int(cls)
                if cls in label_to_pos and src_idx < raw_proba.shape[1]:
                    aligned[:, label_to_pos[cls]] = raw_proba[:, src_idx]
            return aligned
        return raw_proba

    def predict_proba(self, X):
        probas, weights = [], []
        for i, key in enumerate(self.model_keys):
            model = self.models[key]
            if not hasattr(model, 'predict_proba'):
                continue
            probas.append(self._align_proba_one(model, X))
            weights.append(1.0 if self.weights is None else float(self.weights[i]))
        if not probas:
            raise RuntimeError('No valid predict_proba method found in the ensemble.')
            
        W = np.asarray(weights, dtype=np.float32)
        W = W / (W.sum() + 1e-8)
        out = np.zeros_like(probas[0], dtype=np.float32)
        for p, w in zip(probas, W):
            out += w * p
        return out

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

class GuardedAksaraClassifier:
    def __init__(self, base_model, specialist_models=None, n_classes=None):
        self.base_model = base_model
        self.specialist_models = specialist_models or {}
        self.n_classes = int(n_classes if n_classes is not None else N_CLASSES)
        self.classes_ = np.arange(self.n_classes)

    def predict_proba(self, X):
        proba = np.asarray(self.base_model.predict_proba(X), dtype=np.float32)
        if proba.shape[1] != self.n_classes:
            raise RuntimeError(f'proba shape {proba.shape} does not match n_classes={self.n_classes}')
            
        for (a_id, b_id), sp in self.specialist_models.items():
            pair = np.array([a_id, b_id], dtype=int)
            pair_total = proba[:, pair].sum(axis=1)
            top2 = np.argsort(proba, axis=1)[:, -2:]
            involved = np.array([bool(set(row.tolist()) & set(pair.tolist())) for row in top2])
            use_sp = involved & (pair_total > 0.20)
            
            if not np.any(use_sp):
                continue
                
            raw_sp = np.asarray(sp.predict_proba(X[use_sp]), dtype=np.float32)
            sp_classes = np.asarray(sp.classes_).astype(int)
            aligned_pair = np.zeros((raw_sp.shape[0], 2), dtype=np.float32)
            
            for j, cls in enumerate(sp_classes):
                if cls == a_id:
                    aligned_pair[:, 0] = raw_sp[:, j]
                elif cls == b_id:
                    aligned_pair[:, 1] = raw_sp[:, j]
                    
            proba[use_sp, a_id] = pair_total[use_sp] * aligned_pair[:, 0]
            proba[use_sp, b_id] = pair_total[use_sp] * aligned_pair[:, 1]
            
        return proba / (proba.sum(axis=1, keepdims=True) + 1e-8)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    def _find(patterns):
        for pat in patterns:
            hits = sorted(ARTIFACT_DIR.glob(pat))
            if hits:
                return hits[0]
        raise FileNotFoundError(f'Artifact not found: {patterns}')
        
    scaler = joblib.load(_find(['scaler_*.pkl', 'scaler.pkl']))
    pca = joblib.load(_find(['pca_model_*.pkl', 'pca_model.pkl']))
    
    model_path = ARTIFACT_DIR / 'guarded_model.pkl'
    if not model_path.exists():
        model_path = ARTIFACT_DIR / 'stacking_model.pkl'
    model = joblib.load(model_path)
    
    return scaler, pca, model, FeatureExtractor()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def count_large_components(img: np.ndarray) -> int:
    binary = (img > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0
    img_area = img.shape[0] * img.shape[1]
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int((areas > img_area * LARGE_COMP_AREA_RATIO).sum())

def apply_script_filter(proba: np.ndarray, script: str) -> np.ndarray:
    if not script or script == 'All Scripts':
        return proba
    allowed = np.array(
        [i for i, n in enumerate(CLASS_NAMES) if str(n).startswith(script + '_')],
        dtype=int,
    )
    if len(allowed) == 0:
        return proba
    filtered = np.zeros_like(proba, dtype=np.float32)
    filtered[allowed] = proba[allowed]
    return filtered / (filtered.sum() + 1e-8)

def predict_pil(pil_img: Image.Image, script: str = 'All Scripts', top_k: int = 10, use_guardrails: bool = True) -> dict:
    scaler, pca, model, extractor = load_pipeline()
    
    img = preprocess_pil(pil_img)
    n_comp = count_large_components(img)
    feat = extractor.extract(img)
    
    t0 = time.perf_counter()
    feat_s = scaler.transform(feat.reshape(1, -1))
    feat_p = pca.transform(feat_s)
    proba = model.predict_proba(feat_p)[0]
    proba = apply_script_filter(proba, script)
    latency_ms = (time.perf_counter() - t0) * 1000
    
    top_idx = np.argsort(proba)[::-1][:top_k]
    topk = [(str(CLASS_NAMES[i]), float(proba[i])) for i in top_idx]
    
    label = str(CLASS_NAMES[top_idx[0]])
    conf = float(proba[top_idx[0]])
    margin = float(proba[top_idx[0]] - proba[top_idx[1]]) if len(top_idx) > 1 else conf
    
    script_out = label.split('_', 1)[0] if '_' in label else label
    char_out = label.split('_', 1)[1] if '_' in label else label
    
    accepted = True
    reasons = []
    
    if use_guardrails:
        if conf < CONF_THRESHOLD:
            accepted = False
            reasons.append(f'Confidence level ({conf:.1%}) is below the minimum threshold ({CONF_THRESHOLD:.0%}).')
        if margin < MARGIN_THRESHOLD:
            accepted = False
            reasons.append(f'Probability margin between the top candidates ({margin:.1%}) is too small. The model is uncertain.')
        if MULTI_OBJECT_REJECT and n_comp > MAX_LARGE_COMPONENTS:
            accepted = False
            reasons.append(f'The system detected {n_comp} distinct objects in the image. Please ensure only one character is present.')
            
    return {
        'accepted': accepted,
        'reasons': reasons,
        'label': label,
        'script': script_out,
        'character': char_out,
        'confidence': conf,
        'margin': margin,
        'latency_ms': latency_ms,
        'topk': topk,
        'n_comp': n_comp,
        'img_proc': img,
    }

# ===========================================================================
# Streamlit UI
# ===========================================================================
st.set_page_config(
    page_title='Nusantara Script Classifier',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Custom CSS for fade-in animation and native spacing
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title('Nusantara Script')
    st.write('Identify regional script characters using Classical Machine Learning.')
    
    st.divider()
    st.subheader('Prediction Settings')
    
    script_filter = st.selectbox(
        'Script Context',
        ['All Scripts', 'Bali', 'Jawa', 'Sunda'],
        index=0,
        help='Select a specific script if known to narrow down the prediction space and improve accuracy.'
    )
    
    use_guardrails = st.checkbox(
        'Enable Guardrail System',
        value=True,
        help='Prevents the system from returning predictions on blurry or ambiguous images.'
    )
    
    st.divider()
    st.subheader('Model Evaluation Metrics')
    c1, c2 = st.columns(2)
    c1.metric('Accuracy', f'{CFG.get("accuracy", 0):.2%}')
    c2.metric('Macro F1', f'{CFG.get("macro_f1", 0):.2%}')
    
    st.write("F1-Score per Script:")
    per_script = CFG.get('per_script_f1', {})
    for sc, val in per_script.items():
        st.write(f"{sc}: {val:.1%}")
        st.progress(float(val))
        
    st.divider()
    st.caption('System Information')
    st.caption(f"Total Classes: {N_CLASSES}")
    st.caption(f"Input Dimension: {IMG_SIZE}x{IMG_SIZE}px")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
st.header('Nusantara Script Recognition System')
st.write('This application is designed to automatically read and classify handwritten characters from Balinese, Javanese, and Sundanese scripts.')
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_classify, tab_guide, tab_about = st.tabs(['Image Classification', 'User Guide', 'Technical Details'])

# TAB 1: Image Classification
with tab_classify:
    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        'Upload a handwritten character image',
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif', 'tiff']
    )

    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f'Error reading the image file: {e}')
            st.stop()

        with st.spinner('Processing image and extracting features...'):
            try:
                res = predict_pil(pil_img, script=script_filter, top_k=10, use_guardrails=use_guardrails)
            except Exception as e:
                st.error('Prediction failed. Please ensure the image contains a valid character format.')
                st.stop()

        col1, col2 = st.columns([1, 1], gap='large')
        
        with col1:
            st.subheader('Image Preview')
            img_c1, img_c2 = st.columns(2)
            
            with img_c1:
                st.image(pil_img, use_container_width=True)
                st.caption('Original Image (Input)')
                
            with img_c2:
                st.image(res['img_proc'], clamp=True, use_container_width=True)
                st.caption('After Preprocessing (Binarized & Centered)')

        with col2:
            st.subheader('Classification Result')
            
            if res['accepted']:
                with st.container(border=True):
                    st.write("Identified Script:")
                    st.subheader(res['script'])
                    
                    st.write("Character:")
                    st.title(res['character'])
                    
                    st.metric("Confidence Level", f"{res['confidence']:.2%}")
            else:
                st.error("Prediction Rejected by Guardrail")
                st.write("The model identified ambiguity or formatting issues in the uploaded image.")
                for reason in res['reasons']:
                    st.write(f"- {reason}")
                
                st.write(f"Top candidate before rejection: **{res['label']}** ({res['confidence']:.2%})")

            st.caption(f"Inference computation time: {res['latency_ms']:.2f} ms")
            
        st.divider()
        st.subheader('Top 10 Predictions Distribution')
        
        # Prepare dataframe for clean presentation
        df_topk = pd.DataFrame(res['topk'], columns=['Class Label', 'Probability'])
        df_topk['Script'] = df_topk['Class Label'].apply(lambda x: x.split('_')[0] if '_' in x else x)
        df_topk['Character'] = df_topk['Class Label'].apply(lambda x: x.split('_')[1] if '_' in x else x)
        
        # Reorder columns and format probability
        df_display = df_topk[['Script', 'Character', 'Probability']].copy()
        
        # Display as a native dataframe with a ProgressColumn for visual animation
        st.dataframe(
            df_display,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Confidence Score",
                    help="The prediction confidence probability",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Please upload a script character image using the panel above to begin the classification process.")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: User Guide
with tab_guide:
    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    st.subheader('Usage Instructions')
    st.write('1. Ensure you have an image containing only a single script character (Javanese, Sundanese, or Balinese).')
    st.write('2. Upload the image using the drag-and-drop area in the "Image Classification" tab.')
    st.write('3. Select the specific script type from the sidebar menu if known. This narrows the search space and improves accuracy (Optional).')
    st.write('4. Wait a few moments while the system performs binarization, feature extraction, and classification.')
    
    c_tips1, c_tips2 = st.columns(2)
    with c_tips1:
        st.subheader('Optimal Image Criteria')
        st.write('- Clean background with bright colors (white/cream).')
        st.write('- Solid and dark character ink (black/dark blue).')
        st.write('- Captured from a straight, overhead perspective.')
        st.write('- Even lighting without dark shadows covering the character.')
    
    with c_tips2:
        st.subheader('Criteria to Avoid')
        st.write('- Images containing multiple characters, full words, or sentences.')
        st.write('- Lined, checkered, or heavily textured backgrounds.')
        st.write('- Blurry images or extremely low-resolution photos.')
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: Technical Details
with tab_about:
    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    st.subheader('System Pipeline Architecture')
    st.write('This application is built using a pure Classical Machine Learning approach (without Deep Learning architectures), demonstrating computational efficiency for Computer Vision tasks.')
    
    st.write('**Processing Stages:**')
    st.write('1. **Image Preprocessing:** CLAHE, Non-Local Means Denoising, Adaptive Otsu Binarization, area cropping, and image centering.')
    st.write('2. **Manual Feature Extraction:** Histogram of Oriented Gradients (HOG), Local Binary Pattern (LBP), Gabor Filters, Zoning, and Shape/Morphology calculation. The raw feature extraction produces 6,382 dimensions.')
    st.write('3. **Dimensionality Reduction:** Utilizing Principal Component Analysis (PCA) to retain 98% of the variance, reducing the dimensions to approximately 2,479 features.')
    st.write('4. **Ensemble Classification:** Combining multiple high-performing models (such as LightGBM and CatBoost) using a Weighted Soft Vote methodology.')
    
    st.divider()
    
    st.subheader('Development Context')
    st.write('This project is part of the final assignment for the COMP6577001 - Machine Learning course at Bina Nusantara University. The models were trained implementing strict balancing and validation methods, resulting in a minimal gap (0.0002) between testing and validation performance, confirming the absence of overfitting.')
    st.markdown('</div>', unsafe_allow_html=True)