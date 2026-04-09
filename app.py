import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepDent AI",
    page_icon="🦷",
    layout="centered",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0b0f1a;
    color: #e8eaf0;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Animated top accent bar */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2563eb, #7c3aed, #2563eb);
    background-size: 200% 100%;
    animation: slide 3s linear infinite;
    z-index: 9999;
}
@keyframes slide {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.4);
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    color: #f1f3f9;
    line-height: 1.1;
    margin: 0 0 0.8rem;
}
.hero-title span {
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #8b95b0;
    font-size: 1rem;
    font-weight: 300;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Stat row */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 2rem 0;
}
.stat-item { text-align: center; }
.stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #60a5fa;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Divider */
.custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.15), transparent);
    margin: 1.5rem 0;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.8) !important;
    border: 1.5px dashed rgba(37,99,235,0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(37,99,235,0.65) !important;
}

/* Result cards */
.result-card {
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin: 1.5rem 0 0.5rem;
    border: 1.5px solid;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}
.result-card.decay {
    background: rgba(239,68,68,0.08);
    border-color: rgba(239,68,68,0.3);
}
.result-card.healthy {
    background: rgba(16,185,129,0.08);
    border-color: rgba(16,185,129,0.3);
}
.result-icon { font-size: 2rem; line-height: 1; }
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    margin: 0 0 0.3rem;
}
.decay  .result-title { color: #f87171; }
.healthy .result-title { color: #34d399; }
.result-desc {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 0;
}

/* Confidence */
.conf-label {
    font-size: 0.78rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.conf-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #f1f3f9;
    margin-bottom: 0.6rem;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    border-radius: 100px !important;
    background: rgba(30,41,59,0.8) !important;
}
[data-testid="stProgress"] > div > div > div {
    border-radius: 100px !important;
}

/* Disclaimer pills */
.disclaimer {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #fbbf24;
    font-size: 0.83rem;
    line-height: 1.5;
    margin-top: 1rem;
}
.disclaimer.info {
    background: rgba(99,102,241,0.08);
    border-color: rgba(99,102,241,0.25);
    color: #a5b4fc;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    color: #334155;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}
.footer span { color: #475569; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained .h5 model once into RAM and cache it."""
    return tf.keras.models.load_model("deepdent_model.h5")

model = load_model()


# ── PREPROCESS ─────────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert to RGB → resize to 224×224 → normalize to [0,1] → add batch dim.
    Must match the exact pipeline used during training.
    """
    img = pil_image.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ── HERO SECTION ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🦷 AI-Powered Diagnostics</div>
    <h1 class="hero-title">Deep<span>Dent</span> AI</h1>
    <p class="hero-sub">
        Upload a dental OPG X-ray and let the model screen
        for signs of tooth decay in seconds.
    </p>
</div>

<div class="stat-row">
    <div class="stat-item">
        <div class="stat-number">100%</div>
        <div class="stat-label">Val Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">MNV2</div>
        <div class="stat-label">Architecture</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">224px</div>
        <div class="stat-label">Input Size</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">14 MB</div>
        <div class="stat-label">Model Size</div>
    </div>
</div>

<hr class="custom-divider">
""", unsafe_allow_html=True)


# ── FILE UPLOADER ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a Dental X-ray",
    type=["png", "jpg", "jpeg"],   # Streamlit enforces this at the dialog level
    help="Accepted formats: JPG · PNG · JPEG  •  Max 200 MB",
)


# ── MAIN INFERENCE BLOCK ───────────────────────────────────────────────────────
if uploaded_file is not None:

    # Second-layer format guard (in case browser bypasses the filter)
    fname = uploaded_file.name.lower()
    if not (fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg")):
        st.error("❌ Invalid format. Please upload a PNG, JPG, or JPEG file.")
        st.stop()

    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("❌ Could not read the image. Please try a different file.")
        st.stop()

    # Show the uploaded image
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Run the model
    with st.spinner("Running inference…"):
        processed  = preprocess_image(image)
        prediction = model.predict(processed)      # shape: (1, 1) — sigmoid output
        confidence = float(prediction[0][0])       # value ∈ (0, 1)

    # ── Interpret & display result ──────────────────────────────────────────────
    if confidence > 0.5:
        # Decay detected — confidence is the decay probability
        pct = confidence * 100
        st.markdown(f"""
        <div class="result-card decay">
            <div class="result-icon">🚨</div>
            <div>
                <p class="result-title">Potential Decay Detected</p>
                <p class="result-desc">
                    The model has identified patterns consistent with dental caries.
                    A clinical examination by a qualified dentist is strongly recommended.
                </p>
            </div>
        </div>
        <div class="conf-label">AI Confidence</div>
        <div class="conf-value">{pct:.2f}%</div>
        """, unsafe_allow_html=True)
        st.progress(confidence)
        st.markdown("""
        <div class="disclaimer">
            ⚠️ This tool is for <strong>screening purposes only</strong>.
            Always consult a qualified dentist for a clinical diagnosis.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Healthy — flip confidence so it represents "healthy" certainty
        pct = (1 - confidence) * 100
        st.markdown(f"""
        <div class="result-card healthy">
            <div class="result-icon">✅</div>
            <div>
                <p class="result-title">No Decay Detected</p>
                <p class="result-desc">
                    Teeth appear structurally healthy in the uploaded X-ray.
                    Keep up the good oral hygiene!
                </p>
            </div>
        </div>
        <div class="conf-label">AI Confidence</div>
        <div class="conf-value">{pct:.2f}%</div>
        """, unsafe_allow_html=True)
        st.progress(1 - confidence)
        st.markdown("""
        <div class="disclaimer info">
            ℹ️ Regular dental check-ups are still recommended even when no decay is detected.
        </div>
        """, unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    DeepDent AI &nbsp;·&nbsp; Powered by <span>MobileNetV2</span>
    &nbsp;·&nbsp; Built with <span>TensorFlow + Streamlit</span>
    &nbsp;·&nbsp; For screening purposes only
</div>
""", unsafe_allow_html=True)