import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ---------- CONFIG ----------
st.set_page_config(page_title="Qynapse AI", layout="wide")

# ---------- PREMIUM UI ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.big-title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
}
.sub-title {
    text-align:center;
    font-size:22px;
    color:#bbb;
}
.stButton button {
    border-radius:12px;
    height:55px;
    transition:0.3s;
}
.stButton button:hover {
    transform:scale(1.05);
    background:linear-gradient(to right,#00c6ff,#0072ff);
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------- SAFE MODEL LOAD ----------
@st.cache_resource
def load_my_model():
    try:
        model = load_model("brain_tumor_model.h5", compile=False)
        return model
    except Exception as e:
        return None

model = load_my_model()
IMG_SIZE = 224

# ---------- API ----------
API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---------- SESSION ----------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- SIDEBAR ----------
st.sidebar.title("🤖 AI Assistant")

if st.session_state.user:
    st.sidebar.markdown(f"👤 **{st.session_state.user}**")

st.sidebar.markdown("---")

# ---------- CHATBOT ----------
def get_ai(messages):
    if not API_KEY:
        return "⚠️ API key missing"
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type":"application/json"
            },
            json={
                "model":"openai/gpt-4o-mini",
                "messages":messages
            },
            timeout=10
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ AI unavailable"

for msg in st.session_state.chat_history[-5:]:
    st.sidebar.write(("🧑 " if msg["role"]=="user" else "🤖 ") + msg["content"])

with st.sidebar.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask medical doubt...")
    send = st.form_submit_button("Send")

    if send and user_input.strip():
        st.session_state.chat_history.append({"role":"user","content":user_input})
        reply = get_ai(st.session_state.chat_history)
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

# ---------- LANDING ----------
if st.session_state.page == "landing":

    st.markdown("<div class='big-title'>🧠 QYNAPSE PRESENTS</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>AI Healthcare Platform</div>", unsafe_allow_html=True)
    st.markdown("---")

    col = st.columns([1,2,1])[1]
    c1, c2 = col.columns(2)

    if c1.button("🔐 Login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()

    if c2.button("⭐ Review", use_container_width=True):
        st.success("Thanks for visiting!")

# ---------- LOGIN ----------
elif st.session_state.page == "login":

    st.title("🔐 Login")

    if st.button("⬅ Back"):
        st.session_state.page = "landing"
        st.rerun()

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user and pwd:
            st.session_state.user = user
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Enter details")

# ---------- DASHBOARD ----------
elif st.session_state.page == "dashboard":

    st.title("🧠 Select Analysis")

    if st.button("⬅ Logout"):
        st.session_state.user = None
        st.session_state.page = "landing"
        st.rerun()

    diseases = [
        "🧠 Brain Tumor","👁️ Eyelid Tumor","🩺 Sickle Cell",
        "🧬 Cancer","🧫 Pancreas","🦴 Fracture","🍬 Diabetes"
    ]

    cols = st.columns(3)

    for i, d in enumerate(diseases):
        if cols[i%3].button(d, use_container_width=True):
            if "Brain" in d:
                st.session_state.page = "brain"
            else:
                st.session_state.page = "coming"
            st.rerun()

# ---------- COMING ----------
elif st.session_state.page == "coming":

    st.title("🚧 Research in Progress")

    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()

# ---------- BRAIN MODULE ----------
elif st.session_state.page == "brain":

    st.title("🧠 Brain Tumor Detection")

    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()

    # 🔥 MODEL SAFETY CHECK
    if model is None:
        st.error("❌ Model failed to load.")
        st.info("👉 Convert .h5 → .keras for compatibility")
        st.stop()

    file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

    def preprocess(img):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        return np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    def gradcam(img_array):
        try:
            last_conv = None
            for layer in reversed(model.layers):
                if "conv" in layer.name.lower():
                    last_conv = layer
                    break

            if last_conv is None:
                return None

            grad_model = tf.keras.models.Model(
                [model.inputs],
                [last_conv.output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
            conv_outputs = conv_outputs[0]

            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = np.maximum(heatmap,0) / (np.max(heatmap)+1e-8)

            return heatmap
        except:
            return None

    def generate_pdf(result, conf):
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("Brain Tumor Detection Report", styles['Title']),
            Spacer(1,20),
            Paragraph(f"Result: {result}", styles['Normal']),
            Paragraph(f"Confidence: {conf:.2f}%", styles['Normal'])
        ]

        doc.build(content)
        return path

    if file:
        try:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            col1, col2 = st.columns(2)

            with col1:
                st.image(img, use_container_width=True)

            with col2:
                with st.spinner("Analyzing MRI..."):
                    img_input = preprocess(img)
                    prediction = float(model.predict(img_input)[0][0])
                    confidence = prediction * 100

                if prediction > 0.5:
                    result = "Tumor Detected"
                    st.error(f"{result} ❌ ({confidence:.2f}%)")
                else:
                    result = "No Tumor"
                    st.success(f"{result} ✅ ({100-confidence:.2f}%)")

            heatmap = gradcam(img_input)
            if heatmap is not None:
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed = heatmap * 0.4 + img
                st.image(superimposed.astype("uint8"),
                         caption="🔥 AI Focus Area",
                         use_container_width=True)

            pdf_path = generate_pdf(result, confidence)

            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download Report", f, "report.pdf")

        except Exception as e:
            st.error(f"❌ Processing error: {e}")
