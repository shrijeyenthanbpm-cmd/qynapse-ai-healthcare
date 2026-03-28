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

# ---------- STYLE ----------
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
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL (SAFE + CACHED) ----------
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- SIDEBAR ----------
st.sidebar.title("🤖 AI Assistant")

# ---------- CHATBOT ----------
for msg in st.session_state.chat_history[-5:]:
    st.sidebar.write(("🧑 " if msg["role"]=="user" else "🤖 ") + msg["content"])

with st.sidebar.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask medical doubt...")
    send = st.form_submit_button("Send")

    if send and user_input.strip():

        st.session_state.chat_history.append({"role":"user","content":user_input})

        def get_ai(messages):
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
                    }
                )
                return r.json()["choices"][0]["message"]["content"]
            except:
                return "⚠️ AI unavailable"

        reply = get_ai(st.session_state.chat_history)
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

# ---------- LANDING ----------
if st.session_state.page == "landing":

    st.markdown("<div class='big-title'>🧠 QYNAPSE PRESENTS</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>AI Healthcare Platform</div>", unsafe_allow_html=True)

    col = st.columns([1,2,1])[1]
    c1, c2 = col.columns(2)

    if c1.button("🔐 Login"):
        st.session_state.page = "dashboard"
        st.rerun()

    if c2.button("⭐ Review"):
        st.success("Thanks for visiting!")

# ---------- DASHBOARD ----------
elif st.session_state.page == "dashboard":

    st.title("🧠 Select Analysis")

    if st.button("⬅ Logout"):
        st.session_state.page = "landing"
        st.rerun()

    if st.button("🧠 Brain Tumor Detection"):
        st.session_state.page = "brain"
        st.rerun()

# ---------- BRAIN MODULE ----------
elif st.session_state.page == "brain":

    st.title("🧠 Brain Tumor Detection")

    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()

    # ❌ MODEL ERROR HANDLING
    if model is None:
        st.error("❌ Model failed to load. Fix model compatibility.")
        st.info("👉 Convert your .h5 model to .keras format")
        st.stop()

    file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

    def preprocess(img):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        return np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    def generate_pdf(result, conf):
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(path, pagesize=letter)
        style = getSampleStyleSheet()

        content = [
            Paragraph("Brain Tumor Report", style['Title']),
            Spacer(1,20),
            Paragraph(f"Result: {result}", style['Normal']),
            Paragraph(f"Confidence: {conf:.2f}%", style['Normal'])
        ]
        doc.build(content)
        return path

    if file:

        bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)

        st.image(img, caption="Uploaded MRI", use_container_width=True)

        with st.spinner("Analyzing..."):
            try:
                img_in = preprocess(img)
                pred = float(model.predict(img_in)[0][0])
                conf = pred * 100

                if pred > 0.5:
                    result = "Tumor Detected"
                    st.error(f"{result} ❌ ({conf:.2f}%)")
                else:
                    result = "No Tumor"
                    st.success(f"{result} ✅ ({100-conf:.2f}%)")

                # PDF DOWNLOAD
                pdf_path = generate_pdf(result, conf)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download Report", f, "report.pdf")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
