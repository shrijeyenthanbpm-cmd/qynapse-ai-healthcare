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

# ---------- STYLE (PREMIUM UI) ----------
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
.card {
    padding:20px;
    border-radius:15px;
    background:rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    margin:10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = load_model("brain_tumor_model.h5")
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
else:
    st.sidebar.write("Not logged in")

st.sidebar.markdown("---")

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
    st.markdown("---")

    col = st.columns([1,2,1])[1]
    c1, c2 = col.columns(2)

    if c1.button("🔐 Login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()

    if c2.button("⭐ Review", use_container_width=True):
        st.success("Thanks for visiting!")

    st.markdown("<p style='text-align:center;'>Developed by Shrijeyenthan B</p>", unsafe_allow_html=True)

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

    st.markdown("---")

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
    st.info("This module will be updated soon.")

    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()

# ---------- BRAIN MODULE ----------
elif st.session_state.page == "brain":

    st.title("🧠 Brain Tumor Detection")

    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()

    file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

    def gradcam(img_array):
        last = model.get_layer("Conv_1")
        grad_model = tf.keras.models.Model([model.inputs],[last.output,model.output])

        with tf.GradientTape() as tape:
            conv, pred = grad_model(img_array)
            loss = pred[:,0]

        grads = tape.gradient(loss, conv)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv = conv[0]

        heatmap = conv @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap,0)/ (np.max(heatmap)+1e-8)
        return heatmap

    def pdf(result, conf):
        path = tempfile.NamedTemporaryFile(delete=False,suffix=".pdf").name
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
        bytes_data = np.asarray(bytearray(file.read()),dtype=np.uint8)
        img = cv2.imdecode(bytes_data,1)

        col1,col2 = st.columns(2)

        with col1:
            st.image(img,use_container_width=True)

        with col2:
            with st.spinner("Analyzing..."):
                img_r = cv2.resize(img,(IMG_SIZE,IMG_SIZE))/255.0
                img_in = np.reshape(img_r,(1,IMG_SIZE,IMG_SIZE,3))

                pred = model.predict(img_in)[0][0]
                conf = pred*100

            if pred>0.5:
                result="Tumor Detected"
                st.error(f"{result} ❌ ({conf:.2f}%)")
            else:
                result="No Tumor"
                st.success(f"{result} ✅ ({100-conf:.2f}%)")

        try:
            heat = gradcam(img_in)
            heat = cv2.resize(heat,(img.shape[1],img.shape[0]))
            heat = np.uint8(255*heat)
            heat = cv2.applyColorMap(heat,cv2.COLORMAP_JET)

            super = heat*0.4 + img
            st.image(super.astype("uint8"), caption="🔥 AI Focus Area", use_container_width=True)
        except:
            st.warning("GradCAM unavailable")

        path = pdf(result,conf)
        with open(path,"rb") as f:
            st.download_button("📄 Download Report",f,"report.pdf")