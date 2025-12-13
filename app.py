import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Parkinson AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;700&display=swap');

header, footer {visibility: hidden;}

html, body, [class*="css"] {
    font-family: 'Kanit', sans-serif;
}

/* Remove default padding */
.block-container {
    padding: 0rem !important;
    max-width: 100% !important;
}

/* Navbar */
.navbar {
    position: fixed;
    top: 0;
    width: 100%;
    height: 70px;
    background: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 40px;
    z-index: 1000;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Hero */
.hero {
    background-color: #FFDFD0;
    min-height: 100vh;
    padding-top: 100px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    text-align: center;
    overflow: hidden;
}

.hero h1 {
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.hero p {
    font-size: 1.2rem;
    color: #555;
    margin-bottom: 30px;
}

.hero img {
    width: 100%;
    max-width: 1000px;
}

/* Button */
.cta {
    background-color: #8c7ae6;
    color: white;
    padding: 14px 45px;
    border-radius: 50px;
    text-decoration: none;
    font-size: 1.2rem;
    font-weight: 600;
    display: inline-block;
}

/* App container */
.app {
    max-width: 1000px;
    margin: auto;
    padding: 60px 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Navbar
# --------------------------------------------------
st.markdown("""
<div class="navbar">
    <div style="font-size:1.4rem; font-weight:700; color:#885D95;">
        🧬 Parkinson AI
    </div>
    <a href="#app" style="text-decoration:none; font-weight:600; color:#885D95;">
        เริ่มใช้งาน
    </a>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Hero Section
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div>
        <h1>Early detection changes everything</h1>
        <p>
            ใช้ AI ตรวจคัดกรองพาร์กินสันเบื้องต้น<br>
            เพียงวาดเส้นหรืออัปโหลดภาพ
        </p>
        <a href="#app" class="cta">เริ่มทำแบบทดสอบ →</a>
    </div>
</div>
""", unsafe_allow_html=True)


st.image("parkinson_cover.png", use_container_width=True)

# Anchor
st.markdown('<div id="app"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# App Section
# --------------------------------------------------
with st.container():
    st.markdown('<div class="app">', unsafe_allow_html=True)

    # ---------------- Model ----------------
    @st.cache_resource
    def load_model():
        if os.path.exists("(Test_naja)effnet_parkinson_model.keras"):
            return tf.keras.models.load_model(
                "(Test_naja)effnet_parkinson_model.keras"
            )
        return None

    model = load_model()

    if model is None:
        st.warning("⚠️ ไม่พบไฟล์โมเดล ระบบจะทำงานเฉพาะส่วนวาดภาพ")

    def preprocess(img):
        img = np.array(img.convert("RGB"))
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        return np.expand_dims(img, axis=0)

    # ---------------- Spiral ----------------
    st.subheader("1. 🌀 Spiral (ขดลวด)")
    mode = st.radio(
        "เลือกวิธีใส่ภาพ",
        ["Upload", "Draw"],
        horizontal=True
    )

    spiral_image = None

    if mode == "Upload":
        file = st.file_uploader(
            "อัปโหลดภาพ Spiral",
            type=["png", "jpg", "jpeg"]
        )
        if file:
            spiral_image = Image.open(file)
            st.image(spiral_image, width=300)
    else:
        canvas = st_canvas(
            stroke_width=6,
            stroke_color="black",
            background_color="#f5f5f5",
            width=500,
            height=300,
            drawing_mode="freedraw",
            key="spiral"
        )
        if canvas.image_data is not None:
            if np.sum(canvas.image_data) > 0:
                spiral_image = Image.fromarray(
                    canvas.image_data.astype("uint8")
                )

    result_box = st.empty()

    if st.button("🔍 วิเคราะห์ Spiral"):
        if spiral_image is None:
            result_box.warning("กรุณาใส่ภาพก่อน")
        elif model is None:
            result_box.error("ไม่พบโมเดล")
        else:
            x = preprocess(spiral_image)
            pred = model.predict(x)[0][0]
            if pred > 0.5:
                result_box.error(
                    f"มีความเสี่ยง Parkinson (Confidence {pred:.2f})"
                )
            else:
                result_box.success(
                    f"ปกติ (Confidence {pred:.2f})"
                )

    st.markdown('</div>', unsafe_allow_html=True)
