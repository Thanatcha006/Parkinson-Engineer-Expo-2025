import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os
import base64

# ----------------------------------
# Page Config (Mobile First)
# ----------------------------------
st.set_page_config(page_title="Parkinson Tester", layout="wide", initial_sidebar_state="collapsed")


# ----------------------------------
# CSS Styles
# ----------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Open+Sans:wght@400;600;700&display=swap');

    /* Global Settings */
    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
        scroll-behavior: smooth;
        color: #333333;
    }


    /* ซ่อน Header/Footer ของระบบ */
    header, footer {visibility: hidden;}

    /* ปรับสีหัวข้อทั้งหมด */
    h1, h2, h3, h4, h5, h6 {
        color: #4A4A4A !important;
        font-weight: 700 !important;
    }

    /* ปรับสีตัวหนังสือใน Input */
    div[data-testid="stRadio"] label p {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    .stFileUploader label { color: #333333 !important; }
    div[class*="stMarkdown"] p { color: #333333 !important; }

    /* ----------------------------------------------------------- */
    /* ✅ HERO & NAVBAR */
    /* ----------------------------------------------------------- */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 40px;
        background-color: white; 
        border-bottom: 1px solid #eee; 
        color: #555;
        font-weight: 600;
        margin-top: -50px; 
        margin-left: -5rem;
        margin-right: -5rem;
        padding-left: 5rem;
        padding-right: 5rem;
        height: 80px;
        position: relative;
        z-index: 100;
    }
    .hero-purple-container {
        background-color: #885D95;
        
        /* ขยายเต็มจอซ้ายขวา */
        margin-left: -5rem; 
        margin-right: -5rem;
        padding-left: 5rem; 
        padding-right: 5rem;
        
        /* ระยะห่างภายใน */
        padding-top: 60px; 
        padding-bottom: 80px; /* ยืดด้านล่างให้คลุมปุ่ม */
        margin-bottom: 40px; 
        
        /* จัดเนื้อหาตรงกลาง */
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        
        /* เส้นขอบล่าง */
        border-bottom: 1px solid #E0D0E8;
    }
    
    .hero-title {
        color: white;
        font-size: clamp(2.2rem, 4vw, 2.5rem); 
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 20px;
        text-align: center;
    }
    .hero-sub {
        color: #f0f0f0;
        font-size: clamp(1.05rem, 1.5vw, 1.3rem); /* แก้ไขขนาด font ขั้นต่ำ */
        font-weight: 300;
        margin-bottom: 40px;
        line-height: 1.6;
        text-align: center;
    }

    .hero-img-responsive {
        width: 100%;             /* กว้างเต็มพื้นที่ container 100% */
        height: auto;            /* สูงอัตโนมัติ รักษาทรงภาพ */
        margin-top: 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3); /* เงาลอยๆ */
        object-fit: cover;
    }

    .cta-button {
        background-color: white; 
        color: #885D95 !important;
        padding: 18px 60px;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0 4px 15px rgba(136, 93, 149, 0.4);
        transition: transform 0.2s;
        display: block !important;     
        width: fit-content;           
        margin-left: auto !important; 
        margin-right: auto !important;
        margin-bottom: 30px;
    }
    .cta-button:hover {
        transform: translateY(-3px);
        background-color: #f8f8f8;
    }

    /* ----------------------------------------------------------- */
    /* ✅ CARD STYLE: เป้าหมายคือกล่องที่มี border=True */
    /* ----------------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 2px solid #885D95 !important;  /* เส้นขอบสีม่วง หนา 2px */
        border-radius: 20px !important;
        padding: 25px !important;
        box-shadow: 0 4px 15px rgba(136, 93, 149, 0.2) !important; /* เงาสีม่วงจางๆ */
        margin-bottom: 30px !important;
    }
    
    /* แก้ไขหัวข้อ (H3) ภายในการ์ดให้เป็นสีม่วงเข้ม */
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #4A4A4A !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
    }
    
    div.stButton > button {
        width: 100%;
        border-radius: 30px;
        height: 50px;
        font-size: 18px;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# UI Elements
# ----------------------------------

# Navbar
st.markdown("""
<div class="navbar">
    <div style="font-size: 1.3rem; color: #885D95; font-weight:700;">🧬 Parkinson AI</div>
    <div>
        <a href="#info_section" style="text-decoration:none; color:#555; margin-right:20px;">เกี่ยวกับโรค</a>
        <a href="#test_area" style="text-decoration:none; color:#885D95; font-weight:600;">เริ่มใช้งาน</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Content
st.markdown("""
<div class="hero-purple-container">
    <div class="hero-title">“Early detection changes everything.”</div>
    <div class="hero-sub">ใช้ AI ตรวจคัดกรองพาร์กินสันเบื้องต้น แม่นยำ รวดเร็ว และรู้ผลทันที<br>เพียงแค่วาดเส้น หรืออัปโหลดรูปภาพ</div>
    <a href="#test_area" class="cta-button">เริ่มทำแบบทดสอบ ➝</a>
</div>
""", unsafe_allow_html=True)
st.image("parkinson cover.png", width=1000, height=1500)


# ----------------------------------
# Load Model (Add error handling)
# ----------------------------------
@st.cache_resource
def load_spiral_model():
    if os.path.exists("(Test_naja)effnet_parkinson_model.keras"):
        return tf.keras.models.load_model("(Test_naja)effnet_parkinson_model.keras")
    return None

spiral_model = load_spiral_model()

# ----------------------------------
# Preprocess
# ----------------------------------
def preprocess(img):
    img = np.array(img.convert("RGB"))
    img = cv2.resize(img, (256, 256))   
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================================================
# =====================  TEST AREA  =======================
# =========================================================
# จุด Anchor
st.markdown('<div id="test_area" style="padding-top: 50px;"></div>', unsafe_allow_html=True) 

# Layout หลัก
c1, c2, c3 = st.columns([1, 2, 1]) 

with c2: 
    # =====================  การ์ด 1 : SPIRAL  ==================
    # border=True จะไปเรียก CSS stVerticalBlockBorderWrapper ที่เราเขียนไว้
    with st.container(border=True): 
        st.subheader("🌀 Spiral")
        
        spiral_mode = st.radio("เลือกวิธีใส่ภาพ (Spiral)", ["Upload", "Draw"], horizontal=True, key="spiral_mode")
        
        spiral_image = None
        if spiral_mode == "Upload":
            spiral_file = st.file_uploader("อัปโหลด Spiral", type=["png", "jpg", "jpeg"], key="spiral_upload")
            if spiral_file:
                spiral_image = Image.open(spiral_file).convert("RGB")
                st.image(spiral_image, caption="Spiral Preview", use_container_width=True)
        else:
            # Draw Mode - จัดกึ่งกลาง
            dc1, dc2, dc3 = st.columns([0.05, 1, 0.05])
            with dc2:
                spiral_canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=6,
                    stroke_color="black",
                    background_color="#ffffff",
                    height=300,
                    width=450,     
                    drawing_mode="freedraw",
                    key="spiral_draw"
            )
            if spiral_canvas.image_data is not None:
                spiral_image = Image.fromarray(spiral_canvas.image_data.astype("uint8")).convert("RGB")
        
        st.markdown("<br>", unsafe_allow_html=True)
        spiral_result_box = st.empty()


    # =================================================
    # 🌊 การ์ดใบที่ 2 : WAVE
    # =================================================
    with st.container(border=True): 
        st.subheader("🌊 Wave")

        wave_mode = st.radio("เลือกวิธีใส่ภาพ (Wave)", ["Upload", "Draw"], horizontal=True, key="wave_mode")
        
        wave_image = None
        if wave_mode == "Upload":
            wave_file = st.file_uploader("อัปโหลด Wave", type=["png", "jpg", "jpeg"], key="wave_upload")
            if wave_file:
                wave_image = Image.open(wave_file).convert("RGB")
                st.image(wave_image, caption="Wave Preview", use_container_width=True)
        else:
            # Draw Mode
            wc1, wc2, wc3 = st.columns([0.05, 1, 0.05])
            with wc2:
                wave_canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=6,
                    stroke_color="black",
                    background_color="#ffffff",
                    height=300,
                    width=450,
                    drawing_mode="freedraw",
                    key="wave_draw"
                )
            if wave_canvas.image_data is not None:
                wave_image = Image.fromarray(wave_canvas.image_data.astype("uint8")).convert("RGB")

        st.markdown("<br>", unsafe_allow_html=True)
        wave_result_box = st.empty()


    # =================================================
    # ปุ่มประมวลผล
    # =================================================
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 ประมวลผลทั้งหมด", use_container_width=True):
        
        # Spiral
        if spiral_image is not None and spiral_model is not None:
            try:
                input_tensor = preprocess(spiral_image)
                pred = spiral_model.predict(input_tensor)[0][0]
                if pred > 0.5:
                    spiral_result_box.error(f"🌀 Spiral : เสี่ยง Parkinson ({pred:.3f})")
                else:
                    spiral_result_box.success(f"🌀 Spiral : ปกติ ({pred:.3f})")
            except Exception as e:
                spiral_result_box.error(f"Error: {e}")
        elif spiral_image is None:
            spiral_result_box.warning("🌀 Spiral : ยังไม่ได้ใส่ภาพ")
        elif spiral_model is None:
            spiral_result_box.error("❌ ไม่พบไฟล์โมเดล")

        # Wave
        if wave_image is not None:
            wave_result_box.info("🌊 Wave : มีภาพแล้ว (รอโมเดล)")
        else:
            wave_result_box.warning("🌊 Wave : ยังไม่ได้ใส่ภาพ") รูปไม่ขึ้นทั้งที่กรอกถูกแล้ว
