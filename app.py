import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
# ----------------------------------
# Page Config (Mobile First)
# ----------------------------------
st.set_page_config(page_title="Parkinson Tester", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Open+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
        scroll-behavior: smooth;
    }

    .stApp {
        background-color: white;
    }

    header, footer {visibility: hidden;}

    h1, h2, h3, h4, h5, h6 {
        color: #4A4A4A !important;
        font-weight: 700 !important;
    }

    div[data-testid="stRadio"] label p {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    .stFileUploader label {
        color: #333333 !important;
    }
    div[class*="stMarkdown"] p {
        color: #333333 !important;
    }


    /* Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        color: #555;
        font-weight: 600;
        margin-bottom: 20px;position: relative;
        z-index: 100;
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
        
    }

    /* Hero Text */
    .hero-content {
        text-align: center;
        padding-top: 30px;
        padding-bottom: 10px;
        display: flex;           
        flex-direction: column;
        align-items: center;
        justify-content: center;
        
         
    }
    .hero-title {
        color: #4A4A4A;
        font-size: 4rem; 
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 20px;
        text-align: center;
    }
    .hero-sub {
        color: #757575;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 40px;
        line-height: 1.6;
        text-align: center;
    }

    /* ปุ่มกดแบบ Link (CTA) */
    .cta-button {
        background-color: #885D95; 
        color: white !important;
        padding: 18px 60px;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0 4px 15px rgba(136, 93, 149, 0.4);
        transition: transform 0.2s;
        display: inline-block;
        margin-bottom: 30px;
        display: block !important;    
        width: fit-content;          
        margin-left: auto !important; 
        margin-right: auto !important;
        margin-bottom: 30px;
    }
    .cta-button:hover {
        transform: translateY(-3px);
        background-color: #724C7F;
    }

    /* Test Cards */
    .input-card {
        background-color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #eee;
        height: 100%;
    }

    /* Info Section */
    .info-section {
        background-color: white;
        padding: 60px 20px;
        margin-top: 50px;
        border-radius: 40px 40px 0 0;
    }
    
    div.stButton > button {
        width: 100%;
        border-radius: 30px;
        height: 50px;
        font-size: 18px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: black !important;   
        border: 10px solid black !important;   
        border-radius: 20px !important;         
        box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important; 
        padding: 25px !important;               
        margin-bottom: 30px !important;        
    }

</style>
""", unsafe_allow_html=True)

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
st.markdown('<div class="hero-bg-box"></div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="hero-content">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">“Early detection changes everything.”</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ใช้ AI ตรวจคัดกรองพาร์กินสันเบื้องต้น แม่นยำ รวดเร็ว และรู้ผลทันที<br>เพียงแค่วาดเส้น หรืออัปโหลดรูปภาพ</div>', unsafe_allow_html=True)
    st.markdown('<a href="#test_area" class="cta-button">เริ่มทำแบบทดสอบ ➝</a>', unsafe_allow_html=True)
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------
# Load Spiral Model
# ----------------------------------
@st.cache_resource
def load_spiral_model():
    return tf.keras.models.load_model("(Test_naja)effnet_parkinson_model.keras")

spiral_model = load_spiral_model()

# ----------------------------------
# Preprocess (256x256 ตามโมเดล)
# ----------------------------------
def preprocess(img):
    img = np.array(img.convert("RGB"))
    img = cv2.resize(img, (256, 256))   # ✅ สำคัญมาก
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================================================
# =====================  BOX 1 : SPIRAL  ==================
# =========================================================

st.markdown('<div id="test_area"></div>', unsafe_allow_html=True) 

# สร้าง Columns: ซ้าย-ขวา ว่างไว้เพื่อให้เนื้อหาอยู่ตรงกลาง (c2)
c1, c2, c3 = st.columns([1, 2, 1]) 

with c2: 
    # =====================  BOX 1 : SPIRAL  ==================
    # ✅ กล่องที่ 1
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
            # Draw Mode
            dc1, dc2, dc3 = st.columns([0.1, 5, 0.1])
            with dc2:
                spiral_canvas = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=6,
                    stroke_color="black",
                    background_color="white",
                    height=300,
                    width=500,     
                    drawing_mode="freedraw",
                    key="spiral_draw"
                )
            if spiral_canvas.image_data is not None:
                spiral_image = Image.fromarray(spiral_canvas.image_data.astype("uint8")).convert("RGB")
        
        st.markdown("<br>", unsafe_allow_html=True)
        spiral_result_box = st.empty()

    # =====================  BOX 2 : WAVE  =====================
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
            wc1, wc2, wc3 = st.columns([0.1, 5, 0.1])
            with wc2:
                wave_canvas = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=6,
                    stroke_color="black",
                    background_color="white",
                    height=300,
                    width=500,
                    drawing_mode="freedraw",
                    key="wave_draw"
                )
            if wave_canvas.image_data is not None:
                wave_image = Image.fromarray(wave_canvas.image_data.astype("uint8")).convert("RGB")

        st.markdown("<br>", unsafe_allow_html=True)
        wave_result_box = st.empty()

    # =====================  BUTTON  ==================
    
    if st.button("🔍 ประมวลผลทั้งหมด", use_container_width=True):
        
        # Spiral Check
        if spiral_image is not None:
            try:
                input_tensor = preprocess(spiral_image)
                pred = spiral_model.predict(input_tensor)[0][0]
                if pred > 0.5:
                    spiral_result_box.error(f"🌀 Spiral : เสี่ยง Parkinson ({pred:.3f})")
                else:
                    spiral_result_box.success(f"🌀 Spiral : ปกติ ({pred:.3f})")
            except Exception as e:
                spiral_result_box.error(f"Error: {e}")
        else:
            spiral_result_box.warning("🌀 Spiral : ยังไม่ได้ใส่ภาพ")

        # Wave Check
        if wave_image is not None:
            wave_result_box.info("🌊 Wave : มีภาพแล้ว รอโมเดล")
        else:
            wave_result_box.warning("🌊 Wave : ยังไม่ได้ใส่ภาพ")
