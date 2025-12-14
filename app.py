import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os
from datetime import datetime

# ----------------------------------
# 1. Page Config
# ----------------------------------
st.set_page_config(page_title="Parkinson Tester", layout="wide", initial_sidebar_state="collapsed")

if "consent_accepted" not in st.session_state:
    st.session_state.consent_accepted = False

# ----------------------------------
# CSS Styles 
# ----------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Open+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown { 
        font-family: 'Kanit', sans-serif !important; 
        scroll-behavior: smooth;
    }
    
    .stApp {
        background-color: #ffffff !important;
        color: #333333 !important;
    }

    header {visibility: hidden;} /* ซ่อน Header ปกติของ Streamlit */

    /* ========================================= */
    /* 1. FONT SIZE ADJUSTMENT (ยกเว้น Headers) */
    /* ========================================= */
    
    /* เพิ่มขนาดฟอนต์พื้นฐานให้ใหญ่ขึ้น */
    p, li, label, span, div.stMarkdown, .about-text, .stRadio label, .stFileUploader label {
        font-size: 1.25rem !important; /* ใหญ่ขึ้นเป็น 20px */
        line-height: 1.8 !important;
    }
    
    /* ยกเว้น Header ไม่ให้ขยายตาม (คงขนาดเดิมหรือตามคลาสที่กำหนด) */
    h1, h2, h3, .hero-title, .hero-sub, .about-header {
        /* ปล่อยให้ใช้ขนาดที่กำหนดเฉพาะของมัน */
    }

    /* ========================================= */
    /* 2. RESPONSIVE NAVBAR & SIDEBAR LOGIC      */
    /* ========================================= */
    
    /* --- Desktop View (หน้าจอกว้าง) --- */
    @media (min-width: 769px) {
        /* โชว์ Custom Navbar */
        .navbar { display: flex !important; }
        
        /* ซ่อนปุ่ม Hamburger และ Sidebar ของ Streamlit บน Desktop */
        section[data-testid="stSidebar"] { display: none !important; }
        button[kind="header"] { display: none !important; }
    }

    /* --- Mobile View (หน้าจอแคบ) --- */
    @media (max-width: 768px) {
        /* ซ่อน Custom Navbar */
        .navbar { display: none !important; }
        
        /* โชว์ปุ่ม Hamburger ของ Streamlit เพื่อเปิด Sidebar */
        button[kind="header"] { 
            display: block !important; 
            visibility: visible !important;
            color: #885D95 !important;
            position: fixed;
            top: 10px;
            right: 15px;
            z-index: 9999;
            background: white;
            border-radius: 5px;
            padding: 5px;
        }
        
        /* ปรับ Hero Section บนมือถือ */
        .hero-purple-container {
            margin-top: -50px; /* ดึงขึ้นไปแทนที่ navbar */
            padding-top: 80px;
        }
    }

    /* ========================================= */
    /* 3. HERO SECTION                           */
    /* ========================================= */
    .hero-purple-container {
        background-color: #885D95;
        width: 100vw; 
        margin-left: calc(-50vw + 50%); 
        margin-right: calc(-50vw + 50%);
        padding-top: 60px;  
        padding-bottom: 50px;
        margin-bottom: 60px; 
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        padding-left: 20px;
        padding-right: 20px;
    }

    .hero-title {
        color: #ffffff !important;
        font-size: clamp(2.2rem, 5vw, 4rem); /* Header คงเดิม/ใหญ่ตามสัดส่วน */
        font-weight: 700; 
        line-height: 1.2; 
        margin-bottom: 20px;
        text-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .hero-sub {
        color: #f0f0f0 !important;
        font-size: clamp(1.1rem, 2vw, 1.4rem); 
        font-weight: 300; 
        margin-bottom: 30px; 
        line-height: 1.6; 
        max-width: 800px;
    }
    
    .cta-button {
        background-color: #ffffff;
        color: #885D95 !important;
        padding: 18px 60px; 
        border-radius: 50px; 
        font-size: 1.4rem;
        font-weight: 700;
        text-decoration: none;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        display: inline-block;
        transition: all 0.3s ease;
    }
    .cta-button:hover { 
        transform: translateY(-5px); 
        background-color: #f8f8f8;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* ========================================= */
    /* 4. NAVBAR STYLE                           */
    /* ========================================= */
    .navbar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 15px 40px; 
        background-color: #ffffff; 
        border-bottom: none;
        color: #555; font-weight: 600;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        margin-top: -60px; 
        position: relative; z-index: 100;
    }
    
    .nav-links {
        display: flex;
        gap: 30px;
    }
    .nav-links a {
        font-size: 1.3rem; /* เมนูใหญ่ขึ้น */
    }

    /* ========================================= */
    /* 5. ABOUT SECTION (ARTICLE STYLE)          */
    /* ========================================= */
    .about-section {
        background-color: #67ACC3;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        padding: 80px 20px;
        color: white;
        display: flex; 
        flex-direction: column; 
        align-items: center;
        margin-bottom: 80px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .about-content { 
        max-width: 1000px; 
        width: 100%;
        text-align: left; /* จัดชิดซ้ายให้อ่านง่ายแบบบทความ */
    }
    .about-header { 
        font-size: 2.5rem; 
        font-weight: 700; 
        margin-bottom: 40px; 
        text-align: center;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 20px;
    }
    .about-subhead {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #e3f2fd;
    }
    .about-text { 
        font-size: 1.3rem; 
        line-height: 1.9; 
        font-weight: 300; 
        text-align: justify;
    }
    
    /* กรอบรูปใน About */
    .about-img-container {
        text-align: center;
        margin: 30px 0;
    }
    .about-img {
        max-width: 100%;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 4px solid rgba(255,255,255,0.2);
    }
    
    /* ปุ่ม External Link */
    .btn-hospital {
        display: inline-block;
        background-color: #ffffff;
        color: #67ACC3 !important;
        padding: 15px 30px;
        border-radius: 40px;
        font-weight: 700;
        text-decoration: none;
        margin-top: 30px;
        font-size: 1.2rem;
        transition: 0.3s;
        text-align: center;
    }
    .btn-hospital:hover {
        background-color: #f0f0f0;
        transform: scale(1.05);
        color: #558a9e !important;
    }

    /* ========================================= */
    /* 6. CARD & UI ELEMENTS                     */
    /* ========================================= */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #E0D0E8 !important; 
        border-radius: 24px !important;
        padding: 40px !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.1) !important;
        margin-bottom: 40px;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] * {
        color: #333333 !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        text-align: center !important;
        color: #885D95 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 25px !important;
    }

    /* Button Primary (Green) */
    div.stButton > button[kind="primary"] {
        background-color: #86B264 !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(134, 178, 100, 0.3);
        height: 60px; /* ปุ่มใหญ่ขึ้น */
        font-size: 1.3rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #759e56 !important;
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(134, 178, 100, 0.5);
    }

    .disclaimer-header h3 {
        color: #86B264 !important; 
    }
    
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# UI Content: Sidebar (Mobile Only)
# ----------------------------------
# Sidebar จะถูกซ่อนใน Desktop ด้วย CSS และโผล่มาเฉพาะ Mobile
with st.sidebar:
    st.title("เมนูหลัก")
    st.markdown("""
    * [🏠 หน้าหลัก](#top)
    * [📖 เกี่ยวกับโรคพาร์กินสัน](#about_area)
    * [🩺 แบบทดสอบคัดกรอง](#test_area)
    """)
    st.info("แนะนำให้เปิดใช้งานบนคอมพิวเตอร์เพื่อการแสดงผลที่สมบูรณ์ที่สุด")

# ----------------------------------
# UI Content: Main Page
# ----------------------------------

# จุด Anchor สำหรับปุ่ม Home
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# 1. Navbar (Desktop Only)
st.markdown("""
<div class="navbar">
    <div style="font-size: 1.5rem; color: #885D95; font-weight:700;">🧬 Parkinson AI</div>
    <div class="nav-links">
        <a href="#about_area" style="text-decoration:none; color:#67ACC3; font-weight:600;">เกี่ยวกับโรค</a>
        <a href="#test_area" style="text-decoration:none; color:#885D95; font-weight:600;">เริ่มใช้งาน</a>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Hero Section
st.markdown(f"""
<div class="hero-purple-container">
    <div class="hero-title">“Early detection changes everything.”</div>
    <div class="hero-sub">ใช้ AI ตรวจคัดกรองพาร์กินสันเบื้องต้น แม่นยำ รวดเร็ว และรู้ผลทันที<br>เพียงแค่วาดเส้น หรืออัปโหลดรูปภาพ</div>
    <a href="#test_area" class="cta-button">เริ่มทำแบบทดสอบ ➝</a>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 3. ABOUT SECTION (Article Style)
# =========================================================



# ----------------------------------
# 4. Model & Logic
# ----------------------------------
@st.cache_resource
def load_spiral_model():
    if os.path.exists("(Test_naja)effnet_parkinson_model.keras"):
        return tf.keras.models.load_model("(Test_naja)effnet_parkinson_model.keras")
    return None
spiral_model = load_spiral_model()

def preprocess(img):
    img = np.array(img.convert("RGB"))
    img = cv2.resize(img, (256, 256))   
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================================================
# 5. DISCLAIMER / TEST AREA
# =========================================================
st.markdown('<div id="test_area" style="padding-top: 50px;"></div>', unsafe_allow_html=True) 

if not st.session_state.consent_accepted:
    # --- Disclaimer Section ---
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
       with st.container(border=True):
            st.markdown('<div class="disclaimer-header"><h3 style="text-align:center;">⚠️ ข้อควรทราบก่อนทำการทดสอบ</h3></div>', unsafe_allow_html=True)
            
            st.write("ระบบนี้เป็นเครื่องมือคัดกรองเบื้องต้นโดยใช้ปัญญาประดิษฐ์ (AI)")
            st.error("ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ผู้เชี่ยวชาญได้")
            st.write("หากมีอาการผิดปกติหรือความกังวล กรุณาปรึกษาแพทย์เพื่อรับการตรวจเพิ่มเติม")
            
            st.markdown("---")
            st.markdown("**📝 คำแนะนำเพื่อให้ผลลัพธ์แม่นยำขึ้น**")
            st.markdown("""
            * นั่งในท่าที่สบาย แขนวางบนพื้นราบ
            * ทำจิตใจให้สงบ หลีกเลี่ยงความเครียด
            * วาดเส้นด้วยความเร็วและแรงกดตามธรรมชาติ
            """)
            st.markdown("---")

            st.write("อาการมือสั่นอาจเกิดจากหลายสาเหตุ เช่น ความเครียด ภาวะวิตกกังวล หรือโรคอื่นที่ไม่ใช่พาร์กินสัน")
            st.write("ระบบอาจไม่สามารถแยกแยะสาเหตุของอาการมือสั่นได้อย่างสมบูรณ์")
            st.write("ผลลัพธ์จึงควรใช้ประกอบการพิจารณาเท่านั้น")
            
            st.write("") 
            
            accepted = st.checkbox("ข้าพเจ้ารับทราบและยินยอมตามเงื่อนไขข้างต้น")
            
            st.write("")
            
            if st.button("ตกลง / เริ่มทำแบบทดสอบ", disabled=not accepted, type="primary", use_container_width=True):
                st.session_state.consent_accepted = True
                st.rerun()

else:
    # --- Testing Tool Section ---
    c1, c2, c3 = st.columns([1, 2, 1]) 
    with c2: 
        # SPIRAL CARD
        with st.container(border=True): 
            st.subheader("🌀 Spiral")
            spiral_mode = st.radio("เลือกวิธีใส่ภาพ (Spiral)", ["Upload", "Draw"], horizontal=True, key="spiral_mode")
            spiral_image = None
            if spiral_mode == "Upload":
                spiral_file = st.file_uploader("อัปโหลด Spiral", type=["png", "jpg", "jpeg"], key="spiral_upload")
                if spiral_file:
                    spiral_image = Image.open(spiral_file).convert("RGB")
                    st.image(spiral_image, caption="Preview", use_container_width=True)
            else:
                dc1, dc2, dc3 = st.columns([0.05, 1, 0.05])
                with dc2:
                    spiral_canvas = st_canvas(fill_color="rgba(255, 255, 255, 0)", stroke_width=6, stroke_color="black", background_color="#ffffff", height=300, width=450, drawing_mode="freedraw", key="spiral_draw")
                if spiral_canvas.image_data is not None:
                    spiral_image = Image.fromarray(spiral_canvas.image_data.astype("uint8")).convert("RGB")
            st.markdown("<br>", unsafe_allow_html=True)
            spiral_result_box = st.empty()

        # WAVE CARD
        with st.container(border=True): 
            st.subheader("🌊 Wave")
            wave_mode = st.radio("เลือกวิธีใส่ภาพ (Wave)", ["Upload", "Draw"], horizontal=True, key="wave_mode")
            wave_image = None
            if wave_mode == "Upload":
                wave_file = st.file_uploader("อัปโหลด Wave", type=["png", "jpg", "jpeg"], key="wave_upload")
                if wave_file:
                    wave_image = Image.open(wave_file).convert("RGB")
                    st.image(wave_image, caption="Preview", use_container_width=True)
            else:
                wc1, wc2, wc3 = st.columns([0.05, 1, 0.05])
                with wc2:
                    wave_canvas = st_canvas(fill_color="rgba(255, 255, 255, 0)", stroke_width=6, stroke_color="black", background_color="#ffffff", height=300, width=450, drawing_mode="freedraw", key="wave_draw")
                if wave_canvas.image_data is not None:
                    wave_image = Image.fromarray(wave_canvas.image_data.astype("uint8")).convert("RGB")
            st.markdown("<br>", unsafe_allow_html=True)
            wave_result_box = st.empty()

        # PROCESS BUTTON (สีเขียว)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 ประมวลผลทั้งหมด", type="primary", use_container_width=True):
            if spiral_image is not None and spiral_model is not None:
                try:
                    input_tensor = preprocess(spiral_image)
                    pred = spiral_model.predict(input_tensor)[0][0]
                    if pred > 0.5: spiral_result_box.error(f"🌀 Spiral : เสี่ยง Parkinson ({pred:.3f})")
                    else: spiral_result_box.success(f"🌀 Spiral : ปกติ ({pred:.3f})")
                except Exception as e: spiral_result_box.error(f"Error: {e}")
            elif spiral_image is None: spiral_result_box.warning("🌀 Spiral : ยังไม่ได้ใส่ภาพ")
            
            if wave_image is not None: wave_result_box.info("🌊 Wave : มีภาพแล้ว (รอโมเดล)")
            else: wave_result_box.warning("🌊 Wave : ยังไม่ได้ใส่ภาพ")
