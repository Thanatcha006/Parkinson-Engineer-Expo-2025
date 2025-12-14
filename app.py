import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os
from datetime import datetime
import streamlit.components.v1 as components

# ----------------------------------
# 1. Page Config (ต้องอยู่บรรทัดแรกสุด)
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
    
    /* บังคับฟอนต์และสีพื้นฐาน */
    html, body, [class*="css"], .stMarkdown { 
        font-family: 'Kanit', sans-serif !important; 
        scroll-behavior: smooth;

    }
    
    /* บังคับพื้นหลังแอปให้เป็นสีขาว (แก้ Dark Mode) */
    .stApp {
        background-color: #ffffff !important;
        color: #333333 !important;
    }

    header, footer {visibility: hidden;}


    /*  HERO SECTION:  */
    .hero-purple-container {
        background-color: #885D95;
        width: 100vw; 
        margin-left: calc(-50vw + 50%); 
        margin-right: calc(-50vw + 50%);
        padding-top: 60px;  
        padding-bottom: 40px;
        margin-bottom: 40px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        
        /* เพิ่ม Padding ซ้ายขวา เพื่อไม่ให้เนื้อหาชิดขอบจอเกินไป */
        padding-left: 20px;
        padding-right: 20px;
    }


    /* Text Styles */
    .hero-title {
        color: #ffffff !important;
        font-size: clamp(2rem, 5vw, 3.5rem); 
        font-weight: 700; 
        line-height: 1.2; 
        margin-bottom: 20px;
        text-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .hero-sub {
        color: #f0f0f0 !important;
        font-size: clamp(1rem, 2vw, 1.3rem); 
        font-weight: 300; 
        margin-bottom: 30px; 
        line-height: 1.6; 
        max-width: 800px;
    }
    
    /* Button Style */
    .cta-button {
        background-color: #ffffff;
        color: #885D95 !important;
        padding: 18px 60px; 
        border-radius: 50px; 
        font-size: 1.3rem;
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
    

    /* NAVBAR */
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

    /* CARD STYLE */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-width: 5px !important;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(136, 93, 149, 0.15);
        margin-bottom: 30px;
    }
    /* บังคับสีตัวหนังสือในการ์ด */
    div[data-testid="stVerticalBlockBorderWrapper"] * {
        color: #333333 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #4A4A4A !important;
    }

    /* UI Elements Colors */
    div[data-testid="stRadio"] label p { color: #333 !important; font-weight: 600; font-size: 1.1rem !important; }
    .stFileUploader label { color: #333 !important; }
    div[class*="stMarkdown"] p { color: #333 !important; }
    div.stButton > button { width: 100%; border-radius: 30px; height: 50px; font-size: 18px; }

   div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #E0D0E8 !important; 
        border-radius: 24px !important;
        padding: 40px !important;
        /* 🔥 เงาแบบ Popup (Deep Shadow) */
        box-shadow: 0 24px 64px rgba(0,0,0,0.15) !important; 
        margin-bottom: 40px;
    }

    /* --- จัด Text ในการ์ด --- */
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        text-align: center !important;
        color: #885D95 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 25px !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] p, 
    div[data-testid="stVerticalBlockBorderWrapper"] li,
    div[data-testid="stVerticalBlockBorderWrapper"] label {
        color: #333333 !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# 5. UI Content
# ----------------------------------

# Navbar
st.markdown("""
<div class="navbar">
    <div style="font-size: 1.3rem; color: #885D95; font-weight:700;">🧬 Parkinson AI</div>
    <div><a href="#test_area" style="text-decoration:none; color:#885D95; font-weight:600;">เริ่มใช้งาน</a></div>
</div>
""", unsafe_allow_html=True)

# HERO SECTION (เติม f หน้า string เพื่อให้ตัวแปรทำงาน)
st.markdown(f"""
<div class="hero-purple-container">
    <div class="hero-title">“Early detection changes everything.”</div>
    <div class="hero-sub">ใช้ AI ตรวจคัดกรองพาร์กินสันเบื้องต้น แม่นยำ รวดเร็ว และรู้ผลทันที<br>เพียงแค่วาดเส้น หรืออัปโหลดรูปภาพ</div>
    <a href="#test_area" class="cta-button">เริ่มทำแบบทดสอบ ➝</a>
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# 6. Model & Logic
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
# DIACLAIMER
# =========================================================
# จุด Anchor (เป้าหมายการเลื่อน)
st.markdown('<div id="test_area" style="padding-top: 20px;"></div>', unsafe_allow_html=True) 

# --- Logic: เช็คว่ายอมรับข้อตกลงหรือยัง ---
if not st.session_state.consent_accepted:
    # ถ้ายังไม่ยอมรับ ให้แสดง Disclaimer แทนเครื่องมือทดสอบ
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
       with st.container(border=True):
            st.subheader("⚠️ ข้อควรทราบก่อนทำการทดสอบ")
            
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
            
            if st.button("ตกลง / เริ่มทำแบบทดสอบ", disabled=not accepted, type="primary", use_container_width=True):
                st.session_state.consent_accepted = True
                st.rerun()

else:
    # --- ถ้า "ยอมรับแล้ว" ให้แสดงเครื่องมือทดสอบ ---
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

        # PROCESS BUTTON (สีเขียวแบบที่ชอบ)
        st.markdown("""
        <style>
        div.stButton.process-btn > button {
            background-color: #86B264 !important; /* สีเขียว */
            color: white !important;
            border: none !important;
        
            padding: 18px 60px !important;
            border-radius: 50px !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
        
            display: block !important;
            margin: 0 auto !important;
            width: auto !important;
        
            box-shadow: 0 6px 20px rgba(134, 178, 100, 0.4) !important;
            transition: transform 0.2s !important;
        }
        div.stButton.process-btn > button:hover {
            transform: translateY(-5px);
            background-color: #729c52 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 ประมวลผลทั้งหมด", use_container_width=True, key="process_btn"):
            if spiral_image is not None and spiral_model is not None:
                try:
                    input_tensor = preprocess(spiral_image)
                    pred = spiral_model.predict(input_tensor)[0][0]
                    if pred > 0.5: spiral_result_box.error(f"🌀 Spiral : เสี่ยง Parkinson ({pred:.3f})")
                    else: spiral_result_box.success(f"🌀 Spiral : ปกติ ({pred:.3f})")
                except Exception as e: spiral_result_box.error(f"Error: {e}")
            elif spiral_image is None: spiral_result_box.warning("🌀 Spiral : ยังไม่ได้ใส่ภาพ")
            elif spiral_model is None: spiral_result_box.error("❌ ไม่พบไฟล์โมเดล")

            if wave_image is not None: wave_result_box.info("🌊 Wave : มีภาพแล้ว (รอโมเดล)")
            else: wave_result_box.warning("🌊 Wave : ยังไม่ได้ใส่ภาพ")
