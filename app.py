import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os

# ----------------------------------
# 1. Page Config
# ----------------------------------
st.set_page_config(page_title="Parkinson Tester", layout="wide", initial_sidebar_state="collapsed")

if "consent_accepted" not in st.session_state:
    st.session_state.consent_accepted = False

# ----------------------------------
# CSS Styles (เพิ่มส่วนจัดกึ่งกลาง Canvas)
# ----------------------------------
st.markdown('''
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

    /* ... (CSS เดิมของ Hero/Navbar คงไว้ตามเดิม) ... */
    /* ... (ขอย่อส่วน CSS เดิมเพื่อความกระชับ ให้คง CSS เดิมของคุณไว้ แล้วเพิ่มส่วนล่างนี้เข้าไป) ... */

    /* ----- เพิ่ม CSS สำหรับจัดกึ่งกลาง Canvas และขยายการ์ด ----- */
    
    /* ขยายการ์ดให้เต็มพื้นที่ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #E0D0E8 !important; 
        border-radius: 24px !important;
        padding: 40px !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.1) !important;
        margin-bottom: 40px;
        width: 100% !important; /* บังคับเต็มจอ */
    }

    /* จัดกึ่งกลางตัว Canvas */
    div[data-testid="stCanvas"] {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
    }
    
    div[data-testid="stCanvas"] > div {
        /* จัด Toolbar ให้ดูมีระยะห่าง */
        display: flex;
        flex-direction: column; 
        align-items: center;
    }

    /* ปุ่ม Process */
    div.stButton > button[kind="primary"] {
        background-color: #86B264 !important;
        border: none !important; color: white !important;
        box-shadow: 0 4px 15px rgba(134, 178, 100, 0.3);
        height: 60px; font-size: 1.3rem;
        width: 100%;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #759e56 !important; transform: scale(1.02);
    }
    
    .disclaimer-header h3 { color: #86B264 !important; }

    /* ปรับ Hero และ Navbar เดิม (คงไว้) */
    header {visibility: hidden;}
    @media (min-width: 769px) {
        .navbar { display: flex !important; }
        section[data-testid="stSidebar"] { display: none !important; }
        button[kind="header"] { display: none !important; }
    }
    @media (max-width: 768px) {
        .navbar { display: none !important; }
        .hero-purple-container { margin-top: -60px; padding-top: 80px; }
        /* ถ้าเป็นมือถือ ให้ Canvas ไม่ล้นจอเกินไป (Optional) */
        canvas { max-width: 100% !important; }
    }
    .hero-purple-container {
        background-color: #885D95; width: 100vw; 
        margin-left: calc(-50vw + 50%); margin-right: calc(-50vw + 50%);
        padding-top: 60px; padding-bottom: 50px; margin-bottom: 60px; 
        text-align: center; display: flex; flex-direction: column; align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1); padding-left: 20px; padding-right: 20px;
    }
    .hero-title { color: #ffffff !important; font-size: clamp(2.2rem, 5vw, 4rem); font-weight: 700; margin-bottom: 20px; }
    .hero-sub { color: #f0f0f0 !important; font-size: clamp(1.2rem, 2vw, 1.5rem); font-weight: 300; margin-bottom: 30px; max-width: 800px; line-height: 1.6; }
    .cta-button { background-color: #ffffff; color: #885D95 !important; padding: 18px 60px; border-radius: 50px; font-size: 1.4rem; font-weight: 700; text-decoration: none; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2); display: inline-block; transition: all 0.3s ease; }
    .cta-button:hover { transform: translateY(-5px); background-color: #f8f8f8; }
    .navbar { display: flex; justify-content: space-between; align-items: center; padding: 15px 40px; background-color: #ffffff; width: 100vw; margin-left: calc(-50vw + 50%); margin-right: calc(-50vw + 50%); margin-top: -60px; position: relative; z-index: 100; }
    .nav-links { display: flex; gap: 30px; }
    .nav-links a { font-size: 1.3rem; font-weight: 600; text-decoration: none; }
    .about-section { background-color: #67ACC3; width: 100vw; margin-left: calc(-50vw + 50%); margin-right: calc(-50vw + 50%); padding: 80px 20px; color: white; display: flex; flex-direction: column; align-items: center; margin-bottom: 80px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .about-content { max-width: 1000px; width: 100%; text-align: left; }
    .about-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 40px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 20px; color: white !important; }
    .about-subhead { font-size: 1.8rem; font-weight: 600; margin-top: 30px; margin-bottom: 15px; color: #e3f2fd; }
    .about-text, .about-text li { font-size: 1.3rem !important; line-height: 1.9; font-weight: 300; text-align: justify; color: white !important; }
    .about-img-container { text-align: center; margin: 30px 0; }
    .about-img { max-width: 100%; height: auto; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); border: 4px solid rgba(255,255,255,0.2); }
    .btn-hospital { display: inline-block; background-color: #ffffff; color: #67ACC3 !important; padding: 15px 30px; border-radius: 40px; font-weight: 700; text-decoration: none; margin-top: 30px; font-size: 1.2rem; transition: 0.3s; text-align: center; border: 2px solid white; }
    .btn-hospital:hover { background-color: #f0f0f0; transform: scale(1.05); color: #558a9e !important; }
    
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        text-align: center !important; color: #885D95 !important;
        font-size: 2rem !important; font-weight: 700 !important;
        margin-bottom: 25px !important;
    }
</style>
''', unsafe_allow_html=True)

# ... (ส่วน Sidebar, Navbar, Hero, About และ Function Model เหมือนเดิม) ...
# (ใส่โค้ดส่วน UI Sidebar, Navbar, Hero, About, Model Loading เดิมของคุณตรงนี้)
# ...

# =========================================================
# 5. DISCLAIMER / TEST AREA (ส่วนที่แก้ไข)
# =========================================================
st.markdown('<div id="test_area" style="padding-top: 50px;"></div>', unsafe_allow_html=True) 

if not st.session_state.consent_accepted:
    # --- Disclaimer Section ---
    # ใช้ Columns บีบเฉพาะหน้า Disclaimer ก็พอ (เพื่อให้ดูสวยงามไม่กว้างเกินไป)
    d1, d2, d3 = st.columns([1, 2, 1])
    with d2:
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
    # ❌ เอา c1, c2, c3 ออกเพื่อให้การ์ดกว้างเต็มจอ (Full Width)
    
    # --- SPIRAL CARD ---
    with st.container(border=True): 
        st.subheader("🌀 Spiral Task")
        
        # จัด Layout ภายในการ์ด
        col_input, col_display = st.columns([1, 3]) # แบ่งซ้ายขวาเล็กน้อย หรือจะเอาไว้บนล่างก็ได้
        
        # ปรับให้ปุ่มเลือกโหมดอยู่ตรงกลางสวยๆ
        st.write("เลือกวิธีการนำเข้าภาพ:")
        spiral_mode = st.radio("Mode (Spiral)", ["Upload Image", "Draw on Canvas"], horizontal=True, key="spiral_mode", label_visibility="collapsed")
        
        st.markdown("---")

        spiral_image = None
        
        if spiral_mode == "Upload":
            # โหมดอัปโหลด จัดให้อยู่กลาง
            uc1, uc2, uc3 = st.columns([1, 2, 1])
            with uc2:
                spiral_file = st.file_uploader("อัปโหลดรูปภาพ Spiral", type=["png", "jpg", "jpeg"], key="spiral_upload")
                if spiral_file:
                    spiral_image = Image.open(spiral_file).convert("RGB")
                    st.image(spiral_image, caption="Preview", use_container_width=True)

        else: # Mode Draw
            # โหมดวาด - จัดกึ่งกลาง และขยาย Canvas
            # ใช้ Columns จัดกึ่งกลางพื้นที่วาด
            dc1, dc2, dc3 = st.columns([0.1, 1, 0.1]) 
            with dc2:
                # ปรับ width=700, height=500 เพื่อให้ใหญ่ขึ้นสำหรับ PC
                spiral_canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=6,
                    stroke_color="black",
                    background_color="#ffffff",
                    height=500,  # เพิ่มความสูง
                    width=700,   # เพิ่มความกว้าง (บนมือถืออาจต้องเลื่อนซ้ายขวาเล็กน้อย)
                    drawing_mode="freedraw",
                    key="spiral_draw",
                    display_toolbar=True # โชว์ปุ่มลบ/ย้อนกลับ
                )
            
            if spiral_canvas.image_data is not None:
                spiral_image = Image.fromarray(spiral_canvas.image_data.astype("uint8")).convert("RGB")
        
        st.markdown("<br>", unsafe_allow_html=True)
        spiral_result_box = st.empty()

    # --- WAVE CARD ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True): 
        st.subheader("🌊 Wave Task")
        
        st.write("เลือกวิธีการนำเข้าภาพ:")
        wave_mode = st.radio("Mode (Wave)", ["Upload Image", "Draw on Canvas"], horizontal=True, key="wave_mode", label_visibility="collapsed")
        
        st.markdown("---")

        wave_image = None
        if wave_mode == "Upload":
            uc1, uc2, uc3 = st.columns([1, 2, 1])
            with uc2:
                wave_file = st.file_uploader("อัปโหลดรูปภาพ Wave", type=["png", "jpg", "jpeg"], key="wave_upload")
                if wave_file:
                    wave_image = Image.open(wave_file).convert("RGB")
                    st.image(wave_image, caption="Preview", use_container_width=True)
        else:
            # ใช้ Columns จัดกึ่งกลางพื้นที่วาด
            wc1, wc2, wc3 = st.columns([0.1, 1, 0.1])
            with wc2:
                wave_canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=6,
                    stroke_color="black",
                    background_color="#ffffff",
                    height=500, # เพิ่มความสูง
                    width=700,  # เพิ่มความกว้าง
                    drawing_mode="freedraw",
                    key="wave_draw",
                    display_toolbar=True
                )
            if wave_canvas.image_data is not None:
                wave_image = Image.fromarray(wave_canvas.image_data.astype("uint8")).convert("RGB")
        
        st.markdown("<br>", unsafe_allow_html=True)
        wave_result_box = st.empty()

    # --- PROCESS BUTTON ---
    st.markdown("<br>", unsafe_allow_html=True)
    # ปุ่มกดประมวลผลใหญ่ เต็มจอ
    if st.button("🔍 ประมวลผลทั้งหมด (Analyze All)", type="primary", use_container_width=True):
        if spiral_image is not None and spiral_model is not None:
            try:
                input_tensor = preprocess(spiral_image)
                pred = spiral_model.predict(input_tensor)[0][0]
                if pred > 0.5: spiral_result_box.error(f"🌀 Spiral : พบความเสี่ยง Parkinson ({pred:.3f})")
                else: spiral_result_box.success(f"🌀 Spiral : ผลปกติ ({pred:.3f})")
            except Exception as e: spiral_result_box.error(f"Error: {e}")
        elif spiral_image is None: spiral_result_box.warning("🌀 Spiral : กรุณาวาดหรืออัปโหลดภาพก่อน")
        
        if wave_image is not None: wave_result_box.info("🌊 Wave : ได้รับภาพแล้ว (กำลังรอโมเดล)")
        else: wave_result_box.warning("🌊 Wave : กรุณาวาดหรืออัปโหลดภาพก่อน")
