import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os
import time
import base64

# ----------------------------------
# 1. Page Config
# ----------------------------------
st.set_page_config(page_title="Parkinson Tester", layout="wide", initial_sidebar_state="collapsed")

# Initialize Session State
if "consent_accepted" not in st.session_state:
    st.session_state.consent_accepted = False

# เช็ค Query Params
query_params = st.query_params
is_started = query_params.get("start") == "true"

# ----------------------------------
# CSS Styles
# ----------------------------------
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;700&family=Open+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown { 
        font-family: 'Kanit', sans-serif !important; 
        scroll-behavior: smooth;
    }
    .stApp { background-color: #ffffff !important; color: #333333 !important; }

    /* Hide Sidebar & Header */
    section[data-testid="stSidebar"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    
    /* Navbar */
    .navbar {
        display: flex !important;
        justify-content: space-between; align-items: center;
        padding: 15px 20px; 
        background-color: #ffffff; 
        border-bottom: 1px solid #eee;
        width: 100%;
        position: relative; z-index: 999;
        margin-top: -60px;
    }
    .nav-links { display: flex; gap: 20px; }
    .nav-links a { font-weight: 600; text-decoration: none; }

    /* -------------------------------------------------------
       RESULT CARD STYLES (NEW)
       ------------------------------------------------------- */
    .result-card {
        background-color: #67ACC3; /* สีฟ้าเดียวกับ About เดิม */
        color: white;
        border-radius: 20px;
        padding: 30px;
        margin-top: 30px;
        box-shadow: 0 10px 30px rgba(103, 172, 195, 0.4);
        animation: fadeIn 0.8s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-header {
        font-size: 1.8rem;
        font-weight: 700;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 15px;
        margin-bottom: 20px;
    }

    /* กล่องสถานะ (ขาว) */
    .status-box {
        background-color: white;
        border-radius: 12px;
        padding: 15px 20px;
        color: #333;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    /* Progress Bar */
    .confidence-wrapper { margin-bottom: 20px; }
    .progress-track {
        background-color: rgba(255,255,255,0.3);
        height: 12px;
        border-radius: 6px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background-color: #fff; /* สีขาววิ่งบนพื้นฟ้า */
        border-radius: 6px;
    }

    /* Text Details */
    .result-label { font-weight: 600; font-size: 1.2rem; margin-top: 15px; margin-bottom: 5px; color: #e3f2fd; }
    .result-text { font-weight: 300; font-size: 1.1rem; line-height: 1.6; margin-bottom: 15px; }
    .result-list { margin-top: 5px; padding-left: 20px; font-weight: 300; line-height: 1.6; }
    .disclaimer-small {
        font-size: 0.9rem;
        background: rgba(0,0,0,0.1);
        padding: 10px;
        border-radius: 8px;
        margin-top: 20px;
        font-style: italic;
    }

    /* -------------------------------------------------------
       RESPONSIVE
       ------------------------------------------------------- */
    @media (min-width: 992px) {
        .hero-title { font-size: 4rem !important; }
        .hero-sub { font-size: 1.6rem !important; }
        .cta-button { font-size: 1.6rem !important; padding: 20px 70px; }
        
        div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size: 2.5rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] p, label, li { font-size: 1.5rem !important; }
        
        div[data-testid="stCanvas"] button {
            width: 60px !important; height: 60px !important; transform: scale(1.4); margin: 10px 15px !important;
        }
        .nav-links a { font-size: 1.4rem; }
    }

    @media (max-width: 991px) {
        .hero-title { font-size: 2.2rem !important; }
        .hero-sub { font-size: 1.1rem !important; }
        .cta-button { font-size: 1.1rem !important; padding: 12px 30px; }

        div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size: 1.6rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] p, label, li { font-size: 1.1rem !important; }

        div[data-testid="stCanvas"] button {
            width: 40px !important; height: 40px !important; transform: scale(1.0); margin: 5px !important;
        }
        .navbar { flex-direction: column; gap: 10px; padding: 10px; }
        .nav-links a { font-size: 1rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] { padding: 20px !important; }
    }

    /* Fix Canvas Responsive */
    canvas {
        max-width: 100% !important;
        height: auto !important;
        border: 1px solid #ddd;
        border-radius: 8px;
    }
    div[data-testid="stCanvas"] {
        display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%;
    }

    /* Hero Section */
    .hero-purple-container {
        background-color: #885D95; width: 100%; 
        padding: 60px 20px; margin-bottom: 40px; 
        text-align: center; color: white;
        display: flex; flex-direction: column; align-items: center;
    }
    .hero-title { font-weight: 700; margin-bottom: 15px; color: white !important; }
    .hero-sub { font-weight: 300; margin-bottom: 25px; max-width: 800px; color: #f0f0f0 !important; }
    
    /* ปุ่ม HTML <a> เดิม */
    .cta-button {
        background-color: white; color: #885D95 !important;
        border-radius: 50px; font-weight: 700; text-decoration: none;
        display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        cursor: pointer;
    }
    .cta-button:hover {
        transform: translateY(-5px); 
        background-color: #f8f8f8;
    }

    /* Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #E0D0E8 !important; border-radius: 20px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important; margin-bottom: 30px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 { color: #885D95 !important; text-align: center !important; font-weight: 700 !important; }

    div.stButton > button[kind="primary"] {
        background-color: #86B264 !important; border: none !important; color: white !important;
        height: auto; padding: 15px; width: 100%; font-size: 1.3rem; border-radius: 10px;
    }
    
    div[role="radiogroup"] { gap: 15px; }

</style>
''', unsafe_allow_html=True)

# ----------------------------------
# UI Content: Navbar
# ----------------------------------
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <div style="font-size: 1.5rem; color: #885D95; font-weight:700;">🧬 Parkinson AI</div>
    <div class="nav-links">
        <a href="#test_area" style="color:#885D95;">เริ่มใช้งาน</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# UI Content: Hero
# ----------------------------------
st.markdown(f"""
<div class="hero-purple-container">
    <div class="hero-title">“Early detection changes everything.”</div>
    <div class="hero-sub">ใช้ AI ตรวจคัดกรองพาร์กินสันเบื้องต้น แม่นยำ รวดเร็ว และรู้ผลทันที<br>เพียงแค่วาดเส้น หรืออัปโหลดรูปภาพ</div>
    <a href="?start=true" class="cta-button" target="_self">เริ่มทำแบบทดสอบ ➝</a>
</div>
""", unsafe_allow_html=True)

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
# 5. TEST AREA
# =========================================================
if is_started or st.session_state.consent_accepted:

    st.markdown('<div id="test_content_anchor" style="padding-top: 20px;"></div>', unsafe_allow_html=True)

    # JS Auto-scroll
    st.markdown("""
        <script>
            var targetId = 'test_content_anchor';
            var scrollInterval = setInterval(function() {
                var element = window.parent.document.getElementById(targetId);
                if (element) {
                    setTimeout(function(){
                         element.scrollIntoView({behavior: "smooth", block: "center"});
                    }, 300);
                    clearInterval(scrollInterval);
                }
            }, 100);
        </script>
    """, unsafe_allow_html=True)

    if not st.session_state.consent_accepted:
        # Disclaimer
        c1, c2, c3 = st.columns([1, 8, 1]) 
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
        # Test Section
        st.markdown('<div id="test_area" style="padding-top: 40px;"></div>', unsafe_allow_html=True)

        # SPIRAL CARD
        with st.container(border=True): 
            st.subheader("🌀 Spiral")
            spiral_mode = st.radio("เลือกวิธีใส่ภาพ (Spiral)", ["Upload", "Draw"], horizontal=True, key="spiral_mode")
            st.markdown("---")
            spiral_image = None
            if spiral_mode == "Upload":
                uc1, uc2, uc3 = st.columns([0.1, 1, 0.1])
                with uc2:
                    spiral_file = st.file_uploader("อัปโหลด Spiral", type=["png", "jpg", "jpeg"], key="spiral_upload")
                    if spiral_file:
                        spiral_image = Image.open(spiral_file).convert("RGB")
                        st.image(spiral_image, caption="Preview", use_container_width=True)
            else:
                spiral_canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=6,
                    stroke_color="black",
                    background_color="#ffffff",
                    height=500,
                    width=700, 
                    drawing_mode="freedraw",
                    key="spiral_draw",
                    display_toolbar=True
                )
                if spiral_canvas.image_data is not None:
                    spiral_image = Image.fromarray(spiral_canvas.image_data.astype("uint8")).convert("RGB")
            st.markdown("<br>", unsafe_allow_html=True)
            spiral_result_box = st.empty()

        # WAVE CARD
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True): 
            st.subheader("🌊 Wave")
            wave_mode = st.radio("เลือกวิธีใส่ภาพ (Wave)", ["Upload", "Draw"], horizontal=True, key="wave_mode")
            st.markdown("---")
            wave_image = None
            if wave_mode == "Upload":
                uc1, uc2, uc3 = st.columns([0.1, 1, 0.1])
                with uc2:
                    wave_file = st.file_uploader("อัปโหลด Wave", type=["png", "jpg", "jpeg"], key="wave_upload")
                    if wave_file:
                        wave_image = Image.open(wave_file).convert("RGB")
                        st.image(wave_image, caption="Preview", use_container_width=True)
            else:
                wave_canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=6,
                    stroke_color="black",
                    background_color="#ffffff",
                    height=500,
                    width=700,
                    drawing_mode="freedraw",
                    key="wave_draw",
                    display_toolbar=True
                )
                if wave_canvas.image_data is not None:
                    wave_image = Image.fromarray(wave_canvas.image_data.astype("uint8")).convert("RGB")
            st.markdown("<br>", unsafe_allow_html=True)
            wave_result_box = st.empty()

        # PROCESS BUTTON
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 ประมวลผลทั้งหมด", type="primary", use_container_width=True):
            if spiral_image is not None and spiral_model is not None:
                try:
                    input_tensor = preprocess(spiral_image)
                    pred = spiral_model.predict(input_tensor)[0][0]
                    
                    # คำนวณความมั่นใจและสถานะ
                    if pred > 0.5:
                        status_text = "มีความเสี่ยงพาร์กินสัน (Risk Detected)"
                        status_color = "#D32F2F" # สีแดง
                        confidence = pred * 100
                        desc_text = "รูปแบบการวาดมีความสั่นไหวหรือไม่ต่อเนื่อง ซึ่งอาจสัมพันธ์กับอาการเริ่มต้นของโรค"
                    else:
                        status_text = "✅ ไม่พบความผิดปกติเด่นชัด (Normal)"
                        status_color = "#388E3C" # สีเขียว
                        confidence = (1 - pred) * 100
                        desc_text = "รูปแบบการวาดมีความใกล้เคียงกับกลุ่มตัวอย่างทั่วไป ไม่พบอาการสั่นที่ผิดปกติชัดเจน"
                    
                    # สร้าง HTML Result Card
                    result_html = f"""
                    <div class="result-card">
                        <div class="result-header">🧪 ผลการคัดกรองเบื้องต้น (Spiral Test)</div>
                        
                        <div class="status-box" style="color: {status_color};">
                            {status_text}
                        </div>
                        
                        <div class="confidence-wrapper">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <span>ระดับความเชื่อมั่นของโมเดล (Confidence)</span>
                                <span>{confidence:.1f}%</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-fill" style="width: {confidence}%;"></div>
                            </div>
                        </div>
                        
                        <div class="result-label">📝 คำอธิบาย:</div>
                        <div class="result-text">{desc_text}</div>
                        
                        <div class="result-label">💡 คำแนะนำ:</div>
                        <ul class="result-list">
                            <li>หากยังมีความกังวล หรือผลการทดสอบไม่ชัดเจน สามารถทำการทดสอบซ้ำได้</li>
                            <li>ควรทำในสภาวะที่ผ่อนคลาย ไม่เกร็งข้อมือ</li>
                            <li>หากผลระบุว่ามีความเสี่ยง ควรปรึกษาแพทย์เพื่อรับการตรวจวินิจฉัยอย่างละเอียด</li>
                        </ul>
                        
                        <div class="disclaimer-small">
                            ⚠️ หมายเหตุ: ผลลัพธ์นี้เกิดจากการประมวลผลด้วย AI เพื่อการคัดกรองเบื้องต้นเท่านั้น 
                            <b>ไม่ใช่การวินิจฉัยทางการแพทย์</b> โปรดใช้วิจารณญาณ
                        </div>
                    </div>
                    """
                    # แสดงผล Card
                    spiral_result_box.markdown(result_html, unsafe_allow_html=True)
                    
                except Exception as e: 
                    spiral_result_box.error(f"Error: {e}")
            elif spiral_image is None: 
                spiral_result_box.warning("🌀 Spiral : ยังไม่ได้ใส่ภาพ")
            
            # Wave (Placeholder logic)
            if wave_image is not None: 
                wave_result_box.info("🌊 Wave : มีภาพแล้ว (รอโมเดล)")
            else: 
                wave_result_box.warning("🌊 Wave : ยังไม่ได้ใส่ภาพ")

else:
    # ถ้ายังไม่กดปุ่ม -> ไม่แสดงเนื้อหา
    pass
