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
# CSS Styles
# ----------------------------------
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Open+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown, div[data-testid="stDialog"] { 
        font-family: 'Kanit', sans-serif !important; 
        scroll-behavior: smooth;
    }
    .stApp { background-color: #ffffff !important; color: #333333 !important; }

    /* Remove Sidebar */
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

    /* --- Responsive Typography --- */
    @media (min-width: 992px) {
        .hero-title { font-size: 4rem !important; }
        .hero-sub { font-size: 1.6rem !important; }
        .about-text { font-size: 1.5rem !important; }
        
        /* ปุ่ม Start Test (Styled Streamlit Button) บน PC */
        div.stButton.hero-cta > button {
            font-size: 1.6rem !important; padding: 20px 70px !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size: 2.5rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] label,
        div[data-testid="stVerticalBlockBorderWrapper"] li { font-size: 1.5rem !important; }
        
        div[data-testid="stCanvas"] button {
            width: 60px !important; height: 60px !important; transform: scale(1.4); margin: 10px 15px !important;
        }
        .nav-links a { font-size: 1.4rem; }
    }

    @media (max-width: 991px) {
        .hero-title { font-size: 2rem !important; }
        .hero-sub { font-size: 1rem !important; }
        .about-text { font-size: 1rem !important; line-height: 1.5 !important; }
        
        /* ปุ่ม Start Test บน Mobile */
        div.stButton.hero-cta > button {
            font-size: 1.2rem !important; padding: 12px 40px !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size: 1.4rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] label,
        div[data-testid="stVerticalBlockBorderWrapper"] li { font-size: 0.95rem !important; }

        div[data-testid="stCanvas"] button {
            width: 40px !important; height: 40px !important; transform: scale(1.0); margin: 5px !important;
        }
        .navbar { flex-direction: column; gap: 10px; padding: 10px; }
        .nav-links a { font-size: 0.9rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] { padding: 20px !important; }
    }

    /* --- Canvas Fix (Responsive) --- */
    div[data-testid="stCanvas"] canvas {
        max-width: 100% !important;
        width: 100% !important;
        height: auto !important;
        border: 1px solid #ddd;
        border-radius: 8px;
        touch-action: none;
    }
    div[data-testid="stCanvas"] {
        display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; overflow: hidden;
    }

    /* --- Hero Section Styling --- */
    .hero-purple-container {
        background-color: #885D95; width: 100%; 
        padding: 60px 20px; margin-bottom: 40px; 
        text-align: center; color: white;
        display: flex; flex-direction: column; align-items: center;
    }
    .hero-title { font-weight: 700; margin-bottom: 15px; color: white !important; }
    .hero-sub { font-weight: 300; margin-bottom: 25px; max-width: 800px; color: #f0f0f0 !important; }
    
    /* แปลงปุ่ม Streamlit ให้สวยเหมือน CSS Button เดิม */
    div.stButton.hero-cta > button {
        background-color: #ffffff !important;
        color: #885D95 !important;
        border-radius: 50px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton.hero-cta > button:hover {
        transform: translateY(-5px) !important;
        background-color: #f8f8f8 !important;
    }

    /* --- General Styles --- */
    .about-section {
        background-color: #67ACC3; width: 100%; padding: 50px 20px; color: white;
        display: flex; flex-direction: column; align-items: center;
    }
    .about-content { max-width: 1000px; width: 100%; }
    .about-header { font-size: 2rem; font-weight: 700; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 15px; margin-bottom: 30px; }
    .about-img { max-width: 100%; height: auto; border-radius: 10px; margin: 20px 0; border: 4px solid rgba(255,255,255,0.2); }
    .btn-hospital {
        background-color: white; color: #67ACC3 !important; padding: 12px 25px;
        border-radius: 30px; font-weight: 700; text-decoration: none; margin-top: 20px; display: inline-block;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #E0D0E8 !important; border-radius: 20px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important; margin-bottom: 30px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 { color: #885D95 !important; text-align: center !important; font-weight: 700 !important; }

    /* Process Button */
    div.stButton > button[kind="primary"] {
        background-color: #86B264 !important; border: none !important; color: white !important;
        height: auto; padding: 15px; width: 100%; font-size: 1.3rem; border-radius: 10px;
    }
    div[role="radiogroup"] { gap: 15px; }

</style>
''', unsafe_allow_html=True)

# ----------------------------------
# POPUP (DIALOG) LOGIC
# ----------------------------------
@st.dialog(" ") # ใส่ Title ว่างไว้ เพื่อใช้ HTML ด้านในจัดการหัวข้อเองให้สวยงาม
def show_consent_dialog():
    # ใช้ HTML Structure ที่คุณต้องการ + เนื้อหาเดิมครบถ้วน
    st.markdown("""
    <div class="modal-box" style="font-family: 'Kanit', sans-serif;">
        <h3 style="text-align:center; margin-bottom:15px; color:#885D95; font-weight:700;">⚠️ ข้อควรทราบก่อนทำการทดสอบ</h3>
        
        <p style="font-size:1.1rem;">ระบบนี้เป็นเครื่องมือคัดกรองเบื้องต้นโดยใช้ปัญญาประดิษฐ์ (AI)</p> 
        <div style="background-color:#ffe6e6; padding:10px; border-radius:8px; border:1px solid #ffcccc; margin-bottom:15px;">
            <b style="color:#cc0000;">⛔ ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ผู้เชี่ยวชาญได้</b>
        </div>
        <p style="font-size:1.1rem;">หากมีอาการผิดปกติหรือความกังวล กรุณาปรึกษาแพทย์เพื่อรับการตรวจเพิ่มเติม</p>
        
        <hr style="margin: 20px 0; border-top: 1px solid #eee;">
        
        <b style="font-size:1.2rem; color:#333;">📝 คำแนะนำเพื่อให้ผลลัพธ์แม่นยำขึ้น</b>
        <ul style="margin-top:10px; font-size:1.1rem; line-height:1.6;">
            <li>นั่งในท่าที่สบาย แขนวางบนพื้นราบ</li>
            <li>ทำจิตใจให้สงบ หลีกเลี่ยงความเครียด</li>
            <li>วาดเส้นด้วยความเร็วและแรงกดตามธรรมชาติ</li>
        </ul>

        <div style="font-size:0.95rem; color:#666; margin-top:20px; font-style:italic;">
            * อาการมือสั่นอาจเกิดจากหลายสาเหตุ เช่น ความเครียด ภาวะวิตกกังวล หรือโรคอื่นที่ไม่ใช่พาร์กินสัน ผลลัพธ์จึงควรใช้ประกอบการพิจารณาเท่านั้น
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    if st.button("ข้าพเจ้ารับทราบและยินยอม (เริ่มทำแบบทดสอบ)", type="primary", use_container_width=True):
        st.session_state.consent_accepted = True
        st.rerun()

# ----------------------------------
# UI Content: Navbar
# ----------------------------------
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <div style="font-size: 1.5rem; color: #885D95; font-weight:700;">🧬 Parkinson AI</div>
    <div class="nav-links">
        <a href="#about_area" style="color:#67ACC3;">เกี่ยวกับโรค</a>
        <a href="#test_area" style="color:#885D95;">เริ่มใช้งาน</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# UI Content: Hero
# ----------------------------------
st.markdown('<div class="hero-purple-container">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">“Early detection changes everything.”</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ใช้ AI ตรวจคัดกรองพาร์กินสันเบื้องต้น แม่นยำ รวดเร็ว และรู้ผลทันที<br>เพียงแค่วาดเส้น หรืออัปโหลดรูปภาพ</div>', unsafe_allow_html=True)

# ปุ่ม Start Test ที่หน้าตาเหมือน CTA Button แต่เป็น Streamlit Button เพื่อเรียก Python Function
st.markdown('<div class="stButton hero-cta">', unsafe_allow_html=True)
if st.button("เริ่มทำแบบทดสอบ ➝"):
    show_consent_dialog() # เรียก Popup ขึ้นมาเมื่อกด
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------
# UI Content: About
# ----------------------------------
st.markdown('<div id="about_area" style="padding-top: 20px;"></div>', unsafe_allow_html=True) 

image_url = "https://kcmh.chulalongkornhospital.go.th/ec/wp-content/uploads/2019/02/Parkinson-Cover-1024x683.jpg"

about_html = f'''
<div class="about-section">
<div class="about-content">
<div class="about-header">ศูนย์ความเป็นเลิศทางการแพทย์<br>โรคพาร์กินสัน และกลุ่มโรคความเคลื่อนไหวผิดปกติ</div>
<div style="text-align:center;"><img src="{image_url}" class="about-img" alt="Parkinson Info"></div>
<div class="about-text">
โรคพาร์กินสัน (Parkinson’s Disease) ถือเป็นโรคความเสื่อมของระบบประสาทที่พบได้บ่อยเป็นอันดับที่ 2 รองจากโรคอัลไซเมอร์ มักพบในผู้ที่มีอายุ 60 ปีขึ้นไป แต่ในปัจจุบันเริ่มพบผู้ป่วยที่มีอายุน้อยลงเรื่อยๆ สาเหตุหลักเกิดจากการที่เซลล์สมองในส่วนที่สร้างสารสื่อประสาทชื่อ <b>"โดพามีน (Dopamine)"</b> เกิดการเสื่อมสลาย ทำให้สมองไม่สามารถควบคุมการเคลื่อนไหวของร่างกายได้อย่างปกติ
<br><br>
<div style="font-weight:600; margin-bottom:10px; color:#e3f2fd;">อาการที่ควรสังเกต (Warning Signs)</div>
อาการของโรคพาร์กินสันมักเริ่มต้นอย่างช้าๆ และค่อยเป็นค่อยไป โดยสัญญาณเตือนที่สำคัญแบ่งออกเป็น 2 กลุ่ม คือ:
<ul>
<li><b>อาการทางการเคลื่อนไหว:</b> อาการสั่นขณะอยู่นิ่ง (Resting Tremor), การเคลื่อนไหวช้า (Bradykinesia), กล้ามเนื้อแข็งเกร็ง (Rigidity) และการทรงตัวไม่ดี เดินซอยเท้าถี่</li>
<li><b>อาการที่ไม่ใช่การเคลื่อนไหว:</b> การรับรู้กลิ่นลดลง, ท้องผูกเรื้อรัง, นอนละเมอ, ภาวะซึมเศร้า หรือวิตกกังวล ซึ่งอาการเหล่านี้อาจเกิดขึ้นก่อนอาการสั่นหลายปี</li>
</ul>
<div style="font-weight:600; margin-bottom:10px; color:#e3f2fd;">ทำไมการตรวจพบเร็วถึงสำคัญ?</div>
แม้ว่าปัจจุบันโรคพาร์กินสันจะยังไม่สามารถรักษาให้หายขาดได้ แต่การตรวจพบในระยะเริ่มต้น (Early Detection) จะช่วยให้แพทย์สามารถวางแผนการรักษาเพื่อชะลอความเสื่อมของโรค ควบคุมอาการ และช่วยให้ผู้ป่วยสามารถใช้ชีวิตประจำวันได้อย่างมีคุณภาพยาวนานที่สุด
<br><br>
หากท่านหรือคนใกล้ชิดมีอาการที่น่าสงสัย ทางโรงพยาบาลจุฬาลงกรณ์ สภากาชาดไทย มีศูนย์ความเป็นเลิศทางการแพทย์ฯ ที่พร้อมให้คำปรึกษาและดูแลรักษาแบบครบวงจร ท่านสามารถศึกษาข้อมูลเพิ่มเติมได้ที่เว็บไซต์ด้านล่างนี้
</div>
<div style="text-align: center; margin-top: 40px;">
<a href="https://kcmh.chulalongkornhospital.go.th/ec/excellence-for-parkinsons-disease-related-disorders-th/" target="_blank" class="btn-hospital">
🏥 ศึกษาข้อมูลเพิ่มเติม - รพ.จุฬาลงกรณ์
</a>
</div>
</div>
</div>
'''
st.markdown(about_html, unsafe_allow_html=True)

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
st.markdown('<div id="test_area" style="padding-top: 40px;"></div>', unsafe_allow_html=True) 

# แสดงเนื้อหาหลัก (การ์ดทดสอบ) ต่อเมื่อยอมรับ Consent แล้ว
if st.session_state.consent_accepted:
    # -----------------------------------
    # SPIRAL CARD
    # -----------------------------------
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
            # 700px width เป็นค่าเริ่มต้น แต่ CSS จะบีบให้ไม่เกิน 100% ของจอ
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

    # -----------------------------------
    # WAVE CARD
    # -----------------------------------
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

    # -----------------------------------
    # PROCESS BUTTON
    # -----------------------------------
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
