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
# ใช้ ''' เพื่อป้องกันปัญหากับ docstring ทั่วไป
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

    header {visibility: hidden;}

    /* RESPONSIVE NAVBAR */
    @media (min-width: 769px) {
        .navbar { display: flex !important; }
        section[data-testid="stSidebar"] { display: none !important; }
        button[kind="header"] { display: none !important; }
    }

    @media (max-width: 768px) {
        .navbar { display: none !important; }
        button[kind="header"] { 
            display: block !important; 
            visibility: visible !important;
            color: #885D95 !important;
            position: fixed;
            top: 15px; right: 15px;
            z-index: 99999;
            background: rgba(255,255,255,0.9);
            border-radius: 8px;
            padding: 5px;
        }
        .hero-purple-container {
            margin-top: -60px; 
            padding-top: 80px;
        }
    }

    /* HERO SECTION */
    .hero-purple-container {
        background-color: #885D95;
        width: 100vw; 
        margin-left: calc(-50vw + 50%); 
        margin-right: calc(-50vw + 50%);
        padding-top: 60px;  
        padding-bottom: 50px;
        margin-bottom: 60px; 
        text-align: center;
        display: flex; flex-direction: column; align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        padding-left: 20px; padding-right: 20px;
    }

    .hero-title {
        color: #ffffff !important;
        font-size: clamp(2.2rem, 5vw, 4rem); 
        font-weight: 700; margin-bottom: 20px;
    }
    .hero-sub {
        color: #f0f0f0 !important;
        font-size: clamp(1.2rem, 2vw, 1.5rem); 
        font-weight: 300; margin-bottom: 30px; 
        max-width: 800px; line-height: 1.6;
    }
    
    .cta-button {
        background-color: #ffffff;
        color: #885D95 !important;
        padding: 18px 60px; 
        border-radius: 50px; 
        font-size: 1.4rem; font-weight: 700;
        text-decoration: none;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        display: inline-block; transition: all 0.3s ease;
    }
    .cta-button:hover { 
        transform: translateY(-5px); 
        background-color: #f8f8f8;
    }
    
    /* NAVBAR STYLE */
    .navbar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 15px 40px; 
        background-color: #ffffff; 
        border-bottom: none;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        margin-top: -60px; 
        position: relative; z-index: 100;
    }
    .nav-links { display: flex; gap: 30px; }
    .nav-links a { font-size: 1.3rem; font-weight: 600; text-decoration: none; }

    /* ABOUT SECTION */
    .about-section {
        background-color: #67ACC3;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        padding: 80px 20px;
        color: white;
        display: flex; flex-direction: column; align-items: center;
        margin-bottom: 80px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .about-content { max-width: 1000px; width: 100%; text-align: left; }
    
    .about-header { 
        font-size: 2.5rem; font-weight: 700; 
        margin-bottom: 40px; text-align: center;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 20px; color: white !important;
    }
    .about-subhead {
        font-size: 1.8rem; font-weight: 600;
        margin-top: 30px; margin-bottom: 15px; color: #e3f2fd;
    }
    /* ปรับฟอนต์เฉพาะใน about-text */
    .about-text, .about-text li { 
        font-size: 1.3rem !important; 
        line-height: 1.9; 
        font-weight: 300; text-align: justify; color: white !important;
    }
    .about-img-container { text-align: center; margin: 30px 0; }
    .about-img {
        max-width: 100%; height: auto; border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 4px solid rgba(255,255,255,0.2);
    }
    .btn-hospital {
        display: inline-block; background-color: #ffffff;
        color: #67ACC3 !important; padding: 15px 30px;
        border-radius: 40px; font-weight: 700;
        text-decoration: none; margin-top: 30px;
        font-size: 1.2rem; transition: 0.3s;
        text-align: center; border: 2px solid white;
    }
    .btn-hospital:hover {
        background-color: #f0f0f0; transform: scale(1.05);
        color: #558a9e !important;
    }

    /* CARD & BUTTONS */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #E0D0E8 !important; 
        border-radius: 24px !important;
        padding: 40px !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.1) !important;
        margin-bottom: 40px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] * { color: #333333 !important; }
    
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        text-align: center !important; color: #885D95 !important;
        font-size: 2rem !important; font-weight: 700 !important;
        margin-bottom: 25px !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #86B264 !important;
        border: none !important; color: white !important;
        box-shadow: 0 4px 15px rgba(134, 178, 100, 0.3);
        height: 60px; font-size: 1.3rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #759e56 !important; transform: scale(1.02);
    }

    .disclaimer-header h3 { color: #86B264 !important; }
</style>
''', unsafe_allow_html=True)

# ----------------------------------
# UI Content: Sidebar (Mobile Only)
# ----------------------------------
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

# Anchor
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# 1. Navbar (Desktop)
st.markdown("""
<div class="navbar">
    <div style="font-size: 1.5rem; color: #885D95; font-weight:700;">🧬 Parkinson AI</div>
    <div class="nav-links">
        <a href="#about_area" style="color:#67ACC3;">เกี่ยวกับโรค</a>
        <a href="#test_area" style="color:#885D95;">เริ่มใช้งาน</a>
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
# 3. ABOUT SECTION 
# =========================================================
st.markdown('<div id="about_area" style="padding-top: 20px;"></div>', unsafe_allow_html=True) 

image_url = "https://kcmh.chulalongkornhospital.go.th/ec/wp-content/uploads/2019/02/Parkinson-Cover-1024x683.jpg"

# ✅ เปลี่ยนเป็น f-string แบบ Single Quote สามตัว (''') เพื่อให้ Highlighter ไม่เอ๋อ
about_html = f'''
<div class="about-section">
<div class="about-content">
<div class="about-header">ศูนย์ความเป็นเลิศทางการแพทย์<br>โรคพาร์กินสัน และกลุ่มโรคความเคลื่อนไหวผิดปกติ</div>
<div class="about-img-container">
<img src="{image_url}" class="about-img" alt="Parkinson Info">
</div>
<div class="about-text">
โรคพาร์กินสัน (Parkinson’s Disease) ถือเป็นโรคความเสื่อมของระบบประสาทที่พบได้บ่อยเป็นอันดับที่ 2 รองจากโรคอัลไซเมอร์ มักพบในผู้ที่มีอายุ 60 ปีขึ้นไป แต่ในปัจจุบันเริ่มพบผู้ป่วยที่มีอายุน้อยลงเรื่อยๆ สาเหตุหลักเกิดจากการที่เซลล์สมองในส่วนที่สร้างสารสื่อประสาทชื่อ <b>"โดพามีน (Dopamine)"</b> เกิดการเสื่อมสลาย ทำให้สมองไม่สามารถควบคุมการเคลื่อนไหวของร่างกายได้อย่างปกติ
<br><br>
<div class="about-subhead">อาการที่ควรสังเกต (Warning Signs)</div>
อาการของโรคพาร์กินสันมักเริ่มต้นอย่างช้าๆ และค่อยเป็นค่อยไป โดยสัญญาณเตือนที่สำคัญแบ่งออกเป็น 2 กลุ่ม คือ:
<ul>
<li><b>อาการทางการเคลื่อนไหว:</b> อาการสั่นขณะอยู่นิ่ง (Resting Tremor), การเคลื่อนไหวช้า (Bradykinesia), กล้ามเนื้อแข็งเกร็ง (Rigidity) และการทรงตัวไม่ดี เดินซอยเท้าถี่</li>
<li><b>อาการที่ไม่ใช่การเคลื่อนไหว:</b> การรับรู้กลิ่นลดลง, ท้องผูกเรื้อรัง, นอนละเมอ, ภาวะซึมเศร้า หรือวิตกกังวล ซึ่งอาการเหล่านี้อาจเกิดขึ้นก่อนอาการสั่นหลายปี</li>
</ul>
<div class="about-subhead">ทำไมการตรวจพบเร็วถึงสำคัญ?</div>
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
