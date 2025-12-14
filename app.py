import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os
import time

# ----------------------------------
# 1. Page Config
# ----------------------------------
st.set_page_config(page_title="Parkinson Tester", layout="wide", initial_sidebar_state="collapsed")

# Initialize Session State
if "consent_accepted" not in st.session_state:
    st.session_state.consent_accepted = False

# ตรวจสอบ URL Parameter ว่ามีการกดปุ่มเริ่มมาหรือไม่
# ถ้า URL มี ?start=true แสดงว่าผู้ใช้กดปุ่มมาแล้ว
query_params = st.query_params
is_started = query_params.get("start") == "true"

# ----------------------------------
# CSS Styles
# ----------------------------------
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Open+Sans:wght@400;600;700&display=swap');
    
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

    /* -----------------------------------------------------------
       RESPONSIVE TYPOGRAPHY
       ----------------------------------------------------------- */
    @media (min-width: 992px) {
        .hero-title { font-size: 4rem !important; }
        .hero-sub { font-size: 1.6rem !important; }
        .about-text { font-size: 1.5rem !important; }
        
        /* สไตล์ปุ่มเดิมของคุณ */
        .cta-button { font-size: 1.6rem !important; padding: 20px 70px; }
        
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
        .hero-title { font-size: 2.2rem !important; }
        .hero-sub { font-size: 1.1rem !important; }
        .about-text { font-size: 1.1rem !important; line-height: 1.6 !important; }
        
        .cta-button { font-size: 1.1rem !important; padding: 12px 30px; }

        div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size: 1.6rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] label,
        div[data-testid="stVerticalBlockBorderWrapper"] li { font-size: 1.1rem !important; }

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
    
    /* ปุ่ม HTML <a> Class เดิม 100% */
    .cta-button {
        background-color: #ffffff;
        color: #885D95 !important;
        border-radius: 50px; 
        font-weight: 700;
        text-decoration: none;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        display: inline-block; transition: all 0.3s ease;
    }
    .cta-button:hover { 
        transform: translateY(-5px); 
        background-color: #f8f8f8;
    }
    
    /* About Section */
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
        <a href="#about_area" style="color:#67ACC3;">เกี่ยวกับโรค</a>
        <a href="#test_area" style="color:#885D95;">เริ่มใช้งาน</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# UI Content: Hero
# ----------------------------------
# ปุ่ม <a> ที่เมื่อกดจะ Reload หน้าพร้อมพารามิเตอร์ ?start=true
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
# 5. TEST AREA (แทรกเนื้อหาระหว่าง Hero และ About)
# =========================================================
# เนื้อหาจะถูกสร้างขึ้นเมื่อกดปุ่ม (is_started = True) หรือเคยยอมรับเงื่อนไขแล้ว

if is_started or st.session_state.consent_accepted:

    # 1. จุด Anchor ที่จะให้เลื่อนลงมาหา
    st.markdown('<div id="test_content_anchor" style="padding-top: 20px;"></div>', unsafe_allow_html=True)

    # 2. JavaScript เพื่อเลื่อนหน้าจออัตโนมัติ (Auto-scroll)
    # สคริปต์นี้จะ "ตามหา" ID test_content_anchor ทุกๆ 0.1 วินาที จนกว่าจะเจอ แล้วเลื่อนลงมา
    st.markdown("""
        <script>
            var targetId = 'test_content_anchor';
            var scrollInterval = setInterval(function() {
                var element = window.parent.document.getElementById(targetId);
                if (element) {
                    element.scrollIntoView({behavior: "smooth", block: "center"});
                    clearInterval(scrollInterval);
                }
            }, 100);
        </script>
    """, unsafe_allow_html=True)

    if not st.session_state.consent_accepted:
        # Disclaimer Section
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
        # Testing Tool Section (Spiral / Wave)
        
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
                    if pred > 0.5: spiral_result_box.error(f"🌀 Spiral : เสี่ยง Parkinson ({pred:.3f})")
                    else: spiral_result_box.success(f"🌀 Spiral : ปกติ ({pred:.3f})")
                except Exception as e: spiral_result_box.error(f"Error: {e}")
            elif spiral_image is None: spiral_result_box.warning("🌀 Spiral : ยังไม่ได้ใส่ภาพ")
            
            if wave_image is not None: wave_result_box.info("🌊 Wave : มีภาพแล้ว (รอโมเดล)")
            else: wave_result_box.warning("🌊 Wave : ยังไม่ได้ใส่ภาพ")

else:
    # ถ้ายังไม่กดปุ่ม -> ไม่แสดงเนื้อหา
    pass

# =========================================================
# 6. ABOUT SECTION (อยู่ล่างสุดเสมอ)
# =========================================================
st.markdown('<div id="about_area" style="padding-top: 40px;"></div>', unsafe_allow_html=True) 

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
