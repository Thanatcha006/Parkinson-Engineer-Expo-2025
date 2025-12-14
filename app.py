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
# Helper Function: แปลงรูปภาพเป็น Base64
# ----------------------------------
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

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
       RESPONSIVE LAYOUT & TYPOGRAPHY
       ------------------------------------------------------- */
    @media (min-width: 992px) {
        .hero-title { font-size: 4rem !important; }
        .hero-sub { font-size: 1.6rem !important; }
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
    
    /* -------------------------------------------------------
       ABOUT SECTION STYLES (NEW LAYOUT)
       ------------------------------------------------------- */
    .about-section {
        background-color: #67ACC3;
        width: 100%;
        padding: 60px 20px;
        color: white;
        display: flex;
        justify-content: center;
    }
    
    .about-container {
        max-width: 1200px;
        width: 100%;
    }
    
    /* Header ใหญ่เหมือนเดิม */
    .about-header-large {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 20px;
        margin-bottom: 40px;
    }

    /* Grid สำหรับแบ่งซ้ายขวาใน PC */
    .about-body-grid {
        display: grid;
        grid-template-columns: 1fr; /* ค่าเริ่มต้น Mobile: 1 คอลัมน์ */
        gap: 40px;
        align-items: center;
    }

    /* รูปภาพ */
    .about-img-responsive {
        width: 100%;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 4px solid rgba(255, 255, 255, 0.2);
    }
    
    /* เนื้อหาข้อความ */
    .about-text-content {
        font-size: 1.1rem;
        line-height: 1.8;
        font-weight: 300;
        text-align: justify;
    }
    
    /* Quote Box */
    .quote-box {
        background-color: rgba(255, 255, 255, 0.15);
        border-left: 6px solid #ffffff;
        padding: 30px;
        margin-top: 50px;
        border-radius: 10px;
        font-size: 1.4rem;
        font-style: italic;
        font-weight: 500;
        line-height: 1.6;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        width: 100%;
    }

    /* >>> Desktop Only Rules (แบ่งซ้ายขวา) <<< */
    @media (min-width: 992px) {
        .about-body-grid {
            grid-template-columns: 1fr 1.2fr; /* แบ่ง 2 คอลัมน์ (รูป 1 : ข้อความ 1.2) */
        }
        .about-text-content {
            font-size: 1.35rem; /* ขยายตัวหนังสือ */
        }
        .about-header-large {
            font-size: 3.5rem; /* ขยายหัวข้อ */
        }
        .quote-box {
            font-size: 1.6rem;
        }
    }
    /* ------------------------------------------------------- */

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

    # 1. Anchor Point
    st.markdown('<div id="test_content_anchor" style="padding-top: 20px;"></div>', unsafe_allow_html=True)

    # 2. JS Auto-scroll
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
        # Testing Tool Section
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
# 6. ABOUT SECTION (New Layout & Content)
# =========================================================
st.markdown('<div id="about_area" style="padding-top: 40px;"></div>', unsafe_allow_html=True) 

# ดึงรูปจากเครื่อง
img_b64 = get_image_base64("parkinson cover.png")

if img_b64:
    img_tag = f'<img src="data:image/png;base64,{img_b64}" class="about-img-responsive" alt="Parkinson Cover">'
else:
    img_tag = '<div style="background:rgba(255,255,255,0.2); padding:40px; color:white; border-radius:15px; text-align:center; border: 2px dashed white;">⚠️ ไม่พบไฟล์ parkinson cover.png<br>กรุณาวางไฟล์รูปภาพไว้ในโฟลเดอร์เดียวกับไฟล์โค้ด</div>'

# HTML Layout ใหม่
about_html = f'''
<div class="about-section">
    <div class="about-container">
        
        <div class="about-header-large">โรคพาร์กินสัน (Parkinson’s Disease)</div>
        
        <div class="about-body-grid">
            
            <div class="about-image-container">
                {img_tag}
            </div>
            
            <div class="about-text-container">
                <div class="about-text-content">
                    โรคพาร์กินสันเป็นโรคความเสื่อมของระบบประสาทที่พบได้บ่อยเป็นอันดับต้น ๆ ของโลก มักพบในผู้ที่มีอายุ 60 ปีขึ้นไป แต่ในปัจจุบันเริ่มพบผู้ป่วยในวัยที่อายุน้อยลงมากขึ้น สาเหตุหลักเกิดจากการเสื่อมของเซลล์สมองที่สร้างสาร โดพามีน (Dopamine) ซึ่งมีบทบาทสำคัญในการควบคุมการเคลื่อนไหวของร่างกาย เมื่อระดับโดพามีนลดลง จะส่งผลให้การเคลื่อนไหวผิดปกติ
                    <br><br>
                    อาการที่พบบ่อย ได้แก่ มือสั่นขณะอยู่นิ่ง การเคลื่อนไหวช้า กล้ามเนื้อแข็งเกร็ง การทรงตัวไม่ดี รวมถึงอาการอื่น ๆ เช่น การรับรู้กลิ่นลดลง ท้องผูก หรือความผิดปกติของการนอนหลับ ซึ่งอาจเกิดขึ้นก่อนอาการสั่น
                </div>
            </div>
            
        </div>
        
        <div class="quote-box">
            “แม้โรคพาร์กินสันจะยังไม่สามารถรักษาให้หายขาดได้ แต่การตรวจพบตั้งแต่ระยะเริ่มต้นจะช่วยให้สามารถควบคุมอาการ ชะลอความเสื่อมของโรค และช่วยให้ผู้ป่วยมีคุณภาพชีวิตที่ดีขึ้น”
        </div>
        
    </div>
</div>
'''
st.markdown(about_html, unsafe_allow_html=True)
