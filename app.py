import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import base64
import os

# ----------------------------------
# 1. Setup & Helper Functions
# ----------------------------------
st.set_page_config(page_title="Parkinson AI", layout="wide", initial_sidebar_state="collapsed")

# ฟังก์ชันแปลงรูปเป็น Base64 (จำเป็นมากสำหรับ Custom HTML)
def get_img_as_base64(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- โหลดรูปปก (แก้ชื่อไฟล์ตรงนี้ให้ตรงกับไฟล์ของคุณ) ---
cover_image_file = "parkinson cover.svg"  # หรือ .png
img_base64 = get_img_as_base64(cover_image_file)

# ----------------------------------
# 2. Custom CSS & HTML Design
# ----------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Open+Sans:wght@400;600;700&display=swap');

    /* ลบขอบขาวของ Streamlit ออกให้หมด */
    .block-container {{
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }}
    
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    html, body, [class*="css"] {{
        font-family: 'Kanit', sans-serif;
        scroll-behavior: smooth;
    }}

    /* Navbar */
    .navbar {{
        position: fixed;
        top: 0;
        width: 100%;
        z-index: 999;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 40px;
        height: 70px;
        background-color: white; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    
    /* Hero Section (พื้นที่สีพีช) */
    .hero-section {{
        background-color: #FFDFD0;
        min-height: 100vh; /* สูงเต็มจอ */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between; /* ดันข้อความไปบน รูปไปล่าง */
        text-align: center;
        padding-top: 100px;
    }}

    .hero-content {{
        z-index: 10;
        max-width: 800px;
        padding: 0 20px;
    }}

    .hero-title {{
        color: #222;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 15px;
    }}

    .hero-sub {{
        color: #666;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 30px;
    }}

    .cta-button {{
        background-color: #8c7ae6; /* สีม่วงแบบ 16p */
        color: white !important;
        padding: 15px 50px;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0 4px 10px rgba(136, 93, 149, 0.4);
        transition: transform 0.2s;
        display: inline-block;
    }}

    .cta-button:hover {{
        transform: translateY(-3px);
        background-color: #7b6ac4;
    }}

    /* รูปภาพ */
    .hero-image-container {{
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: flex-end;
    }}
    
    .hero-img {{
        width: 100%;
        max-width: 1000px; 
        height: auto;
        display: block;
    }}
    
    /* พื้นที่สำหรับ App Logic */
    .app-container {{
        max-width: 1000px;
        margin: 0 auto;
        padding: 60px 20px;
        background-color: white;
    }}
    
    /* ปรับแต่งปุ่ม Streamlit ให้สวยขึ้น */
    div.stButton > button {{
        background-color: #8c7ae6; 
        color: white;
        border-radius: 50px;
        padding: 12px 30px;
        font-weight: 600;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    div.stButton > button:hover {{
        background-color: #7b6ac4;
        color: white;
    }}

</style>

<div class="navbar">
    <div style="font-size: 1.4rem; color: #885D95; font-weight:700;">🧬 Parkinson AI</div>
    <div>
        <a href="#test_area" style="text-decoration:none; color:#885D95; font-weight:600;">เริ่มใช้งาน</a>
    </div>
</div>

<div class="hero-section">
    <div class="hero-content">
        <div class="hero-title">“Early detection changes everything.”</div>
        <div class="hero-sub">ใช้ AI ตรวจคัดกรองพาร์กินสันเบื้องต้น แม่นยำ รวดเร็ว และรู้ผลทันที<br>เพียงแค่วาดเส้น หรืออัปโหลดรูปภาพ</div>
        <a href="#test_area" class="cta-button">เริ่มทำแบบทดสอบ ➝</a>
    </div>
    
    <div class="hero-image-container">
        <img src="data:image/svg+xml;base64,{img_base64}" class="hero-img">
    </div>
</div>

<div id="test_area"></div>

""", unsafe_allow_html=True)

# ----------------------------------
# 3. Streamlit Logic (ส่วนทำงาน)
# ----------------------------------

# ใช้ Container ครอบเพื่อให้เนื้อหาอยู่กึ่งกลาง
with st.container():
    st.markdown('<div class="app-container">', unsafe_allow_html=True)

    # Load Model
    @st.cache_resource
    def load_spiral_model():
        try:
            return tf.keras.models.load_model("(Test_naja)effnet_parkinson_model.keras")
        except:
            return None

    spiral_model = load_spiral_model()

    if spiral_model is None:
        st.warning("⚠️ ไม่พบไฟล์โมเดล .keras ระบบจะทำงานเฉพาะส่วนวาดภาพ")

    def preprocess(img):
        img = np.array(img.convert("RGB"))
        img = cv2.resize(img, (256, 256)) 
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # ================= BOX 1 : SPIRAL =================
    st.subheader("1. 🌀 Spiral (ขดลวด)")

    spiral_mode = st.radio("เลือกวิธีใส่ภาพ (Spiral)", ["Upload", "Draw"], horizontal=True, key="spiral_mode")
    spiral_image = None

    if spiral_mode == "Upload":
        spiral_file = st.file_uploader("อัปโหลด Spiral", type=["png", "jpg", "jpeg"], key="spiral_upload")
        if spiral_file:
            spiral_image = Image.open(spiral_file).convert("RGB")
            st.image(spiral_image, caption="Spiral Preview", width=300)
    else: # Draw
        st.caption("วาดภาพขดลวดลงในกรอบด้านล่าง:")
        spiral_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=6,
            stroke_color="black",
            background_color="#f9f9f9", # ใส่สีพื้นหลังให้อ่อนๆ จะได้เห็นขอบเขต
            height=300,
            width=500,
            drawing_mode="freedraw",
            key="spiral_draw"
        )
        if spiral_canvas.image_data is not None:
            # ตรวจสอบว่ามีการวาดเส้นหรือไม่ (ป้องกันภาพว่าง)
            if np.sum(spiral_canvas.image_data) > 0: 
                spiral_image = Image.fromarray(spiral_canvas.image_data.astype("uint8")).convert("RGB")

    spiral_result_box = st.empty()
    st.markdown("---")

    # ================= BOX 2 : WAVE =================
    st.subheader("2. 🌊 Wave (คลื่น)")

    wave_mode = st.radio("เลือกวิธีใส่ภาพ (Wave)", ["Upload", "Draw"], horizontal=True, key="wave_mode")
    wave_image = None

    if wave_mode == "Upload":
        wave_file = st.file_uploader("อัปโหลด Wave", type=["png", "jpg", "jpeg"], key="wave_upload")
        if wave_file:
            wave_image = Image.open(wave_file).convert("RGB")
            st.image(wave_image, caption="Wave Preview", width=300)
    else: # Draw
        st.caption("วาดภาพคลื่นลงในกรอบด้านล่าง:")
        wave_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=6,
            stroke_color="black",
            background_color="#f9f9f9",
            height=300,
            width=500,
            drawing_mode="freedraw",
            key="wave_draw"
        )
        if wave_canvas.image_data is not None:
             if np.sum(wave_canvas.image_data) > 0:
                wave_image = Image.fromarray(wave_canvas.image_data.astype("uint8")).convert("RGB")

    wave_result_box = st.empty()
    st.markdown("---")

    # ================= BUTTON & PROCESS =================
    if st.button("🔍 ประมวลผลทั้งหมด"):
        
        # --- Process Spiral ---
        if spiral_image is not None and spiral_model is not None:
            try:
                input_tensor = preprocess(spiral_image)
                pred = spiral_model.predict(input_tensor)[0][0]

                if pred > 0.5:
                    spiral_result_box.error(f"🌀 Spiral: มีความเสี่ยง Parkinson (Confidence: {pred:.3f})")
                else:
                    spiral_result_box.success(f"🌀 Spiral: ปกติ (Confidence: {pred:.3f})")
            except Exception as e:
                spiral_result_box.error(f"Error: {e}")
        elif spiral_image is None:
            spiral_result_box.warning("🌀 Spiral: กรุณาใส่ภาพก่อน")

        # --- Process Wave ---
        if wave_image is not None:
            wave_result_box.info("🌊 Wave: ระบบกำลังพัฒนาโมเดล")
        else:
            wave_result_box.warning("🌊 Wave: กรุณาใส่ภาพก่อน")

    st.markdown('</div>', unsafe_allow_html=True) # ปิด div app-container
