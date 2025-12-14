import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas
import os
import time
import base64
import textwrap
import joblib

# ----------------------------------
# 1. Page Config
# ----------------------------------
st.set_page_config(page_title="Parkinson Tester", layout="wide", initial_sidebar_state="collapsed")

if "consent_accepted" not in st.session_state:
    st.session_state.consent_accepted = False

query_params = st.query_params
is_started = query_params.get("start") == "true"

# ----------------------------------
# Helper Function
# ----------------------------------
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# --- [ฟังก์ชันช่วยแกะ Model จาก Dictionary] (สำคัญมาก ห้ามลบ) ---
def extract_model_from_dict(loaded_object, model_name="Model"):
    # ถ้าสิ่งที่โหลดมาเป็นโมเดลเลย (มีคำสั่ง predict) ให้ส่งคืนค่าเดิม
    if hasattr(loaded_object, "predict"):
        return loaded_object
    
    # ถ้าเป็น Dictionary ให้พยายามหา Key ที่น่าจะเป็นโมเดล
    if isinstance(loaded_object, dict):
        possible_keys = ['model', 'classifier', 'clf', 'estimator', 'knn', 'svm', 'pipeline']
        for key in possible_keys:
            if key in loaded_object:
                return loaded_object[key]
        
        # กรณีหา Key ไม่เจอ ให้ลองดึง Value ตัวแรกออกมา (เผื่อฟลุ๊ค)
        if len(loaded_object) > 0:
            return list(loaded_object.values())[0]

    return loaded_object

# --- [ฟังก์ชันแสดงคลิปตัวอย่างแบบ Expander] ---
def show_demo_clip(file_root_name):
    with st.expander(f"🎥 คลิกเพื่อดูตัวอย่างการวาด ({file_root_name})"):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if os.path.exists(f"{file_root_name}.mp4"):
                st.video(f"{file_root_name}.mp4")
                st.caption("ตัวอย่างการวาด")
            elif os.path.exists(f"{file_root_name}.mov"):
                st.video(f"{file_root_name}.mov")
                st.caption("ตัวอย่างการวาด")
            elif os.path.exists(f"{file_root_name}.MOV"):
                st.video(f"{file_root_name}.MOV")
                st.caption("ตัวอย่างการวาด")
            elif os.path.exists(f"{file_root_name}.gif"):
                st.image(f"{file_root_name}.gif", use_container_width=True)
                st.caption("ตัวอย่างการวาด")
            else:
                st.info(f"💡 (ยังไม่มีไฟล์ตัวอย่าง {file_root_name} ในโฟลเดอร์)")

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

    section[data-testid="stSidebar"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    
    .navbar {
        display: flex !important;
        justify-content: space-between; align-items: center;
        padding: 15px 20px; 
        background-color: #ffffff; 
        border-bottom: 1px solid #eee;
        width: 100%;
        position: relative; z-index: 999;
        margin-top: -60px;
        box-sizing: border-box;
    }
    .nav-links { display: flex; gap: 20px; }
    .nav-links a { font-weight: 600; text-decoration: none; }

    div[data-testid="stExpander"] details > summary {
        background-color: #F5BA9F !important;
        color: black !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid #DF6456 !important;
    }
    div[data-testid="stExpander"] details > summary:hover {
        color: black !important;
        opacity: 0.9;
    }

    .result-card {
        color: white;
        border-radius: 20px;
        padding: 30px;
        margin-top: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        animation: fadeIn 0.8s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-header {
        font-size: 1.8rem;
        font-weight: 700;
        border-bottom: 2px solid rgba(255,255,255,0.4);
        padding-bottom: 15px;
        margin-bottom: 20px;
    }
    .status-box {
        background-color: white;
        border-radius: 12px;
        padding: 15px 20px;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 25px;
        display: flex; align-items: center; gap: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .confidence-wrapper { margin-bottom: 20px; }
    .progress-track {
        background-color: rgba(255,255,255,0.4);
        height: 12px;
        border-radius: 6px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background-color: #fff;
        border-radius: 6px;
    }
    .result-label { font-weight: 600; font-size: 1.2rem; margin-top: 15px; margin-bottom: 5px; color: #f0f0f0; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
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

    @media (min-width: 992px) {
        .hero-title { font-size: 4rem !important; }
        .hero-sub { font-size: 1.6rem !important; }
        .cta-button { font-size: 1.6rem !important; padding: 20px 70px; }
        div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size: 2.5rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] p, label { font-size: 1.5rem !important; }
        div[data-testid="stCanvas"] button { width: 60px !important; height: 60px !important; transform: scale(1.4); margin: 10px 15px !important; }
        .nav-links a { font-size: 1.4rem; }
    }
    @media (max-width: 991px) {
        .hero-title { font-size: 2.2rem !important; }
        .hero-sub { font-size: 1.1rem !important; }
        .cta-button { font-size: 1.1rem !important; padding: 12px 30px; }
        div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size: 1.6rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] p, label { font-size: 1.1rem !important; }
        div[data-testid="stCanvas"] button { width: 40px !important; height: 40px !important; transform: scale(1.0); margin: 5px !important; }
        .navbar { flex-direction: column; gap: 10px; padding: 10px; }
        .nav-links a { font-size: 1rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] { padding: 20px !important; }
    }

    canvas { max-width: 100% !important; height: auto !important; border: 1px solid #ddd; border-radius: 8px; }
    div[data-testid="stCanvas"] { display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; }

    .hero-purple-container {
        background-color: #885D95; width: 100%; padding: 60px 20px; margin-bottom: 40px; 
        text-align: center; color: white; display: flex; flex-direction: column; align-items: center;
        box-sizing: border-box;
    }
    .hero-title { font-weight: 700; margin-bottom: 15px; color: white !important; }
    .hero-sub { font-weight: 300; margin-bottom: 25px; max-width: 800px; color: #f0f0f0 !important; }
    
    .cta-button {
        background-color: white; color: #885D95 !important; border-radius: 50px; font-weight: 700; text-decoration: none;
        display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.2); cursor: pointer;
    }
    .cta-button:hover { transform: translateY(-5px); background-color: #f8f8f8; }
    
    .about-section { 
        background-color: #67ACC3; 
        width: 100%; 
        padding: 60px 20px; 
        color: white; 
        display: flex; 
        justify-content: center; 
        box-sizing: border-box; 
        overflow-x: hidden; 
    }
    .about-container { max-width: 1200px; width: 100%; box-sizing: border-box; }
    .about-header-large { font-size: 2.8rem; font-weight: 700; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 20px; margin-bottom: 40px; }
    
    .about-body-grid { 
        display: grid; 
        grid-template-columns: 1fr; 
        gap: 40px; 
        align-items: center; 
        width: 100%;
    }
    
    @media (min-width: 992px) {
        .about-body-grid { grid-template-columns: 1fr 1.2fr; }
        .about-text-content { font-size: 1.35rem !important; text-align: left; }
        .about-image-container { text-align: center; }
        .about-img-responsive { max-width: 100%; }
        .quote-box { font-size: 1.6rem !important; }
    }
    @media (max-width: 991px) {
        .about-header-large { font-size: 2rem; }
        .about-text-content { font-size: 1.1rem !important; text-align: justify; }
        .about-image-container { text-align: center; margin-bottom: 20px; }
        .about-img-responsive { max-width: 80%; }
    }

    .about-img-responsive { height: auto; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); border: 4px solid rgba(255, 255, 255, 0.3); }
    .about-text-content { line-height: 1.8; font-weight: 300; }
    .quote-box {
        background-color: rgba(255, 255, 255, 0.15); border-left: 6px solid #ffffff; padding: 30px; margin-top: 50px;
        border-radius: 10px; font-size: 1.3rem; font-style: italic; font-weight: 500; line-height: 1.6;
        text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); width: 100%; grid-column: 1 / -1;
        box-sizing: border-box;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff !important; border: 1px solid #E0D0E8 !important; border-radius: 20px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important; margin-bottom: 30px; }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 { color: #885D95 !important; text-align: center !important; font-weight: 700 !important; }
    div.stButton > button[kind="primary"] { background-color: #DF6456 !important; border: none !important; color: white !important; height: auto; padding: 15px; width: 100%; font-size: 1.3rem; border-radius: 10px; }
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
        <a href="#about_area" style="color:#67ACC3;">About Parkinson</a>
        <a href="?start=true" target="_self" style="color:#885D95;">Take the test</a>
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
# --- LOAD SPIRAL MODEL (แก้ไข: รองรับทั้ง Model และ Dict) ---
@st.cache_resource
def load_spiral_model():
    if os.path.exists("model_spiral_final_production.joblib"):
        loaded = joblib.load("model_spiral_final_production.joblib")
        return extract_model_from_dict(loaded, "Spiral")
    return None
spiral_model = load_spiral_model()

# --- LOAD WAVE MODEL (แก้ไข: รองรับทั้ง Model และ Dict) ---
@st.cache_resource
def load_wave_model():
    if os.path.exists("model_wave_final_production.joblib"):
        loaded = joblib.load("model_wave_final_production.joblib")
        return extract_model_from_dict(loaded, "Wave")
    return None
wave_model = load_wave_model()

# --- เพิ่มฟังก์ชัน HOG ---
def HOG_img(img):
    hog_img = hog(img,
                orientations=9,            # 9 ทิศทาง
                pixels_per_cell=(12, 12),    # ขนาดของช่อง ยิ่งค่าน้อยยิ่งละเอียด
                cells_per_block=(2, 2),    # รวมกลุ่มกัน 2*2 ช่อง เพื่อปรับแสง
                block_norm='L2-Hys',           # Normalization using L1-norm.
                feature_vector=True)       # Return the data as a feature vector
    return hog_img

# --- ปรับแก้ฟังก์ชัน Preprocess เพื่อใช้ Threshold + HOG ---
def preprocess(img_pil):
    # 1. แปลง PIL Image เป็น Numpy Array
    # img_pil.convert("RGB") จะได้ RGB
    img = np.array(img_pil.convert("RGB"))
    
    # 2. แปลง RGB เป็น BGR (เพื่อให้เหมือน cv2.imread ในตอนเทรน)
    # เพราะ cv2.imread อ่านสีเป็น BGR แต่ PIL อ่านเป็น RGB
    img = img[:, :, ::-1].copy() 
    
    # 3. แปลงเป็น Grayscale (ตามโค้ดเทรน)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 4. Resize เป็น 200x200 (ตามโค้ดเทรน)
    img = cv2.resize(img, (200, 200))
    
    # 5. Threshold ด้วย OTSU (ตามโค้ดเทรนเป๊ะๆ)
    # [1] คือเอาเฉพาะตัวภาพผลลัพธ์
    img = cv2.threshold(img, 
                        0, 
                        255, 
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 6. ส่งเข้า HOG
    # (สมมติว่าตอนเทรนคุณเอาภาพจาก Data.append(img) ไปเข้า HOG ต่อ)
    feature_vector = HOG_img(img)
    
    # Return ทั้ง Vector (เข้าโมเดล) และ รูปภาพ (ไว้แสดง Debug)
    return feature_vector.reshape(1, -1), img
# =========================================================
# 5. TEST AREA
# =========================================================
if is_started or st.session_state.consent_accepted:

    st.markdown('<div id="test_content_anchor" style="padding-top: 20px;"></div>', unsafe_allow_html=True)

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
        c1, c2, c3 = st.columns([1, 8, 1]) 
        with c2:
           with st.container(border=True):
                st.markdown('<div class="disclaimer-header"><h3 style="text-align:center;">⚠️ ข้อควรทราบก่อนทำการทดสอบ</h3></div>', unsafe_allow_html=True)
                st.write("ระบบนี้เป็นเครื่องมือคัดกรองเบื้องต้นโดยใช้ปัญญาประดิษฐ์ (AI)")
                st.error("ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ผู้เชี่ยวชาญได้")
                st.write("หากมีอาการผิดปกติหรือมีความกังวล กรุณาปรึกษาแพทย์เพื่อรับการตรวจเพิ่มเติม")
                st.markdown("---")
                
                st.markdown("""
                <div style="font-size: 1.1rem !important; font-weight: 600; margin-bottom: 10px;">📝 คำแนะนำเพื่อให้ผลลัพธ์แม่นยำขึ้น</div>
                <ul style="margin-bottom: 20px; line-height: 1.6; padding-left: 20px;">
                    <li style="font-size: 1.1rem !important;">นั่งในท่าที่สบาย แขนวางบนพื้นราบ</li>
                    <li style="font-size: 1.1rem !important;">ทำจิตใจให้สงบ หลีกเลี่ยงเครื่องดื่มคาเฟอีนหรือสารกระตุ้นก่อนทำการทดสอบ</li>
                    <li style="font-size: 1.1rem !important;">วาดเส้นด้วยความเร็วและแรงกดตามธรรมชาติ</li>
                </ul>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.write("อาการมือสั่นอาจเกิดจากหลายสาเหตุ เช่น ความเครียด ภาวะวิตกกังวล หรือโรคอื่น ๆ ที่ไม่ใช่พาร์กินสัน")
                st.write("ระบบอาจไม่สามารถแยกแยะสาเหตุของอาการมือสั่นได้อย่างสมบูรณ์")
                st.write("ผลลัพธ์จึงควรใช้ประกอบการพิจารณาเท่านั้น")
                st.write("") 
                accepted = st.checkbox("รับทราบและยอมรับตามเงื่อนไขข้างต้น")
                st.write("")
                if st.button("เริ่มทำแบบทดสอบ", disabled=not accepted, type="primary", use_container_width=True):
                    st.session_state.consent_accepted = True
                    st.rerun()
    else:
        st.markdown('<div id="test_area" style="padding-top: 40px;"></div>', unsafe_allow_html=True)
        # SPIRAL CARD
        with st.container(border=True): 
            st.subheader("🌀 Spiral")
            
            # --- [เรียกใช้ฟังก์ชัน Expander สำหรับ Demo] ---
            # วาดเส้นวนออกจากกึ่งกลางด้วยความเร็วสม่ำเสมอ
            st.write("วาดเส้นวนออกจากกึ่งกลางด้วยความเร็วสม่ำเสมอ")
            show_demo_clip("spiral_demo")
            st.markdown("---")
            # --------------------------------

            spiral_mode = st.radio("เลือกวิธีใส่ภาพ (Spiral)", ["Upload", "Draw"], horizontal=True, key="spiral_mode")
            st.markdown("---")
            spiral_image = None
            if spiral_mode == "Upload":
                uc1, uc2, uc3 = st.columns([0.1, 1, 0.1])
                with uc2:
                    spiral_file = st.file_uploader("อัปโหลด Spiral", type=["png", "jpg", "jpeg"], key="spiral_upload")
                    if spiral_file:
                        spiral_image = Image.open(spiral_file).convert("RGB")
                        st.image(spiral_image, caption="Preview", use_container_width=300)
            else:
                spiral_canvas = st_canvas(fill_color="rgba(255, 255, 255, 0)", stroke_width=6, stroke_color="black", background_color="#ffffff", height=500, width=700, drawing_mode="freedraw", key="spiral_draw", display_toolbar=True)
                if spiral_canvas.image_data is not None:
                    spiral_image = Image.fromarray(spiral_canvas.image_data.astype("uint8")).convert("RGB")
            st.markdown("<br>", unsafe_allow_html=True)
            spiral_result_box = st.empty()

        # WAVE CARD
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True): 
            st.subheader("🌊 Wave")
            
            # --- [เรียกใช้ฟังก์ชัน Expander สำหรับ Demo] ---
            # วาดเส้นคลื่นจากบนลงล่างด้วยความเร็วสม่ำเสมอ
            st.write("วาดเส้นคลื่นจากบนลงล่างด้วยความเร็วสม่ำเสมอ")
            show_demo_clip("wave_demo")
            st.markdown("---")
            # -----------------------------

            wave_mode = st.radio("เลือกวิธีใส่ภาพ (Wave)", ["Upload", "Draw"], horizontal=True, key="wave_mode")
            st.markdown("---")
            wave_image = None
            if wave_mode == "Upload":
                uc1, uc2, uc3 = st.columns([0.1, 1, 0.1])
                with uc2:
                    wave_file = st.file_uploader("อัปโหลด Wave", type=["png", "jpg", "jpeg"], key="wave_upload")
                    if wave_file:
                        wave_image = Image.open(wave_file).convert("RGB")
                        st.image(wave_image, caption="Preview", use_container_width=300)
            else:
                wave_canvas = st_canvas(fill_color="rgba(255, 255, 255, 0)", stroke_width=6, stroke_color="black", background_color="#ffffff", height=500, width=700, drawing_mode="freedraw", key="wave_draw", display_toolbar=True)
                if wave_canvas.image_data is not None:
                    wave_image = Image.fromarray(wave_canvas.image_data.astype("uint8")).convert("RGB")
            st.markdown("<br>", unsafe_allow_html=True)
            wave_result_box = st.empty()

    # PROCESS BUTTON
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 ประมวลผลทั้งหมด", type="primary", use_container_width=True):
                
                # --- [LOGIC CHECK] เช็คก่อนว่ามีรูปไหม ---
                if spiral_image is None and wave_image is None:
                    st.warning("⚠️ กรุณาวาดเส้นหรืออัปโหลดรูปภาพอย่างน้อย 1 รายการก่อนกดประมวลผล")
                
                else:
                    # --- HELPER: ฟังก์ชันดึงค่าความน่าจะเป็น ---
                    def get_model_probability(model, input_data):
                        if hasattr(model, "predict_proba"):
                            try:
                                probs = model.predict_proba(input_data)
                                return probs[0][1] 
                            except Exception:
                                pass
    
                        raw_pred = model.predict(input_data)
                        if hasattr(raw_pred, "ndim") and raw_pred.ndim > 1:
                            val = raw_pred[0][0]
                        else:
                            val = raw_pred[0]
                        return float(val)
    
                    # --- PART 1: SPIRAL PROCESSING ---
                    # เช็คตรงนี้: ถ้ามีรูปค่อยทำ ถ้าไม่มีรูปก็ข้ามไปเลย (ไม่ต้องขึ้นเตือนให้รก)
                    if spiral_image is not None: 
                        if spiral_model is not None:
                            try:
                                # Preprocess และ Unpack ค่า
                                input_tensor, processed_img_show = preprocess(spiral_image)
                                
                                with st.expander("🕵️ Debug: สิ่งที่ Spiral Model เห็น"):
                                    st.image(processed_img_show, caption="Processed Image", width=200, clamp=True)
    
                                # ทำนายผล
                                pred = get_model_probability(spiral_model, input_tensor)
                                
                                if pred > 0.5:
                                    card_bg = "#E4C728"
                                    status_text = "⚠️ พบรูปแบบที่อาจสัมพันธ์กับอาการสั่นแบบโรคพาร์กินสัน"
                                    status_color = "#856404"
                                    confidence = pred * 100
                                    desc_text = "ตรวจพบรูปแบบการวาดที่มีความไม่สม่ำเสมอ ซึ่งอาจสัมพันธ์กับความผิดปกติของการควบคุมการเคลื่อนไหว"
                                    rec_list = "<li>ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อรับการตรวจและวินิจฉัยเพิ่มเติม</li><li>สามารถทำการทดสอบซ้ำเมื่ออยู่ในสภาวะที่ผ่อนคลาย</li>"
                                else:
                                    card_bg = "#86B264" 
                                    status_text = "✅ ไม่พบความผิดปกติเด่นชัด (Normal)"
                                    status_color = "#388E3C"
                                    confidence = (1 - pred) * 100
                                    desc_text = "รูปแบบการวาดมีความใกล้เคียงกับกลุ่มตัวอย่างทั่วไป ไม่พบอาการสั่นที่ผิดปกติชัดเจน"
                                    rec_list = "<li>หากยังมีความกังวล หรือผลการทดสอบไม่ชัดเจน สามารถทำการทดสอบซ้ำได้</li><li>ควรทำในสภาวะที่ผ่อนคลาย ไม่เกร็งข้อมือ</li><li>หากผลระบุว่ามีความเสี่ยง ควรปรึกษาแพทย์เพื่อรับการตรวจวินิจฉัยอย่างละเอียด</li>"
                                
                                result_html = textwrap.dedent(f"""
            <div class="result-card" style="background-color: {card_bg};">
                <div class="result-header">🧪 ผลการคัดกรองเบื้องต้น (Spiral Test)</div>
                <div class="status-box" style="color: {status_color};">{status_text}</div>
                <div class="confidence-wrapper">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <span>ระดับความเชื่อมั่นของโมเดล (Confidence)</span><span>{confidence:.1f}%</span>
                    </div>
                    <div class="progress-track"><div class="progress-fill" style="width: {confidence}%;"></div></div>
                </div>
                <div class="result-label">📝 คำอธิบาย:</div>
                <div class="result-text">{desc_text}</div>
                <div class="result-label">💡 คำแนะนำ:</div>
                <ul class="result-list">{rec_list}</ul>
                <div class="disclaimer-small">⚠️ หมายเหตุ: ผลลัพธ์นี้เป็นการคัดกรองเบื้องต้นเท่านั้น <b>ไม่ใช่การวินิจฉัยทางการแพทย์</b> โปรดใช้วิจารณญาณ</div>
            </div>
            """).strip()
                                spiral_result_box.markdown(result_html, unsafe_allow_html=True)
                            except Exception as e: 
                                spiral_result_box.error(f"Error Spiral: {e}")
                        else:
                            spiral_result_box.error("❌ ไม่พบไฟล์โมเดล Spiral")
    
    
                    # --- PART 2: WAVE PROCESSING ---
                    # เช็คตรงนี้: ถ้ามีรูปค่อยทำ ถ้าไม่มีก็ข้าม (ไม่ต้องเตือน)
                    if wave_image is not None:
                        if wave_model is not None:
                            try:
                                # Preprocess และ Unpack ค่า
                                input_tensor_w, processed_img_show_w = preprocess(wave_image)
                                
                                with st.expander("🕵️ Debug: สิ่งที่ Wave Model เห็น"):
                                     st.image(processed_img_show_w, caption="Processed Image", width=200, clamp=True)
    
                                # ทำนายผล
                                pred_w = get_model_probability(wave_model, input_tensor_w)
                                
                                if pred_w > 0.5:
                                    card_bg_w = "#E4C728"
                                    status_text_w = "⚠️ พบรูปแบบที่อาจสัมพันธ์กับอาการสั่นแบบโรคพาร์กินสัน"
                                    status_color_w = "#856404"
                                    confidence_w = pred_w * 100
                                    desc_text_w = "ตรวจพบรูปแบบการวาดที่มีความไม่สม่ำเสมอ ซึ่งอาจสัมพันธ์กับความผิดปกติของการควบคุมการเคลื่อนไหว"
                                    rec_list_w = "<li>ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อรับการตรวจและวินิจฉัยเพิ่มเติม</li><li>สามารถทำการทดสอบซ้ำในสภาวะที่ผ่อนคลาย</li>"
                                else:
                                    card_bg_w = "#86B264" 
                                    status_text_w = "✅ ไม่พบความผิดปกติเด่นชัด (Normal)"
                                    status_color_w = "#388E3C"
                                    confidence_w = (1 - pred_w) * 100
                                    desc_text_w = "รูปแบบการวาดมีความใกล้เคียงกับกลุ่มตัวอย่างทั่วไป ไม่พบอาการสั่นที่ผิดปกติชัดเจน"
                                    rec_list_w = "<li>หากยังมีความกังวล หรือผลการทดสอบไม่ชัดเจน สามารถทำการทดสอบซ้ำได้</li><li>ควรทำในสภาวะที่ผ่อนคลาย ไม่เกร็งข้อมือ</li><li>หากผลระบุว่ามีความเสี่ยง ควรปรึกษาแพทย์เพื่อรับการตรวจวินิจฉัยอย่างละเอียด</li>"
                                
                                result_html_w = textwrap.dedent(f"""
            <div class="result-card" style="background-color: {card_bg_w};">
                <div class="result-header">🧪 ผลการคัดกรองเบื้องต้น (Wave Test)</div>
                <div class="status-box" style="color: {status_color_w};">{status_text_w}</div>
                <div class="confidence-wrapper">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <span>ระดับความเชื่อมั่นของโมเดล (Confidence)</span><span>{confidence_w:.1f}%</span>
                    </div>
                    <div class="progress-track"><div class="progress-fill" style="width: {confidence_w}%;"></div></div>
                </div>
                <div class="result-label">📝 คำอธิบาย:</div>
                <div class="result-text">{desc_text_w}</div>
                <div class="result-label">💡 คำแนะนำ:</div>
                <ul class="result-list">{rec_list_w}</ul>
                <div class="disclaimer-small">⚠️ หมายเหตุ: ผลลัพธ์นี้เป็นการคัดกรองเบื้องต้นเท่านั้น <b>ไม่ใช่การวินิจฉัยทางการแพทย์</b> โปรดใช้วิจารณญาณ</div>
            </div>
            """).strip()
                                wave_result_box.markdown(result_html_w, unsafe_allow_html=True)
                            
                            except Exception as e:
                                wave_result_box.error(f"Error Wave: {e}")
                        else:
                            wave_result_box.info("🌊 มีภาพ Wave แต่ยังไม่มีไฟล์โมเดล Wave")

else:
    pass

# =========================================================
# 6. ABOUT SECTION
# =========================================================
st.markdown('<div id="about_area" style="padding-top: 40px;"></div>', unsafe_allow_html=True) 

img_b64 = get_image_base64("parkinson cover.png")
if img_b64:
    img_tag = f'<img src="data:image/png;base64,{img_b64}" class="about-img-responsive" alt="Parkinson Cover">'
else:
    img_tag = '<div style="background:rgba(255,255,255,0.2); padding:40px; color:white; border-radius:15px; text-align:center; border: 2px dashed white;">⚠️ ไม่พบไฟล์ parkinson cover.png</div>'

about_html = textwrap.dedent(f"""
<div class="about-section">
    <div class="about-container">
        <div class="about-header-large">ทำความรู้จักกับ โรคพาร์กินสัน (Parkinson’s Disease)</div>
        <div class="about-body-grid">
            <div class="about-image-container">{img_tag}</div>
            <div class="about-text-container">
                <div class="about-text-content">
                    โรคพาร์กินสันเป็นโรคความเสื่อมของระบบประสาทที่พบได้บ่อยเป็นอันดับต้น ๆ ของโลก มักพบในผู้ที่มีอายุ 60 ปีขึ้นไป แต่ในปัจจุบันเริ่มพบผู้ป่วยในวัยที่อายุน้อยลงมากขึ้น สาเหตุหลักเกิดจากการเสื่อมของเซลล์สมองที่สร้างสาร โดพามีน (Dopamine) ซึ่งมีบทบาทสำคัญในการควบคุมการเคลื่อนไหวของร่างกาย เมื่อระดับโดพามีนลดลง จะส่งผลให้การเคลื่อนไหวผิดปกติ
                    <br><br>
                    อาการที่พบบ่อย ได้แก่ มือสั่นขณะอยู่นิ่ง การเคลื่อนไหวช้า กล้ามเนื้อแข็งเกร็ง การทรงตัวไม่ดี รวมถึงอาการอื่น ๆ เช่น การรับรู้กลิ่นลดลง ท้องผูก หรือความผิดปกติของการนอนหลับ ซึ่งอาจเกิดขึ้นก่อนอาการสั่น
                </div>
            </div>
            <div class="quote-box">
                “แม้โรคพาร์กินสันจะยังไม่สามารถรักษาให้หายขาดได้ แต่การตรวจพบตั้งแต่ระยะเริ่มต้นจะช่วยให้สามารถควบคุมอาการ ชะลอความเสื่อมของโรค และช่วยให้ผู้ป่วยมีคุณภาพชีวิตที่ดีขึ้น”
            </div>
        </div>
    </div>
</div>
""").strip()

st.markdown(about_html, unsafe_allow_html=True)
