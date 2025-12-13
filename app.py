import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import base64

# ----------------------------------
# Page Config (Mobile First)
# ----------------------------------
st.set_page_config(page_title="Parkinson Tester", layout="centered")
st.markdown("""
    <style>
    .header {
        font-family: 'Times New Roman', Times, serif;
        font-size: 80px !important;
        font-weight: 500;
        color: #000000;
        text-align: center;
        margin-bottom: 10px;
        margin-top: -50px;
    }
    .blue-container {
        background-color: #FCECE4;
        background-image: linear-gradient(to right, #FCECE4, #E8F0F8); 
        padding: 40px;            
        border-radius: 20px;    
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        display: flex;
        justify-content: center;
    }
    .inner-image {
        width: 100%;
        max-width: 700px;      
        border-radius: 15px;
        display: block;
    }
    .stApp {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<p class="header">Parkinson Detection</p>', unsafe_allow_html=True)
img_cover = "parkinson cover.svg"
st.markdown(f""
    <div class="ipad-container">
        <img src="{image_cover}" class="inner-img" >
    </div>
    "", unsafe_allow_html=True)
        
st.divider()

# ----------------------------------
# Load Spiral Model
# ----------------------------------
@st.cache_resource
def load_spiral_model():
    return tf.keras.models.load_model("(Test_naja)effnet_parkinson_model.keras")

spiral_model = load_spiral_model()

# ----------------------------------
# Preprocess (256x256 ตามโมเดล)
# ----------------------------------
def preprocess(img):
    img = np.array(img.convert("RGB"))
    img = cv2.resize(img, (256, 256))   # ✅ สำคัญมาก
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================================================
# =====================  BOX 1 : SPIRAL  ==================
# =========================================================
st.subheader("🌀 Spiral")

spiral_mode = st.radio(
    "เลือกวิธีใส่ภาพ (Spiral)",
    ["Upload", "Draw"],
    horizontal=True,
    key="spiral_mode"
)

spiral_image = None

if spiral_mode == "Upload":
    spiral_file = st.file_uploader(
        "อัปโหลด Spiral",
        type=["png", "jpg", "jpeg"],
        key="spiral_upload"
    )
    if spiral_file:
        spiral_image = Image.open(spiral_file).convert("RGB")
        st.image(
            spiral_image,
            caption="Spiral Preview",
            use_container_width=True
        )

else:  # Draw Mode
    spiral_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=6,
        stroke_color="black",
        background_color="white",
        height=300,
        width=500,     # ✅ แนวนอน
        drawing_mode="freedraw",
        key="spiral_draw"
    )
    if spiral_canvas.image_data is not None:
        spiral_image = Image.fromarray(
            spiral_canvas.image_data.astype("uint8")
        ).convert("RGB")

# ✅ ช่องแสดงผล Spiral (อยู่ก่อน divider)
spiral_result_box = st.empty()

st.divider()

# =========================================================
# =====================  BOX 2 : WAVE  =====================
# =========================================================
st.subheader("🌊 Wave")

wave_mode = st.radio(
    "เลือกวิธีใส่ภาพ (Wave)",
    ["Upload", "Draw"],
    horizontal=True,
    key="wave_mode"
)

wave_image = None

if wave_mode == "Upload":
    wave_file = st.file_uploader(
        "อัปโหลด Wave",
        type=["png", "jpg", "jpeg"],
        key="wave_upload"
    )
    if wave_file:
        wave_image = Image.open(wave_file).convert("RGB")
        st.image(
            wave_image,
            caption="Wave Preview",
            use_container_width=True
        )

else:  # Draw Mode
    wave_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=6,
        stroke_color="black",
        background_color="white",
        height=300,
        width=500,     # ✅ แนวนอน
        drawing_mode="freedraw",
        key="wave_draw"
    )
    if wave_canvas.image_data is not None:
        wave_image = Image.fromarray(
            wave_canvas.image_data.astype("uint8")
        ).convert("RGB")

# ✅ ช่องแสดงผล Wave (อยู่ก่อน divider)
wave_result_box = st.empty()

st.divider()

# =========================================================
# =====================  PROCESS BUTTON  ==================
# =========================================================
if st.button("🔍 ประมวลผลทั้งหมด",use_container_width=True):

    # ---------- Spiral Prediction ----------
    if spiral_image is not None:
        try:
            input_tensor = preprocess(spiral_image)
            pred = spiral_model.predict(input_tensor)[0][0]

            if pred > 0.5:
                spiral_result_box.error(
                    f"🌀 Spiral : เสี่ยง Parkinson ({pred:.3f})"
                )
            else:
                spiral_result_box.success(
                    f"🌀 Spiral : ปกติ ({pred:.3f})"
                )
        except Exception as e:
            spiral_result_box.error(f"เกิดข้อผิดพลาดในการประมวลผล Spiral: {e}")
    else:
        spiral_result_box.warning("🌀 Spiral : ยังไม่ได้ใส่ภาพ")

    # ---------- Wave Status Only ----------
    if wave_image is not None:
        wave_result_box.info(
            "🌊 Wave : มีภาพแล้ว แต่ยังไม่มีโมเดลสำหรับประมวลผล"
        )
    else:
        wave_result_box.warning("🌊 Wave : ยังไม่ได้ใส่ภาพ")
