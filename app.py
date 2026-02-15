import streamlit as st
import pandas as pd
from PIL import Image
import io
import torch
import clip
import logging
from typing import List, Tuple, Dict

# --- 0. CAU HINH HE THONG (SYSTEM CONFIG) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cac gioi han he thong
MAX_IMAGES = 50
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
THUMBNAIL_SIZE = (300, 300)
CLIP_INPUT_SIZE = (224, 224)

# --- 1. THIET LAP GIAO DIEN & CSS (UI/UX) ---
st.set_page_config(
    page_title="AI Master V10 - Hashtag Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    div[data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        height: 3em;
    }
    /* Button Primary (Green) */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #1e6b41 !important;
        box-shadow: 0 4px 8px rgba(33, 115, 70, 0.4);
    }
    /* Download Button (Green) */
    div[data-testid="stDownloadButton"] > button {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
    }
    div.stSelectbox > label {
        font-weight: 600;
        color: #333;
    }
    div.stTextInput > label {
        font-weight: 600;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 AI MASTER V10 - HASHTAG PRO")
st.markdown("#### Quy trinh toi uu: Object -> Style -> Color -> Mood -> Gender")
st.markdown("---")

# --- 2. DU LIEU PHAN LOAI (DATASET) ---
# Dữ liệu gốc dùng cho AI train/predict (Không có None)
AI_STYLES = [
    "2D", "3D", "Cute", "Animeart", "Realism",
    "Aesthetic", "Cool", "Fantasy", "Comic", "Horror",
    "Cyberpunk", "Lofi", "Minimalism", "Digitalart", "Cinematic",
    "Pixelart", "Scifi", "Vangoghart"
]

AI_COLORS = [
    "Black", "White", "Blackandwhite", "Red", "Yellow",
    "Blue", "Green", "Pink", "Orange", "Pastel",
    "Hologram", "Vintage", "Colorful", "Neutral", "Light",
    "Dark", "Warm", "Cold", "Neon", "Gradient",
    "Purple", "Brown", "Grey"
]

# Dữ liệu hiển thị UI (Có thêm None)
UI_STYLES = ["None"] + AI_STYLES
UI_COLORS = ["None"] + AI_COLORS
UI_MOODS = ["None", "Happy", "Sad", "Lonely", "Lovely", "Funny", "ZenMode"]
UI_GENDERS = ["None", "Male", "Female", "Non-binary", "Unisex"]

# --- 3. KHOI DONG AI ENGINE ---
@st.cache_resource
def load_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"System running on: {device}")
    
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Tao vector cho Style va Color (AI chi doan 2 cai nay)
        s_prompts = [f"a {s} style artwork" for s in AI_STYLES]
        c_prompts = [f"dominant color is {c}" for c in AI_COLORS]
        
        s_vectors = clip.tokenize(s_prompts).to(device)
        c_vectors = clip.tokenize(c_prompts).to(device)
        
        with torch.no_grad():
            s_feat = model.encode_text(s_vectors)
            c_feat = model.encode_text(c_vectors)
            s_feat /= s_feat.norm(dim=-1, keepdim=True)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)
            
        return model, preprocess, s_feat, c_feat, device
    except Exception as e:
        logger.error(f"Critical Error - Model Load Failed: {e}")
        raise e

try:
    with st.spinner("⏳ Dang khoi dong he thong AI..."):
        model, preprocess, s_feat, c_feat, device = load_engine()
except Exception as e:
    st.error(f"Loi he thong: {e}")
    st.stop()

# --- 4. HAM XU LY ANH (LOGIC) ---
def process_single_image(file_obj) -> Dict:
    try:
        file_bytes = file_obj.getvalue()
        original_img = Image.open(io.BytesIO(file_bytes))
        
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
            
        thumb = original_img.copy()
        thumb.thumbnail(THUMBNAIL_SIZE)
        
        input_img = original_img.resize(CLIP_INPUT_SIZE)
        img_input = preprocess(input_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feat = model.encode_image(img_input)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        # AI du doan Style va Color
        s_idx = (100.0 * img_feat @ s_feat.T).softmax(dim=-1).argmax().item()
        c_idx = (100.0 * img_feat @ c_feat.T).softmax(dim=-1).argmax().item()
        
        predicted_style = AI_STYLES[s_idx]
        predicted_color = AI_COLORS[c_idx]
        
        return {
            "status": "ok",
            "filename": file_obj.name,
            "image_obj": thumb,
            "object": "",            # Mac dinh rong
            "style": predicted_style, # AI doan
            "color": predicted_color, # AI doan
            "mood": "None",          # Mac dinh None
            "gender": "None"         # Mac dinh None
        }
    except Exception as e:
        logger.error(f"Error processing {file_obj.name}: {e}")
        return {"status": "error", "filename": file_obj.name, "msg": str(e)}

def display_image_editor(idx: int, item: Dict, start_num: int):
    with st.container(border=True):
        # Hien thi anh va ten
        c_img, c_info = st.columns([1, 2])
        with c_img:
            st.image(item["image_obj"], use_container_width=True)
            st.caption(f"#{start_num + idx}")
            
        with c_info:
            st.write(f"**{item['filename']}**")
            
            # 1. Object (Text Input)
            new_obj = st.text_input("Object (Doi tuong)", value=item["object"], key=f"obj_{idx}", placeholder="VD: Girl, Cat, House...")
            
            c1, c2 = st.columns(2)
            with c1:
                # 2. Style
                curr_style = item["style"] if item["style"] in UI_STYLES else "None"
                new_s = st.selectbox("Style", UI_STYLES, index=UI_STYLES.index(curr_style), key=f"s_{idx}")
                
                # 4. Mood
                curr_mood = item["mood"] if item["mood"] in UI_MOODS else "None"
                new_m = st.selectbox("Mood", UI_MOODS, index=UI_MOODS.index(curr_mood), key=f"m_{idx}")
                
            with c2:
                # 3. Color
                curr_color = item["color"] if item["color"] in UI_COLORS else "None"
                new_c = st.selectbox("Color", UI_COLORS, index=UI_COLORS.index(curr_color), key=f"c_{idx}")
                
                # 5. Gender
                curr_gender = item["gender"] if item["gender"] in UI_GENDERS else "None"
                new_g = st.selectbox("Gender", UI_GENDERS, index=UI_GENDERS.index(curr_gender), key=f"g_{idx}")

        # Cap nhat Session State
        st.session_state["results"][idx]["object"] = new_obj
        st.session_state["results"][idx]["style"] = new_s
        st.session_state["results"][idx]["color"] = new_c
        st.session_state["results"][idx]["mood"] = new_m
        st.session_state["results"][idx]["gender"] = new_g

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Bang Dieu Khien")
    st.info("💡 **Huong dan:** \n1. Tai anh -> AI doan Style/Color.\n2. Nhap Object, chon Mood/Gender.\n3. Xuat Excel.")
    
    start_idx = st.number_input("STT bat dau:", value=1, step=1, min_value=1)
    
    uploaded_files = st.file_uploader(
        f"Tai anh len (Toi da {MAX_IMAGES}):", 
        type=['png','jpg','jpeg','webp'], 
        accept_multiple_files=True
    )
    
    analyze_btn = st.button("🚀 BAT DAU PHAN TICH", type="primary")
    
    st.markdown("---")
    if st.button("🔄 Lam moi he thong"):
        st.session_state.clear()
        st.rerun()

# --- 6. MAIN LOGIC ---
if "results" not in st.session_state:
    st.session_state["results"] = []

if analyze_btn and uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.error(f"⚠️ Qua so luong cho phep ({MAX_IMAGES}).")
        st.stop()
        
    temp_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        if file.size > MAX_FILE_SIZE_BYTES:
            st.warning(f"⚠️ Bo qua: {file.name} (>10MB)")
            continue
            
        status_text.text(f"Dang xu ly: {file.name} ({i+1}/{total_files})...")
        res = process_single_image(file)
        
        if res["status"] == "ok":
            res["id"] = i
            temp_results.append(res)
        else:
            st.warning(f"⚠️ Loi: {res['filename']}")
            
        progress_bar.progress((i+1)/total_files)
    
    st.session_state["results"] = temp_results
    status_text.success(f"✅ Xong! Hay review va dien thong tin ben duoi.")
    progress_bar.empty()

# --- 7. EXPORT & DISPLAY ---
if st.session_state["results"]:
    st.divider()
    
    # --- LOGIC XUAT EXCEL ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"📊 Review & Chinh sua ({len(st.session_state['results'])} anh)")
    with c2:
        export_data = []
        for i, item in enumerate(st.session_state["results"]):
            # Logic gop Hashtag: Object -> Style -> Color -> Mood -> Gender
            tags = []
            if item["object"].strip(): tags.append(item["object"].strip())
            if item["style"] != "None": tags.append(item["style"])
            if item["color"] != "None": tags.append(item["color"])
            if item["mood"] != "None": tags.append(item["mood"])
            if item["gender"] != "None": tags.append(item["gender"])
            
            final_string = ", ".join(tags)
            
            export_data.append({
                "STT": start_idx + i,
                "Ten file": item["filename"],
                "Final Prompt": final_string, # Cot gop quan trong nhat
                "Object": item["object"],
                "Style": item["style"],
                "Color": item["color"],
                "Mood": item["mood"],
                "Gender": item["gender"]
            })
            
        df = pd.DataFrame(export_data)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
            worksheet = writer.sheets['Sheet1']
            # Format cot
            worksheet.set_column(0, 0, 5)  # STT
            worksheet.set_column(1, 1, 25) # Ten file
            worksheet.set_column(2, 2, 50) # Final Prompt (Rong hon)
            worksheet.set_column(3, 7, 15) # Cac cot thanh phan
            
        st.download_button(
            label="📥 TAI FILE EXCEL",
            data=buffer.getvalue(),
            file_name="hashtags_v10.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Grid layout 2 cot de tiet kiem dien tich
    cols = st.columns(2)
    for i, item in enumerate(st.session_state["results"]):
        with cols[i % 2]: 
            display_image_editor(i, item, start_idx)

elif not uploaded_files:
    st.info("👈 Tai anh len de bat dau quy trinh V10.")
