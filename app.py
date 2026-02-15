import streamlit as st
import pandas as pd
from PIL import Image
import io
import torch
import open_clip
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
    page_title="AI Master V10.1 - Hashtag Pro",
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
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
    }
    div[data-testid="stDownloadButton"] > button {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 AI MASTER V10.1 - HASHTAG PRO")
st.markdown("#### Quy trinh toi uu: Object -> Style -> Color -> Mood -> Gender")
st.markdown("---")

# --- 2. DU LIEU PHAN LOAI (DATASET) ---
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
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        s_prompts = [f"a {s} style artwork" for s in AI_STYLES]
        c_prompts = [f"dominant color is {c}" for c in AI_COLORS]
        
        s_vectors = tokenizer(s_prompts).to(device)
        c_vectors = tokenizer(c_prompts).to(device)
        
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
    with st.spinner("⏳ Dang khoi dong he thong OpenCLIP..."):
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
        
        input_img = preprocess(original_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feat = model.encode_image(input_img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        s_idx = (100.0 * img_feat @ s_feat.T).softmax(dim=-1).argmax().item()
        c_idx = (100.0 * img_feat @ c_feat.T).softmax(dim=-1).argmax().item()
        
        return {
            "status": "ok",
            "filename": file_obj.name,
            "image_obj": thumb,
            "object": "",            
            "style": AI_STYLES[s_idx], 
            "color": AI_COLORS[c_idx], 
            "mood": "None",          
            "gender": "None"         
        }
    except Exception as e:
        logger.error(f"Error processing {file_obj.name}: {e}")
        return {"status": "error", "filename": file_obj.name, "msg": str(e)}

def display_image_editor(idx: int, item: Dict, start_num: int):
    with st.container(border=True):
        c_img, c_info = st.columns([1, 2])
        with c_img:
            st.image(item["image_obj"], use_container_width=True)
            st.caption(f"#{start_num + idx}")
            
        with c_info:
            st.write(f"**{item['filename']}**")
            # QUAN TRONG: Lay gia tri tu input cap nhat ngay vao session_state
            new_obj = st.text_input("Object", value=item["object"], key=f"obj_{idx}")
            
            c1, c2 = st.columns(2)
            with c1:
                curr_style = item["style"] if item["style"] in UI_STYLES else "None"
                new_s = st.selectbox("Style", UI_STYLES, index=UI_STYLES.index(curr_style), key=f"s_{idx}")
                
                curr_mood = item["mood"] if item["mood"] in UI_MOODS else "None"
                new_m = st.selectbox("Mood", UI_MOODS, index=UI_MOODS.index(curr_mood), key=f"m_{idx}")
                
            with c2:
                curr_color = item["color"] if item["color"] in UI_COLORS else "None"
                new_c = st.selectbox("Color", UI_COLORS, index=UI_COLORS.index(curr_color), key=f"c_{idx}")
                
                curr_gender = item["gender"] if item["gender"] in UI_GENDERS else "None"
                new_g = st.selectbox("Gender", UI_GENDERS, index=UI_GENDERS.index(curr_gender), key=f"g_{idx}")

        # CAP NHAT DU LIEU (Commit changes)
        st.session_state["results"][idx]["object"] = new_obj
        st.session_state["results"][idx]["style"] = new_s
        st.session_state["results"][idx]["color"] = new_c
        st.session_state["results"][idx]["mood"] = new_m
        st.session_state["results"][idx]["gender"] = new_g

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Bang Dieu Khien")
    start_idx = st.number_input("STT bat dau:", value=1, step=1, min_value=1)
    
    uploaded_files = st.file_uploader(
        f"Tai anh len (Toi da {MAX_IMAGES}):", 
        type=['png','jpg','jpeg','webp'], 
        accept_multiple_files=True
    )
    
    analyze_btn = st.button("🚀 BAT DAU PHAN TICH", type="primary")
    
    if st.button("🔄 Lam moi he thong"):
        st.session_state.clear()
        st.rerun()

# --- 6. MAIN LOGIC ---
if "results" not in st.session_state:
    st.session_state["results"] = []

if analyze_btn and uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.error(f"⚠️ Qua so luong ({MAX_IMAGES}).")
        st.stop()
        
    temp_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Dang xu ly: {file.name} ({i+1}/{total_files})...")
        res = process_single_image(file)
        if res["status"] == "ok":
            res["id"] = i
            temp_results.append(res)
        progress_bar.progress((i+1)/total_files)
    
    st.session_state["results"] = temp_results
    status_text.success(f"✅ Xong! Review ben duoi.")
    progress_bar.empty()

# --- 7. EXPORT & DISPLAY (DA TOI UU THU TU) ---
if st.session_state["results"]:
    st.divider()
    
    # 1. Tao khung chua cho nut download (De no hien thi o tren cung)
    download_container = st.container()
    
    # 2. Hien thi Editor va Cap nhat Du lieu (Chay truoc de lay data moi nhat)
    cols = st.columns(2)
    for i, item in enumerate(st.session_state["results"]):
        with cols[i % 2]: 
            display_image_editor(i, item, start_idx)

    # 3. Xu ly va Hien thi nut Download (Chay sau cung nhung hien thi o tren cung)
    with download_container:
        c1, c2 = st.columns([3, 1])
        with c1: st.subheader(f"📊 Review ({len(st.session_state['results'])} anh)")
        with c2:
            export_data = []
            for i, item in enumerate(st.session_state["results"]):
                tags = []
                # Object duoc lay truc tiep tu session_state da cap nhat
                obj_text = item["object"].strip()
                if obj_text: tags.append(obj_text)
                
                if item["style"] != "None": tags.append(item["style"])
                if item["color"] != "None": tags.append(item["color"])
                if item["mood"] != "None": tags.append(item["mood"])
                if item["gender"] != "None": tags.append(item["gender"])
                
                final_string = ", ".join(tags)
                
                export_data.append({
                    "STT": start_idx + i,
                    "Ten file": item["filename"],
                    "Final Prompt": final_string,
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
                worksheet.set_column(0, 0, 5)
                worksheet.set_column(1, 1, 25)
                worksheet.set_column(2, 2, 50)
                
            st.download_button(
                label="📥 TAI FILE EXCEL",
                data=buffer.getvalue(),
                file_name="hashtags_final.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    st.markdown("<br>", unsafe_allow_html=True)

elif not uploaded_files:
    st.info("👈 Tai anh len de bat dau.")
