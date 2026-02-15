import streamlit as st
import pandas as pd
from PIL import Image
import io
import torch
import open_clip
import logging
from typing import List, Tuple, Dict

# --- 0. CAU HINH HE THONG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
MAX_IMAGES = 100
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
THUMBNAIL_SIZE = (300, 600) # Tang chieu cao cho thumbnail vi la anh doc
CLIP_INPUT_SIZE = (224, 224)

# --- 1. THIET LAP GIAO DIEN & CSS ---
st.set_page_config(
    page_title="AI Master V11 - Vertical UI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho Anh Doc (Vertical)
st.markdown("""
    <style>
    /* 1. Khung the san pham (Card) */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    
    /* 2. Hinh anh: Bo tron, do bong nhe */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        object-fit: cover; /* Dam bao anh full khung */
    }
    
    /* 3. Nut bam xanh */
    div[data-testid="stButton"] > button[kind="primary"], 
    div[data-testid="stDownloadButton"] > button {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
        font-weight: bold;
    }

    /* 4. Tieu de nho gon hon */
    h3 { font-size: 1.2rem !important; margin-bottom: 0px; }
    
    /* 5. Giam khoang cach giua cac phan tu */
    .stSelectbox { margin-bottom: -10px !important; }
    .stTextInput { margin-bottom: -10px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📱 AI MASTER V11 - VERTICAL OPTIMIZED")
st.markdown("---")

# --- 2. DATASET ---
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

# --- 3. LOAD AI ---
@st.cache_resource
def load_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"System running on: {device}")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        s_vectors = tokenizer([f"a {s} style artwork" for s in AI_STYLES]).to(device)
        c_vectors = tokenizer([f"dominant color is {c}" for c in AI_COLORS]).to(device)
        
        with torch.no_grad():
            s_feat = model.encode_text(s_vectors)
            c_feat = model.encode_text(c_vectors)
            s_feat /= s_feat.norm(dim=-1, keepdim=True)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)
            
        return model, preprocess, s_feat, c_feat, device
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

try:
    with st.spinner("⏳ Loading AI Engine..."):
        model, preprocess, s_feat, c_feat, device = load_engine()
except Exception as e:
    st.error(f"Loi: {e}")
    st.stop()

# --- 4. LOGIC XU LY ---
def process_single_image(file_obj) -> Dict:
    try:
        file_bytes = file_obj.getvalue()
        original_img = Image.open(io.BytesIO(file_bytes))
        if original_img.mode != "RGB": original_img = original_img.convert("RGB")
        
        # Resize thumbnail thong minh cho anh doc
        thumb = original_img.copy()
        thumb.thumbnail(THUMBNAIL_SIZE)
        
        input_img = preprocess(original_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feat = model.encode_image(input_img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        s_idx = (100.0 * img_feat @ s_feat.T).softmax(dim=-1).argmax().item()
        c_idx = (100.0 * img_feat @ c_feat.T).softmax(dim=-1).argmax().item()
        
        return {
            "status": "ok", "filename": file_obj.name, "image_obj": thumb,
            "object": "", "style": AI_STYLES[s_idx], "color": AI_COLORS[c_idx], 
            "mood": "None", "gender": "None"
        }
    except Exception as e:
        return {"status": "error", "filename": file_obj.name, "msg": str(e)}

# --- 5. COMPACT EDITOR (UI TOI UU CHO ANH DOC) ---
def display_compact_card(idx: int, item: Dict, start_num: int):
    # Khung chua tung anh
    with st.container(border=True):
        # 1. Hien thi anh full width cua cot
        st.image(item["image_obj"], use_container_width=True)
        st.caption(f"#{start_num + idx} - {item['filename']}")
        
        # 2. Input Object (Quan trong nhat -> De tren cung)
        new_obj = st.text_input("Object", value=item["object"], key=f"obj_{idx}", label_visibility="collapsed", placeholder="Nhap Object...")
        
        # 3. Grid 2x2 cho 4 thong so con lai (Tiet kiem dien tich)
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            curr_s = item["style"] if item["style"] in UI_STYLES else "None"
            new_s = st.selectbox("Style", UI_STYLES, index=UI_STYLES.index(curr_s), key=f"s_{idx}", label_visibility="collapsed")
        with r1_c2:
            curr_c = item["color"] if item["color"] in UI_COLORS else "None"
            new_c = st.selectbox("Color", UI_COLORS, index=UI_COLORS.index(curr_c), key=f"c_{idx}", label_visibility="collapsed")
            
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            curr_m = item["mood"] if item["mood"] in UI_MOODS else "None"
            new_m = st.selectbox("Mood", UI_MOODS, index=UI_MOODS.index(curr_m), key=f"m_{idx}", label_visibility="collapsed")
        with r2_c2:
            curr_g = item["gender"] if item["gender"] in UI_GENDERS else "None"
            new_g = st.selectbox("Gender", UI_GENDERS, index=UI_GENDERS.index(curr_g), key=f"g_{idx}", label_visibility="collapsed")

        # Cap nhat Session State
        st.session_state["results"][idx].update({
            "object": new_obj, "style": new_s, "color": new_c, "mood": new_m, "gender": new_g
        })

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Bang Dieu Khien")
    
    # --- CONTROL LAYOUT ---
    st.markdown("### 👁️ Che do hien thi")
    # Mac dinh la 4 cot (Phu hop anh doc)
    cols_per_row = st.slider("So luong anh tren 1 dong:", min_value=2, max_value=6, value=4, help="Keo ve 3 neu muon anh to hon, keo len 5-6 neu muon xem nhieu anh.")
    
    st.divider()
    
    start_idx = st.number_input("STT bat dau:", value=1)
    uploaded_files = st.file_uploader(f"Tai anh ({MAX_IMAGES} max):", type=['png','jpg','jpeg','webp'], accept_multiple_files=True)
    
    analyze_btn = st.button("🚀 BAT DAU", type="primary")
    if st.button("🔄 RESET"):
        st.session_state.clear()
        st.rerun()

# --- 7. MAIN LOGIC ---
if "results" not in st.session_state: st.session_state["results"] = []

if analyze_btn and uploaded_files:
    if len(uploaded_files) > MAX_IMAGES: st.error("Qua so luong!"); st.stop()
    
    temp_res = []
    prog = st.progress(0); st_text = st.empty()
    
    for i, f in enumerate(uploaded_files):
        st_text.text(f"Xu ly: {f.name}...")
        res = process_single_image(f)
        if res["status"]=="ok": res["id"]=i; temp_res.append(res)
        prog.progress((i+1)/len(uploaded_files))
        
    st.session_state["results"] = temp_res
    st_text.success("✅ Xong!"); prog.empty()

# --- 8. DISPLAY & EXPORT ---
if st.session_state["results"]:
    # Container nut Download
    dl_cont = st.container()
    
    st.divider()
    
    # --- DYNAMIC GRID LAYOUT ---
    # Day la thuat toan chia cot thong minh
    grid_cols = st.columns(cols_per_row)
    for i, item in enumerate(st.session_state["results"]):
        # i % cols_per_row se xac dinh anh nam o cot nao (0, 1, 2, 3...)
        with grid_cols[i % cols_per_row]:
            display_compact_card(i, item, start_idx)
            
    # Logic Download (Nam o cuoi code nhung hien thi o tren cung nho st.container)
    with dl_cont:
        c1, c2 = st.columns([3, 1])
        with c1: st.subheader(f"📊 Ket qua: {len(st.session_state['results'])} anh")
        with c2:
            ex_data = []
            for item in st.session_state["results"]:
                tags = [t for t in [item["object"].strip(), item["style"], item["color"], item["mood"], item["gender"]] if t and t != "None"]
                ex_data.append({
                    "STT": start_idx + st.session_state["results"].index(item),
                    "Ten file": item["filename"],
                    "Final Prompt": ", ".join(tags),
                    "Object": item["object"], "Style": item["style"], "Color": item["color"], "Mood": item["mood"], "Gender": item["gender"]
                })
            
            df = pd.DataFrame(ex_data)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
                worksheet = writer.sheets['Sheet1']
                worksheet.set_column(0, 0, 5)
                worksheet.set_column(1, 1, 25)
                worksheet.set_column(2, 2, 50)
                
            st.download_button("📥 TAI EXCEL", buf.getvalue(), "hashtags_v11.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif not uploaded_files:
    st.info("👈 Tai anh len de bat dau (Ho tro anh doc 9:16 cuc tot).")
