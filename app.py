ưimport streamlit as st
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
    page_title="AI Master V9 - Content Optimizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* 1. Vien anh mem mai */
    div[data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 2. Style chung cho cac nut bam */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        height: 3em;
    }
    
    /* 3. Button Primary (Green) */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #1e6b41 !important;
        box-shadow: 0 4px 8px rgba(33, 115, 70, 0.4);
    }

    /* 4. Download Button (Green) */
    div[data-testid="stDownloadButton"] > button {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #1e6b41 !important;
        border-color: #1e6b41 !important;
        box-shadow: 0 4px 8px rgba(33, 115, 70, 0.4);
    }
    div[data-testid="stDownloadButton"] > button:active {
        background-color: #1e6b41 !important;
        color: white !important;
    }

    /* 5. Dropdown Label */
    div.stSelectbox > label {
        font-weight: 600;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ AI MASTER V9 - CONTENT OPTIMIZER")
st.markdown("#### He thong tu dong phan tich va toi uu hoa Hashtag cho hinh anh")
st.markdown("---")

# --- 2. DU LIEU PHAN LOAI (DATASET) ---
STYLES = [
    "2D", "3D", "Cute", "Animeart", "Realism",
    "Aesthetic", "Cool", "Fantasy", "Comic", "Horror",
    "Cyberpunk", "Lofi", "Minimalism", "Digitalart", "Cinematic",
    "Pixelart", "Scifi", "Vangoghart"
]

COLORS = [
    "Black", "White", "Blackandwhite", "Red", "Yellow",
    "Blue", "Green", "Pink", "Orange", "Pastel",
    "Hologram", "Vintage", "Colorful", "Neutral", "Light",
    "Dark", "Warm", "Cold", "Neon", "Gradient",
    "Purple", "Brown", "Grey"
]

# --- 3. KHOI DONG AI ENGINE ---
@st.cache_resource
def load_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"System running on: {device}")
    
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        s_prompts = [f"a {s} style artwork" for s in STYLES]
        c_prompts = [f"dominant color is {c}" for c in COLORS]
        
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

# --- 4. HAM XU LY ANH (OPTIMIZED) ---
def process_single_image(file_obj) -> Dict:
    try:
        file_bytes = file_obj.getvalue()
        original_img = Image.open(io.BytesIO(file_bytes))
        
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
            
        # RAM Saver
        thumb = original_img.copy()
        thumb.thumbnail(THUMBNAIL_SIZE)
        
        # CPU Saver
        input_img = original_img.resize(CLIP_INPUT_SIZE)
        img_input = preprocess(input_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feat = model.encode_image(img_input)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        s_idx = (100.0 * img_feat @ s_feat.T).softmax(dim=-1).argmax().item()
        c_idx = (100.0 * img_feat @ c_feat.T).softmax(dim=-1).argmax().item()
        
        return {
            "status": "ok",
            "filename": file_obj.name,
            "image_obj": thumb,
            "style": STYLES[s_idx],
            "color": COLORS[c_idx]
        }
    except Exception as e:
        logger.error(f"Error processing {file_obj.name}: {e}")
        return {"status": "error", "filename": file_obj.name, "msg": str(e)}

def display_image_editor(idx: int, item: Dict, start_num: int):
    with st.container(border=True):
        st.image(item["image_obj"], use_container_width=True)
        st.caption(f"#{start_num + idx} - {item['filename']}")
        
        c1, c2 = st.columns(2)
        with c1:
            new_s = st.selectbox("Phong cach", STYLES, index=STYLES.index(item["style"]), key=f"s_{idx}")
        with c2:
            new_c = st.selectbox("Mau chu dao", COLORS, index=COLORS.index(item["color"]), key=f"c_{idx}")
        
        st.session_state["results"][idx]["style"] = new_s
        st.session_state["results"][idx]["color"] = new_c

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Bang Dieu Khien")
    
    st.info("💡 **Huong dan:** Tai anh len -> He thong tu dong gan the -> Tai file Excel.")
    
    start_idx = st.number_input("So thu tu bat dau (STT):", value=1, step=1, min_value=1)
    
    uploaded_files = st.file_uploader(
        f"Tai anh len (Toi da {MAX_IMAGES} anh):", 
        type=['png','jpg','jpeg','webp'], 
        accept_multiple_files=True,
        help="Ho tro dinh dang PNG, JPG, WEBP. Dung luong toi da 10MB/anh."
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
        st.error(f"⚠️ Vui long tai len toi da {MAX_IMAGES} anh.")
        st.stop()
        
    temp_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        if file.size > MAX_FILE_SIZE_BYTES:
            st.warning(f"⚠️ Bo qua: {file.name} (>10MB)")
            continue
            
        status_text.text(f"Dang phan tich: {file.name} ({i+1}/{total_files})...")
        res = process_single_image(file)
        
        if res["status"] == "ok":
            res["id"] = i
            temp_results.append(res)
        else:
            st.warning(f"⚠️ Loi anh {res['filename']}: {res['msg']}")
            
        progress_bar.progress((i+1)/total_files)
    
    st.session_state["results"] = temp_results
    status_text.success(f"✅ Hoan tat! Da xu ly {len(temp_results)} anh.")
    progress_bar.empty()

# --- 7. EXPORT & DISPLAY ---
if st.session_state["results"]:
    st.divider()
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"📊 Ket qua phan tich ({len(st.session_state['results'])} anh)")
        st.caption("Kiem tra va chinh sua truoc khi xuat file.")
    with c2:
        export_data = []
        for i, item in enumerate(st.session_state["results"]):
            export_data.append({
                "STT": start_idx + i,
                "Ten tap tin": item["filename"],
                "Hashtag Style": item["style"],
                "Hashtag Color": item["color"]
            })
        df = pd.DataFrame(export_data)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
            worksheet = writer.sheets['Sheet1']
            worksheet.set_column(0, 0, 5)
            worksheet.set_column(1, 1, 30)
            worksheet.set_column(2, 3, 20)
            
        st.download_button(
            label="📥 TAI VE FILE EXCEL",
            data=buffer.getvalue(),
            file_name="ket_qua_hashtags.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, item in enumerate(st.session_state["results"]):
        with cols[i % 3]: 
            display_image_editor(i, item, start_idx)

elif not uploaded_files:
    st.info("👈 Vui long tai anh len tu thanh dieu khien ben trai de bat dau.")
    with st.expander("ℹ️ Gioi thieu tinh nang"):
        st.markdown("""
        **AI Master V9** su dung cong nghe CLIP de:
        1.  **Nhan dien Style & Color** tu dong.
        2.  **Toi uu hoa** quy trinh lam noi dung.
        3.  **Xuat Excel** nhanh chong.
        """)
