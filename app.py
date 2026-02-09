import streamlit as st
import pandas as pd
from PIL import Image
import io
import torch
import logging
import gc
import os

# Thay thế: import clip -> open_clip
import open_clip

# --- 0. CẤU HÌNH HỆ THỐNG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Các giới hạn hệ thống
MAX_IMAGES = 50                  
MAX_FILE_SIZE_MB = 10            
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
THUMBNAIL_SIZE = (300, 300)      
CLIP_INPUT_SIZE = (224, 224)     

# --- 1. THIẾT LẬP GIAO DIỆN & CSS ---
st.set_page_config(
    page_title="AI Master V9 - Content Optimizer", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (giữ nguyên như cũ)
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
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #1e6b41 !important;
        box-shadow: 0 4px 8px rgba(33, 115, 70, 0.4);
    }
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
    div.stSelectbox > label {
        font-weight: 600; 
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ AI MASTER V9 - CONTENT OPTIMIZER")
st.markdown("#### Hệ thống tự động phân tích và tối ưu hóa Hashtag cho hình ảnh")
st.markdown("---")

# --- 2. DỮ LIỆU PHÂN LOẠI ---
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

# --- 3. KHỞI ĐỘNG AI ENGINE (Thay đổi ở đây) ---
@st.cache_resource
def load_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"System running on: {device}")
    
    try:
        # Thay đổi: Dùng open_clip thay vì clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='openai',
            device=device
        )
        model.eval()
        
        # Tokenize prompts
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        s_prompts = [f"a {s} style artwork" for s in STYLES]
        c_prompts = [f"dominant color is {c}" for c in COLORS]
        
        s_tokens = tokenizer(s_prompts).to(device)
        c_tokens = tokenizer(c_prompts).to(device)
        
        with torch.no_grad():
            s_feat = model.encode_text(s_tokens)
            c_feat = model.encode_text(c_tokens)
            s_feat /= s_feat.norm(dim=-1, keepdim=True)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)
            
        return model, preprocess, s_feat, c_feat, device
    except Exception as e:
        logger.error(f"Critical Error - Model Load Failed: {e}")
        raise e

try:
    with st.spinner("⏳ Đang khởi động hệ thống AI (Lần đầu sẽ mất khoảng 30s)..."):
        model, preprocess, s_feat, c_feat, device = load_engine()
except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")
    st.stop()

# --- 4. HÀM XỬ LÝ ẢNH (Giữ nguyên như cũ) ---
def process_single_image(file_obj) -> dict:
    # ... (code cũ giữ nguyên)
