import streamlit as st
import pandas as pd
from PIL import Image
import io
import torch
import clip
import logging
from typing import List, Tuple, Dict

# --- 0. CẤU HÌNH HỆ THỐNG (SYSTEM CONFIG) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Các giới hạn hệ thống
MAX_IMAGES = 50         
MAX_FILE_SIZE_MB = 10           
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
THUMBNAIL_SIZE = (300, 300)     
CLIP_INPUT_SIZE = (224, 224)    

# --- 1. THIẾT LẬP GIAO DIỆN & CSS (UI/UX) ---
st.set_page_config(
    page_title="AI Master V9 - Content Optimizer", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Ép màu XANH LÁ cho cả nút PHÂN TÍCH và TẢI EXCEL
st.markdown("""
    <style>
    /* 1. Viền ảnh mềm mại */
    div[data-testid="stImage"] {
        border-radius: 8px; 
        overflow: hidden; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 2. Style chung cho các nút bấm */
    .stButton>button {
        width: 100%; 
        border-radius: 6px; 
        font-weight: 600; 
        height: 3em;
    }
    
    /* 3. [QUAN TRỌNG] ÉP MÀU XANH CHO NÚT PHÂN TÍCH (Primary Button) */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #1e6b41 !important;
        box-shadow: 0 4px 8px rgba(33, 115, 70, 0.4);
    }

    /* 4. [QUAN TRỌNG] ÉP MÀU XANH CHO NÚT TẢI EXCEL (Download Button) */
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
st.markdown("#### Hệ thống tự động phân tích và tối ưu hóa Hashtag cho hình ảnh")
st.markdown("---")

# --- 2. DỮ LIỆU PHÂN LOẠI (DATASET) ---
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

# --- 3. KHỞI ĐỘNG AI ENGINE ---
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
    with st.spinner("⏳ Đang khởi động hệ thống AI..."):
        model, preprocess, s_feat, c_feat, device = load_engine()
except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")
    st.stop()

# --- 4. HÀM XỬ LÝ ẢNH (OPTIMIZED) ---
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
            new_s = st.selectbox("Phong cách", STYLES, index=STYLES.index(item["style"]), key=f"s_{idx}")
        with c2:
            new_c = st.selectbox("Màu chủ đạo", COLORS, index=COLORS.index(item["color"]), key=f"c_{idx}")
        
        st.session_state["results"][idx]["style"] = new_s
        st.session_state["results"][idx]["color"] = new_c

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    
    st.info("💡 **Hướng dẫn:** Tải ảnh lên -> Hệ thống tự động gắn thẻ -> Tải file Excel.")
    
    start_idx = st.number_input("Số thứ tự bắt đầu (STT):", value=1, step=1, min_value=1)
    
    uploaded_files = st.file_uploader(
        f"Tải ảnh lên (Tối đa {MAX_IMAGES} ảnh):", 
        type=['png','jpg','jpeg','webp'], 
        accept_multiple_files=True,
        help="Hỗ trợ định dạng PNG, JPG, WEBP. Dung lượng tối đa 10MB/ảnh."
    )
    
    # Nút này sẽ có MÀU XANH do CSS (kind="primary")
    analyze_btn = st.button("🚀 BẮT ĐẦU PHÂN TÍCH", type="primary")
    
    st.markdown("---")
    # Nút này giữ nguyên màu mặc định (Trắng/Xám)
    if st.button("🔄 Làm mới hệ thống"):
        st.session_state.clear()
        st.rerun()

# --- 6. MAIN LOGIC ---
if "results" not in st.session_state:
    st.session_state["results"] = []

if analyze_btn and uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.error(f"⚠️ Vui lòng tải lên tối đa {MAX_IMAGES} ảnh.")
        st.stop()
        
    temp_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        if file.size > MAX_FILE_SIZE_BYTES:
            st.warning(f"⚠️ Bỏ qua: {file.name} (>10MB)")
            continue
            
        status_text.text(f"Đang phân tích: {file.name} ({i+1}/{total_files})...")
        res = process_single_image(file)
        
        if res["status"] == "ok":
            res["id"] = i
            temp_results.append(res)
        else:
            st.warning(f"⚠️ Lỗi ảnh {res['filename']}: {res['msg']}")
            
        progress_bar.progress((i+1)/total_files)
    
    st.session_state["results"] = temp_results
    status_text.success(f"✅ Hoàn tất! Đã xử lý {len(temp_results)} ảnh.")
    progress_bar.empty()

# --- 7. EXPORT & DISPLAY ---
if st.session_state["results"]:
    st.divider()
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"📊 Kết quả phân tích ({len(st.session_state['results'])} ảnh)")
        st.caption("Kiểm tra và chỉnh sửa trước khi xuất file.")
    with c2:
        export_data = []
        for i, item in enumerate(st.session_state["results"]):
            export_data.append({
                "STT": start_idx + i,
                "Tên tập tin": item["filename"],
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
            
        # Nút này sẽ có MÀU XANH do CSS (stDownloadButton)
        st.download_button(
            label="📥 TẢI VỀ FILE EXCEL",
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
    st.info("👈 Vui lòng tải ảnh lên từ thanh điều khiển bên trái để bắt đầu.")
    with st.expander("ℹ️ Giới thiệu tính năng"):
        st.markdown("""
        **AI Master V9** sử dụng công nghệ CLIP để:
        1.  **Nhận diện Style & Color** tự động.
        2.  **Tối ưu hóa** quy trình làm nội dung.
        3.  **Xuất Excel** nhanh chóng.
        """)
requirements như nào

