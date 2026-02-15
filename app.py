import streamlit as st
import pandas as pd
from PIL import Image
import io
import torch
import logging
import gc
import os
import open_clip  # Dùng thư viện mới, ổn định hơn CLIP cũ
from typing import Dict

# --- 0. CẤU HÌNH HỆ THỐNG (SYSTEM CONFIG) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix lỗi xung đột thư viện

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Các giới hạn hệ thống để bảo vệ RAM
MAX_IMAGES = 50                  
MAX_FILE_SIZE_MB = 10            
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
THUMBNAIL_SIZE = (400, 400)      # Tăng nhẹ độ nét cho thumbnail
CLIP_INPUT_SIZE = (224, 224)     

# --- 1. THIẾT LẬP GIAO DIỆN & CSS (UI/UX CAO CẤP) ---
st.set_page_config(
    page_title="AI Master V9 - Immortal Edition", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Giao diện thẻ bài 3D, nút xanh lá, bố cục gọn
st.markdown("""
    <style>
    /* Card chứa ảnh: Bo góc, đổ bóng nhẹ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    /* Hiệu ứng Hover: Nổi lên khi di chuột vào */
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        border-color: #217346;
    }

    /* Ảnh: Bo góc khớp với card */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        object-fit: cover;
    }

    /* Nút bấm (Primary): Màu xanh thương hiệu */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: bold;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #1e6b41 !important;
        box-shadow: 0 4px 8px rgba(33, 115, 70, 0.4);
    }

    /* Nút Download */
    div[data-testid="stDownloadButton"] > button {
        background-color: #217346 !important;
        border-color: #1e6b41 !important;
        color: white !important;
        width: 100%;
        border-radius: 8px;
    }
    
    /* Chỉnh font chữ Caption */
    .stCaption {
        font-size: 0.9em;
        font-weight: 500;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ AI MASTER V9 - IMMORTAL EDITION")
st.markdown("#### 🚀 Hệ thống tối ưu Hashtag & Content tự động (Powered by OpenCLIP)")
st.divider()

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

# --- 3. KHỞI ĐỘNG AI ENGINE (OPEN_CLIP) ---
@st.cache_resource
def load_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"System running on: {device}")
    
    try:
        # Load Model OpenCLIP (ViT-L-14 - Nhẹ & Chuẩn)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', 
            pretrained='openai',
            device=device
        )
        model.eval()
        
        # Tokenizer
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        
        # Pre-compute Text Embeddings (Chạy 1 lần dùng mãi mãi)
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
    with st.spinner("⏳ Đang triệu hồi AI Engine (Lần đầu mất khoảng 30s)..."):
        model, preprocess, s_feat, c_feat, device = load_engine()
except Exception as e:
    st.error(f"❌ Lỗi khởi động AI: {e}")
    st.stop()

# --- 4. HÀM XỬ LÝ ẢNH (CORE LOGIC) ---
def process_single_image(file_obj) -> Dict:
    try:
        file_bytes = file_obj.getvalue()
        original_img = Image.open(io.BytesIO(file_bytes))
        
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
            
        # 1. Tạo Thumbnail cho UI (Resize vừa đủ để hiển thị đẹp)
        thumb = original_img.copy()
        thumb.thumbnail(THUMBNAIL_SIZE)
        
        # 2. Xử lý ảnh cho AI (Resize về 224x224)
        input_img = original_img.resize(CLIP_INPUT_SIZE)
        img_input = preprocess(input_img).unsqueeze(0).to(device)
        
        # 3. Chạy AI phân tích
        with torch.no_grad():
            img_feat = model.encode_image(img_input)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        # 4. So khớp vector
        s_idx = (100.0 * img_feat @ s_feat.T).softmax(dim=-1).argmax().item()
        c_idx = (100.0 * img_feat @ c_feat.T).softmax(dim=-1).argmax().item()
        
        # 5. Dọn dẹp RAM ngay lập tức
        del original_img
        del input_img
        del img_input
        del img_feat
        
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

# --- 5. SIDEBAR (BẢNG ĐIỀU KHIỂN) ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    st.success(f"🟢 System Ready: {device.upper()}")
    
    start_idx = st.number_input("🔢 Số thứ tự bắt đầu (STT):", value=1, step=1, min_value=1)
    
    uploaded_files = st.file_uploader(
        f"📂 Tải ảnh lên (Max {MAX_IMAGES}):", 
        type=['png','jpg','jpeg','webp'], 
        accept_multiple_files=True
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze_btn = st.button("🚀 PHÂN TÍCH", type="primary")
    with col_btn2:
        if st.button("🔄 LÀM MỚI"):
            st.session_state.clear()
            st.rerun()
            
    st.info("💡 **Tips:** Ảnh càng nhẹ phân tích càng nhanh.")

# --- 6. MAIN FLOW (LUỒNG XỬ LÝ CHÍNH) ---
if "results" not in st.session_state:
    st.session_state["results"] = []

if analyze_btn and uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.error(f"⚠️ Quá tải! Vui lòng chỉ tải tối đa {MAX_IMAGES} ảnh.")
        st.stop()
        
    temp_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    
    # Reset & Dọn rác trước khi chạy
    st.session_state["results"] = []
    gc.collect()
    
    for i, file in enumerate(uploaded_files):
        if file.size > MAX_FILE_SIZE_BYTES:
            st.warning(f"⚠️ Bỏ qua: {file.name} (>10MB)")
            continue
            
        status_text.markdown(f"**Đang xử lý:** `{file.name}` ({i+1}/{total_files})")
        res = process_single_image(file)
        
        if res["status"] == "ok":
            res["id"] = i # ID tạm để map dữ liệu
            temp_results.append(res)
        
        progress_bar.progress((i+1)/total_files)
        
        # Dọn rác mỗi 5 ảnh để tránh tràn RAM Cloud
        if i % 5 == 0:
            gc.collect()
    
    st.session_state["results"] = temp_results
    status_text.success(f"✅ Hoàn tất! Đã xử lý {len(temp_results)} ảnh.")
    progress_bar.empty()
    gc.collect()

# --- 7. HIỂN THỊ KẾT QUẢ (GRID 3 CỘT) ---
if st.session_state["results"]:
    # Phần Header kết quả & Nút tải về
    col_header, col_download = st.columns([3, 1])
    
    with col_header:
        st.subheader(f"📊 Kết quả phân tích")
    
    with col_download:
        # Xử lý xuất Excel
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
            worksheet.set_column('B:B', 30) # Rộng cột Tên file
            
        st.download_button(
            label="📥 TẢI EXCEL NGAY",
            data=buffer.getvalue(),
            file_name="ket_qua_hashtags.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.markdown("---")

    # GRID SYSTEM: Hiển thị 3 ảnh/hàng
    results = st.session_state["results"]
    
    # Bước nhảy = 3 (Mỗi lần lấy 3 ảnh)
    for i in range(0, len(results), 3):
        cols = st.columns(3) # Tạo 3 cột
        batch = results[i:i+3] # Lấy nhóm 3 ảnh
        
        for j, item in enumerate(batch):
            with cols[j]: # Bỏ vào cột tương ứng
                with st.container(border=True):
                    # Hiển thị ảnh
                    st.image(item["image_obj"], use_container_width=True)
                    
                    # Tên file (Cắt ngắn nếu dài quá)
                    f_name = item['filename']
                    if len(f_name) > 25: f_name = f_name[:22] + "..."
                    st.caption(f"#{start_idx + i + j}. {f_name}")
                    
                    # Dropdown chỉnh sửa (Ẩn label cho gọn)
                    new_s = st.selectbox(
                        "Style", STYLES, 
                        index=STYLES.index(item["style"]), 
                        key=f"s_{item['id']}",
                        label_visibility="collapsed"
                    )
                    new_c = st.selectbox(
                        "Color", COLORS, 
                        index=COLORS.index(item["color"]), 
                        key=f"c_{item['id']}",
                        label_visibility="collapsed"
                    )
                    
                    # Cập nhật data gốc nếu user chọn lại
                    st.session_state["results"][item['id']]["style"] = new_s
                    st.session_state["results"][item['id']]["color"] = new_c

elif not uploaded_files:
    # Màn hình chờ
    st.info("👈 Vui lòng tải ảnh từ cột bên trái để bắt đầu.")

