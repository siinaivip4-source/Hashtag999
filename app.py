"""
ENTERPRISE CONTENT TAGGER SYSTEM
Developed by: [SiinNoBox Team]
Version: 14.0 (Enterprise Edition)
Description: Automated image analysis and metadata tagging tool using OpenCLIP.
"""

import streamlit as st
import pandas as pd
from PIL import Image
import io
import torch
import open_clip
import logging
from typing import List, Dict, Optional

# --- 1. SYSTEM CONFIGURATION ---
# Thiết lập Logging chuẩn
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Các hằng số cấu hình hệ thống
CONFIG = {
    "MAX_IMAGES": 100,
    "MAX_FILE_SIZE_MB": 10,
    "THUMBNAIL_SIZE": (300, 600),
    "CLIP_INPUT_SIZE": (224, 224),
    "MODEL_NAME": "ViT-B-32",
    "PRETRAINED": "openai"
}

# --- 2. UI/UX CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Content Tagger",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Giao diện phẳng, chuyên nghiệp, tối giản
st.markdown("""
    <style>
    /* Tổng thể container */
    .main {
        background-color: #ffffff;
    }
    
    /* Card sản phẩm */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #e9ecef;
    }
    
    /* Hình ảnh */
    div[data-testid="stImage"] img {
        border-radius: 4px;
        object-fit: contain;
    }
    
    /* Nút bấm Primary (Xanh Doanh Nghiệp) */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #0f5132 !important;
        border-color: #0f5132 !important;
        color: white !important;
        border-radius: 4px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Nút Download */
    div[data-testid="stDownloadButton"] > button {
        background-color: #0f5132 !important;
        border-color: #0f5132 !important;
        color: white !important;
        border-radius: 4px;
        font-weight: 500;
        width: 100%;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #212529;
    }
    
    /* Tinh chỉnh Input */
    .stSelectbox, .stTextInput {
        font-size: 0.9rem;
    }
    div[data-testid="stCaptionContainer"] {
        font-size: 0.8rem;
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# Header Trang trọng
st.title("HỆ THỐNG PHÂN TÍCH & TỐI ƯU HÓA NỘI DUNG")
st.markdown("**Phiên bản Doanh nghiệp (Enterprise Edition)** | Powered by OpenCLIP AI")
st.divider()

# --- 3. DATA DICTIONARIES (BUSINESS LOGIC) ---

# Danh sách Style và Color chuẩn
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

# Danh sách hiển thị trên UI (Thêm tùy chọn None)
UI_STYLES = ["None"] + AI_STYLES
UI_COLORS = ["None"] + AI_COLORS
UI_MOODS = ["None", "Happy", "Sad", "Lonely", "Lovely", "Funny", "ZenMode"]
UI_GENDERS = ["None", "Male", "Female", "Non-binary", "Unisex"]

# Từ điển ánh xạ Prompt (Guardrails Logic)
# Mục đích: Định nghĩa chính xác ngữ nghĩa để AI không hiểu sai
STYLE_PROMPT_MAP = {
    "2D": "flat 2d illustration vector art cartoon style",
    "3D": "3d computer graphics blender render c4d realistic material",
    "Cute": "cute kawaii chibi adorable character design soft shapes",
    "Animeart": "anime style japanese manga illustration cel shaded",
    "Realism": "photorealistic photography 4k high definition real life",
    "Aesthetic": "aesthetic artistic beautiful composition trending on artstation",
    "Cool": "cool stylish edgy fashion streetwear vibe",
    "Fantasy": "fantasy art magic dungeons and dragons medieval warrior",
    "Comic": "comic book style bold lines pop art western comic marvel dc",
    "Horror": "horror scary creepy dark nightmare monster gore",
    "Cyberpunk": "cyberpunk futuristic sci-fi neon high tech city low life",
    "Lofi": "lofi hip hop style chill retro anime aesthetic study girl",
    "Minimalism": "minimalism simple clean lines minimal art negative space",
    "Digitalart": "digital art digital painting wacom tablet drawing concept art",
    "Cinematic": "cinematic movie scene dramatic lighting wide shot film grain",
    "Pixelart": "pixel art 8-bit retro video game style sprite",
    "Scifi": "sci-fi science fiction space future technology alien spaceship",
    "Vangoghart": "vincent van gogh style starry night impressionism oil painting swirl"
}

COLOR_PROMPT_MAP = {
    "Black": "mostly black dark void background",
    "White": "mostly pure white bright background",
    "Blackandwhite": "black and white monochrome photography greyscale",
    "Red": "dominant bright red color object or clothes",
    "Yellow": "dominant bright yellow color sunlight or object",
    "Blue": "dominant blue color sky ocean or object",
    "Green": "dominant green color nature plants or object",
    "Pink": "dominant pink color cute flower or object",
    "Orange": "dominant orange color sunset or fruit",
    "Pastel": "soft pastel colors light desaturated tones",
    "Hologram": "holographic iridescent rainbow silver metallic texture",
    "Vintage": "vintage retro style sepia old photo paper",
    "Colorful": "many different vibrant colors rainbow confetti",
    "Neutral": "neutral beige earth tones minimalist skin tone",
    "Light": "bright high key lighting sunny atmosphere",
    "Dark": "dark dim lighting low light night shadow",
    "Warm": "warm colors temperature red orange yellow heating",
    "Cold": "cold colors temperature blue cyan ice cool lighting",
    "Neon": "glowing neon lights cyberpunk laser",
    "Gradient": "smooth color gradient transition blurred background",
    "Purple": "dominant purple violet lavender color",
    "Brown": "dominant brown color wood earth chocolate",
    "Grey": "dominant grey color concrete silver metal",
}

# --- 4. CORE ENGINE FUNCTIONS ---

@st.cache_resource
def load_ai_engine():
    """
    Khởi tạo model AI và cache lại để tối ưu hiệu suất.
    Sử dụng OpenCLIP ViT-B-32 pretrained OpenAI.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"AI Engine initializing on: {device}")
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CONFIG["MODEL_NAME"], 
            pretrained=CONFIG["PRETRAINED"], 
            device=device
        )
        tokenizer = open_clip.get_tokenizer(CONFIG["MODEL_NAME"])
        
        # Tạo Text Embeddings từ Dictionary (Dual Guardrails)
        s_texts = [STYLE_PROMPT_MAP.get(s, f"a {s} style artwork") for s in AI_STYLES]
        c_texts = [COLOR_PROMPT_MAP.get(c, f"dominant color is {c}") for c in AI_COLORS]
        
        s_vectors = tokenizer(s_texts).to(device)
        c_vectors = tokenizer(c_texts).to(device)
        
        with torch.no_grad():
            s_feat = model.encode_text(s_vectors)
            c_feat = model.encode_text(c_vectors)
            # Normalize vectors
            s_feat /= s_feat.norm(dim=-1, keepdim=True)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)
            
        return model, preprocess, s_feat, c_feat, device
    except Exception as e:
        logger.error(f"Critical Error loading model: {e}")
        raise e

# Khởi tạo Engine
try:
    with st.spinner("Đang khởi tạo hệ thống xử lý AI... Vui lòng đợi."):
        model, preprocess, s_feat, c_feat, device = load_ai_engine()
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}. Vui lòng liên hệ quản trị viên.")
    st.stop()

def analyze_image(file_obj) -> Dict:
    """
    Phân tích một hình ảnh đơn lẻ và trả về dự đoán Style/Color.
    """
    try:
        file_bytes = file_obj.getvalue()
        original_img = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB nếu cần
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
            
        # Tạo Thumbnail
        thumb = original_img.copy()
        thumb.thumbnail(CONFIG["THUMBNAIL_SIZE"])
        
        # Preprocess cho AI
        input_img = preprocess(original_img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            img_feat = model.encode_image(input_img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        # Tính toán độ tương đồng (Cosine Similarity)
        s_probs = (100.0 * img_feat @ s_feat.T).softmax(dim=-1)
        c_probs = (100.0 * img_feat @ c_feat.T).softmax(dim=-1)
        
        s_idx = s_probs.argmax().item()
        c_idx = c_probs.argmax().item()
        
        return {
            "status": "success",
            "filename": file_obj.name,
            "image_obj": thumb,
            "object": "",
            "style": AI_STYLES[s_idx],
            "color": AI_COLORS[c_idx],
            "mood": "None",
            "gender": "None"
        }
    except Exception as e:
        logger.error(f"Error analyzing {file_obj.name}: {e}")
        return {"status": "error", "filename": file_obj.name, "msg": str(e)}

# --- 5. UI COMPONENTS ---

def render_image_card(idx: int, item: Dict, start_num: int):
    """
    Hiển thị thẻ chỉnh sửa thông tin cho từng ảnh.
    Thiết kế tối ưu cho chiều dọc (Vertical Layout).
    """
    with st.container(border=True):
        # Hiển thị ảnh
        st.image(item["image_obj"], use_container_width=True)
        st.caption(f"STT: {start_num + idx} | File: {item['filename']}")
        
        # Input chính: Object
        new_obj = st.text_input(
            "Đối tượng (Object)", 
            value=item["object"], 
            key=f"obj_{idx}", 
            label_visibility="collapsed", 
            placeholder="Nhập tên đối tượng..."
        )
        
        # Grid layout cho các thông số
        c1, c2 = st.columns(2)
        with c1:
            curr_s = item["style"] if item["style"] in UI_STYLES else "None"
            new_s = st.selectbox("Style", UI_STYLES, index=UI_STYLES.index(curr_s), key=f"s_{idx}", label_visibility="collapsed")
            
            curr_m = item["mood"] if item["mood"] in UI_MOODS else "None"
            new_m = st.selectbox("Mood", UI_MOODS, index=UI_MOODS.index(curr_m), key=f"m_{idx}", label_visibility="collapsed")
            
        with c2:
            curr_c = item["color"] if item["color"] in UI_COLORS else "None"
            new_c = st.selectbox("Color", UI_COLORS, index=UI_COLORS.index(curr_c), key=f"c_{idx}", label_visibility="collapsed")
            
            curr_g = item["gender"] if item["gender"] in UI_GENDERS else "None"
            new_g = st.selectbox("Gender", UI_GENDERS, index=UI_GENDERS.index(curr_g), key=f"g_{idx}", label_visibility="collapsed")

        # Cập nhật Session State ngay lập tức
        st.session_state["results"][idx].update({
            "object": new_obj,
            "style": new_s,
            "color": new_c,
            "mood": new_m,
            "gender": new_g
        })

# --- 6. SIDEBAR CONTROL ---
with st.sidebar:
    st.header("Cấu hình & Dữ liệu")
    
    st.subheader("Cấu hình hiển thị")
    cols_per_row = st.slider("Số cột hiển thị:", min_value=2, max_value=6, value=4, help="Điều chỉnh số lượng ảnh trên một hàng.")
    
    st.divider()
    
    st.subheader("Nhập liệu")
    start_idx = st.number_input("Số thứ tự bắt đầu:", value=1, step=1)
    
    uploaded_files = st.file_uploader(
        f"Tải ảnh lên (Tối đa {CONFIG['MAX_IMAGES']}):", 
        type=['png','jpg','jpeg','webp'], 
        accept_multiple_files=True,
        help="Hỗ trợ định dạng JPG, PNG, WEBP."
    )
    
    st.markdown("---")
    
    process_btn = st.button("▶ XỬ LÝ DỮ LIỆU", type="primary")
    
    if st.button("⟲ Đặt lại hệ thống"):
        st.session_state.clear()
        st.rerun()

# --- 7. MAIN APPLICATION LOGIC ---

# Khởi tạo session state
if "results" not in st.session_state:
    st.session_state["results"] = []

# Xử lý khi bấm nút
if process_btn and uploaded_files:
    if len(uploaded_files) > CONFIG["MAX_IMAGES"]:
        st.error(f"Vui lòng tải lên tối đa {CONFIG['MAX_IMAGES']} ảnh một lần.")
        st.stop()
        
    processed_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Đang xử lý: {file.name} ({i+1}/{total})...")
        res = analyze_image(file)
        
        if res["status"] == "success":
            res["id"] = i
            processed_results.append(res)
        else:
            st.warning(f"Không thể xử lý ảnh: {res['filename']} - Lỗi: {res['msg']}")
            
        progress_bar.progress((i+1)/total)
    
    st.session_state["results"] = processed_results
    status_text.success("Quá trình xử lý hoàn tất.")
    progress_bar.empty()

# --- 8. DISPLAY & EXPORT SECTION ---

if st.session_state["results"]:
    # Container xuất dữ liệu (Luôn nằm trên cùng)
    export_container = st.container()
    st.divider()
    
    # Hiển thị lưới ảnh (Grid Layout)
    grid = st.columns(cols_per_row)
    for i, item in enumerate(st.session_state["results"]):
        col_idx = i % cols_per_row
        with grid[col_idx]:
            render_image_card(i, item, start_idx)
            
    # Xử lý Logic Xuất Excel (Đặt trong container trên cùng)
    with export_container:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader(f"Kết quả phân tích ({len(st.session_state['results'])} mục)")
            st.caption("Vui lòng kiểm tra và chỉnh sửa thông tin bên dưới trước khi xuất file.")
        
        with c2:
            export_data = []
            for item in st.session_state["results"]:
                # Logic ghép chuỗi Hashtag
                tags = []
                obj = item["object"].strip()
                if obj: tags.append(obj)
                if item["style"] != "None": tags.append(item["style"])
                if item["color"] != "None": tags.append(item["color"])
                if item["mood"] != "None": tags.append(item["mood"])
                if item["gender"] != "None": tags.append(item["gender"])
                
                final_prompt = ", ".join(tags)
                
                export_data.append({
                    "STT": start_idx + st.session_state["results"].index(item),
                    "Tên tập tin": item["filename"],
                    "Hashtag Tổng hợp": final_prompt,
                    "Object": item["object"],
                    "Style": item["style"],
                    "Color": item["color"],
                    "Mood": item["mood"],
                    "Gender": item["gender"]
                })
            
            # Tạo file Excel trong bộ nhớ
            df = pd.DataFrame(export_data)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
                worksheet = writer.sheets['Data']
                # Định dạng cột
                worksheet.set_column('A:A', 5)   # STT
                worksheet.set_column('B:B', 25)  # Tên file
                worksheet.set_column('C:C', 50)  # Hashtag
                worksheet.set_column('D:H', 15)  # Các cột khác
                
            st.download_button(
                label="📥 XUẤT BÁO CÁO EXCEL",
                data=buffer.getvalue(),
                file_name="Analysed_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

elif not uploaded_files:
    st.info("Hệ thống đã sẵn sàng. Vui lòng tải dữ liệu hình ảnh từ thanh điều khiển để bắt đầu.")
