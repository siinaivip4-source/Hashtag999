"""
ENTERPRISE CONTENT TAGGER SYSTEM
Developed by: [SiinNoBox Team]
Version: 14.1 (UI Fixed)
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

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
    initial_sidebar_state="expanded" # Mặc định mở, nhưng vẫn có nút đóng
)

# Custom CSS: Giao diện phẳng, chuyên nghiệp + FIX LỖI MẤT NÚT SIDEBAR
st.markdown("""
    <style>
    /* Tổng thể container */
    .main { background-color: #ffffff; }
    
    /* Card sản phẩm */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #e9ecef;
    }
    
    /* Hình ảnh */
    div[data-testid="stImage"] img { border-radius: 4px; object-fit: contain; }
    
    /* Nút bấm Primary */
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
    h1, h2, h3 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #212529; }
    .stSelectbox, .stTextInput { font-size: 0.9rem; }
    div[data-testid="stCaptionContainer"] { font-size: 0.8rem; color: #6c757d; }

    /* --- [FIX] KHÔI PHỤC NÚT ĐÓNG/MỞ SIDEBAR --- */
    
    /* 1. Hiển thị rõ ràng nút mũi tên (Chevron) ở góc trái */
    button[kind="header"] {
        background-color: transparent !important;
        color: #212529 !important; /* Màu đen đậm để dễ nhìn */
        opacity: 1 !important;
        display: block !important;
        z-index: 999999 !important; /* Luôn nằm trên cùng */
    }
    
    /* 2. Đảm bảo thanh header của sidebar không bị ẩn */
    div[data-testid="stSidebarNav"] {
        display: block !important;
    }

    /* 3. Màu sắc khi hover vào nút đóng mở */
    button[kind="header"]:hover {
        color: #0f5132 !important; /* Xanh doanh nghiệp khi di chuột */
        background-color: #f0f0f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("HỆ THỐNG PHÂN TÍCH & TỐI ƯU HÓA NỘI DUNG")
st.markdown("**Phiên bản Doanh nghiệp (Enterprise Edition)** | Powered by OpenCLIP AI")
st.divider()

# --- 3. DATA DICTIONARIES ---
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

# Guardrails Logic
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

# --- 4. CORE ENGINE ---
@st.cache_resource
def load_ai_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"AI Engine initializing on: {device}")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(CONFIG["MODEL_NAME"], pretrained=CONFIG["PRETRAINED"], device=device)
        tokenizer = open_clip.get_tokenizer(CONFIG["MODEL_NAME"])
        
        s_texts = [STYLE_PROMPT_MAP.get(s, f"a {s} style artwork") for s in AI_STYLES]
        c_texts = [COLOR_PROMPT_MAP.get(c, f"dominant color is {c}") for c in AI_COLORS]
        
        s_vectors = tokenizer(s_texts).to(device)
        c_vectors = tokenizer(c_texts).to(device)
        
        with torch.no_grad():
            s_feat = model.encode_text(s_vectors)
            c_feat = model.encode_text(c_vectors)
            s_feat /= s_feat.norm(dim=-1, keepdim=True)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)
        return model, preprocess, s_feat, c_feat, device
    except Exception as e:
        logger.error(f"Critical Error: {e}")
        raise e

try:
    with st.spinner("Đang khởi tạo hệ thống xử lý AI... Vui lòng đợi."):
        model, preprocess, s_feat, c_feat, device = load_ai_engine()
except Exception as e:
    st.error(f"Lỗi khởi tạo: {e}"); st.stop()

def analyze_image(file_obj) -> Dict:
    try:
        file_bytes = file_obj.getvalue()
        original_img = Image.open(io.BytesIO(file_bytes))
        if original_img.mode != "RGB": original_img = original_img.convert("RGB")
        
        thumb = original_img.copy()
        thumb.thumbnail(CONFIG["THUMBNAIL_SIZE"])
        
        input_img = preprocess(original_img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(input_img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        s_idx = (100.0 * img_feat @ s_feat.T).softmax(dim=-1).argmax().item()
        c_idx = (100.0 * img_feat @ c_feat.T).softmax(dim=-1).argmax().item()
        
        return {"status": "success", "filename": file_obj.name, "image_obj": thumb, "object": "", 
                "style": AI_STYLES[s_idx], "color": AI_COLORS[c_idx], "mood": "None", "gender": "None"}
    except Exception as e:
        return {"status": "error", "filename": file_obj.name, "msg": str(e)}

# --- 5. UI COMPONENTS ---
def render_image_card(idx: int, item: Dict, start_num: int):
    with st.container(border=True):
        st.image(item["image_obj"], use_container_width=True)
        st.caption(f"STT: {start_num + idx} | File: {item['filename']}")
        
        new_obj = st.text_input("Đối tượng (Object)", value=item["object"], key=f"obj_{idx}", label_visibility="collapsed", placeholder="Nhập tên đối tượng...")
        
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

        st.session_state["results"][idx].update({"object": new_obj, "style": new_s, "color": new_c, "mood": new_m, "gender": new_g})

# --- 6. SIDEBAR & MAIN LOGIC ---
with st.sidebar:
    st.header("Cấu hình & Dữ liệu")
    st.subheader("Cấu hình hiển thị")
    cols_per_row = st.slider("Số cột:", 2, 6, 4)
    st.divider()
    st.subheader("Nhập liệu")
    start_idx = st.number_input("Số thứ tự bắt đầu:", value=1, step=1)
    uploaded_files = st.file_uploader(f"Tải ảnh lên ({CONFIG['MAX_IMAGES']} max):", type=['png','jpg','jpeg','webp'], accept_multiple_files=True)
    st.markdown("---")
    process_btn = st.button("▶ XỬ LÝ DỮ LIỆU", type="primary")
    if st.button("⟲ Đặt lại hệ thống"): st.session_state.clear(); st.rerun()

if "results" not in st.session_state: st.session_state["results"] = []

if process_btn and uploaded_files:
    if len(uploaded_files) > CONFIG["MAX_IMAGES"]: st.error("Quá giới hạn ảnh."); st.stop()
    processed_results = []
    progress_bar = st.progress(0); status_text = st.empty()
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Đang xử lý: {file.name}...")
        res = analyze_image(file)
        if res["status"] == "success": res["id"] = i; processed_results.append(res)
        progress_bar.progress((i+1)/len(uploaded_files))
    st.session_state["results"] = processed_results
    status_text.success("Xử lý hoàn tất."); progress_bar.empty()

# --- 7. EXPORT ---
if st.session_state["results"]:
    export_container = st.container(); st.divider()
    grid = st.columns(cols_per_row)
    for i, item in enumerate(st.session_state["results"]):
        with grid[i % cols_per_row]: render_image_card(i, item, start_idx)
            
    with export_container:
        c1, c2 = st.columns([3, 1])
        with c1: st.subheader(f"Kết quả phân tích ({len(st.session_state['results'])} mục)")
        with c2:
            export_data = []
            for item in st.session_state["results"]:
                tags = [t for t in [item["object"].strip(), item["style"], item["color"], item["mood"], item["gender"]] if t and t != "None"]
                export_data.append({
                    "STT": start_idx + st.session_state["results"].index(item),
                    "Tên tập tin": item["filename"], "Hashtag Tổng hợp": ", ".join(tags),
                    "Object": item["object"], "Style": item["style"], "Color": item["color"], "Mood": item["mood"], "Gender": item["gender"]
                })
            df = pd.DataFrame(export_data)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
                worksheet = writer.sheets['Data']
                worksheet.set_column('A:A', 5); worksheet.set_column('B:B', 25); worksheet.set_column('C:C', 50)
            st.download_button("📥 XUẤT BÁO CÁO EXCEL", buffer.getvalue(), "Analysed_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
elif not uploaded_files: st.info("Hệ thống sẵn sàng. Vui lòng tải dữ liệu.")
