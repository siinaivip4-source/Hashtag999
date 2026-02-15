"""
ENTERPRISE CONTENT TAGGER SYSTEM
Developed by: [SiinNoBox Team]
Version: 16.0 (Ultimate Precision)
Description: Automated image analysis with strict Object-Oriented Guardrails.
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
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Enterprise + Fixed Sidebar Button
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background-color: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #e9ecef;
    }
    div[data-testid="stImage"] img { border-radius: 4px; object-fit: contain; }
    div[data-testid="stButton"] > button[kind="primary"], div[data-testid="stDownloadButton"] > button {
        background-color: #0f5132 !important; border-color: #0f5132 !important; color: white !important; font-weight: bold;
    }
    /* FIX NÚT SIDEBAR */
    button[kind="header"] { color: #212529 !important; display: block !important; opacity: 1 !important; }
    div[data-testid="stSidebarNav"] { display: block !important; }
    </style>
""", unsafe_allow_html=True)

st.title("HỆ THỐNG PHÂN TÍCH & TỐI ƯU HÓA NỘI DUNG")
st.markdown("**Phiên bản V16 (Ultimate Precision)** | Dual Guardrails Technology")
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

# --- [CORE] GUARDRAILS LOGIC (ĐỊNH NGHĨA CHẶT CHẼ) ---

# 1. STYLE: Định nghĩa dựa trên kỹ thuật tạo hình (Render/Vector/Photo...)
STYLE_PROMPT_MAP = {
    # 2D: Phải là vector, nét phẳng, không có bóng đổ khối
    "2D": "flat 2d vector art, simple lines, cartoon illustration, no realistic shading",
    
    # 3D: Phải là đồ họa máy tính, render, khối nổi
    "3D": "3d computer graphics, blender render, c4d, unreal engine, volumetric lighting, plastic material",
    
    # Realism: Phải là ảnh chụp thật
    "Realism": "real life photography, 4k photo, raw camera image, hyperrealistic skin texture, dslr",
    
    # Anime: Đặc trưng truyện tranh Nhật
    "Animeart": "anime style, japanese manga, cel shading, 2d character design, waifu",
    
    # Cinematic: Ánh sáng điện ảnh, giống phim
    "Cinematic": "cinematic movie scene, dramatic lighting, film grain, wide shot, movie poster style",
    
    # Digital Art: Vẽ trên máy tính (Wacom)
    "Digitalart": "digital painting, wacom tablet drawing, highly detailed concept art, artstation style",
    
    # Pixel Art: Ô vuông
    "Pixelart": "pixel art, 8-bit, 16-bit, dot matrix, blocky edges, retro game",
    
    # Vangogh: Nét cọ sơn dầu
    "Vangoghart": "vincent van gogh style, oil painting, thick brush strokes, impressionism, starry night",
    
    # Cyberpunk: Đèn Neon + Công nghệ cao
    "Cyberpunk": "cyberpunk city, neon lights, high tech low life, futuristic sci-fi, cyborg",
    
    # Lofi: Chill, nét mềm, hoạt hình retro
    "Lofi": "lofi hip hop style, chill vibes, retro anime aesthetic, study girl, soft lighting",
    
    # Vintage: Màu cũ, nhiễu hạt
    "Vintage": "vintage retro style, sepia tone, old photograph, film grain, noise, 1980s",
    
    # Horror: Tối tăm, đáng sợ
    "Horror": "horror theme, scary, creepy, dark nightmare, monster, gore, blood",
    
    # Minimalism: Ít chi tiết
    "Minimalism": "minimalism, simple clean lines, minimal art, negative space, simple background",
    
    # Cute: Dễ thương, tròn trịa
    "Cute": "cute kawaii, chibi style, adorable character, soft shapes, pastel vibe",
    
    # Cool: Thời trang, ngầu (Dành cho Fashion)
    "Cool": "cool stylish fashion, streetwear, edgy vibe, magazine cover, posing",
    
    # Aesthetic: Bố cục đẹp, nghệ thuật
    "Aesthetic": "aesthetic artistic composition, beautiful lighting, dreamy atmosphere, tumblr style",
    
    # Fantasy: Phép thuật, trung cổ
    "Fantasy": "fantasy art, magic, dungeons and dragons, medieval armor, sword, wizard",
    
    # Comic: Nét đậm, truyện tranh Mỹ
    "Comic": "comic book style, bold outlines, pop art, marvel dc style, halftone dots",
    
    # Scifi: Vũ trụ, tàu không gian
    "Scifi": "sci-fi, science fiction, outer space, spaceship, alien, futuristic technology"
}

# 2. COLOR: Định nghĩa dựa trên Vật thể/Quần áo (Đã Fix lỗi Black/Red)
COLOR_PROMPT_MAP = {
    "Black": "black clothing, black outfit, black object, black fashion, matte black material",
    "White": "white clothing, white outfit, white object, bright white surface",
    "Blackandwhite": "black and white photography, monochrome, greyscale image",
    "Red": "bright red clothing, red car, red flower, crimson object, strong red color",
    "Yellow": "bright yellow clothing, yellow object, sunflower color, golden yellow",
    "Blue": "blue clothing, blue sky, blue ocean, blue object, cyan",
    "Green": "green clothing, green plants, nature, forest, green object",
    "Pink": "pink clothing, pink flower, magenta, hot pink object",
    "Orange": "orange clothing, orange fruit, sunset color, pumpkin orange",
    "Pastel": "soft pastel colors, pale pink blue yellow, baby colors",
    "Hologram": "holographic texture, iridescent rainbow reflection, metallic silver rainbow",
    "Vintage": "sepia tone, old photograph style, retro brown filter",
    "Colorful": "rainbow colors, many different vibrant colors, confetti, festival",
    "Neutral": "beige clothing, cream color, skin tone, earth tones, sand color",
    "Light": "bright image, high key lighting, sunny day, white background",
    "Dark": "low key lighting, night scene, shadows, silhouette, dark room",
    "Warm": "warm lighting, orange tone, golden hour, cozy atmosphere",
    "Cold": "cold lighting, blue tone, winter atmosphere, ice",
    "Neon": "glowing neon signs, cyberpunk lights, laser beam",
    "Gradient": "smooth color gradient background, blurred transition",
    "Purple": "purple clothing, violet, lavender object, grape color",
    "Brown": "brown clothing, wooden texture, chocolate color, soil",
    "Grey": "grey clothing, concrete wall, silver metal, grey object, ash color",
}

# --- 4. CORE ENGINE ---
@st.cache_resource
def load_ai_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"AI Engine initializing on: {device}")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(CONFIG["MODEL_NAME"], pretrained=CONFIG["PRETRAINED"], device=device)
        tokenizer = open_clip.get_tokenizer(CONFIG["MODEL_NAME"])
        
        # Tạo Vectors từ Map đã định nghĩa
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
            
        # Tính toán xác suất
        s_probs = (100.0 * img_feat @ s_feat.T).softmax(dim=-1)
        c_probs = (100.0 * img_feat @ c_feat.T).softmax(dim=-1)
        
        s_idx = s_probs.argmax().item()
        c_idx = c_probs.argmax().item()
        
        # Debug Confidence
        s_score = s_probs[0][s_idx].item() * 100
        c_score = c_probs[0][c_idx].item() * 100
        
        return {
            "status": "success", "filename": file_obj.name, "image_obj": thumb, 
            "object": "", 
            "style": AI_STYLES[s_idx], 
            "color": AI_COLORS[c_idx], 
            "confidence_s": f"{s_score:.1f}%",
            "confidence_c": f"{c_score:.1f}%",
            "mood": "None", "gender": "None"
        }
    except Exception as e:
        return {"status": "error", "filename": file_obj.name, "msg": str(e)}

# --- 5. UI COMPONENTS ---
def render_image_card(idx: int, item: Dict, start_num: int):
    with st.container(border=True):
        st.image(item["image_obj"], use_container_width=True)
        st.caption(f"STT: {start_num + idx} | File: {item['filename']}")
        
        # Hiển thị độ tin cậy của AI (Rất tốt để debug)
        st.caption(f"🎯 AI: {item['style']} ({item.get('confidence_s','?')}) | {item['color']} ({item.get('confidence_c','?')})")
        
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
