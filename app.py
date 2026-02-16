"""
ENTERPRISE CONTENT TAGGER SYSTEM
Developed by: [SiinNoBox Team]
Version: 20.1 (Fix NameError)
Description: Support both File Upload & Local Folder Scan.
"""

import streamlit as st
import pandas as pd
from PIL import Image
import io
import torch
import open_clip
import logging
import os
from typing import List, Dict, Union

# --- 1. SYSTEM CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "MAX_IMAGES": 1000,
    "THUMBNAIL_SIZE": (300, 600),
    "CLIP_INPUT_SIZE": (224, 224),
    "MODEL_NAME": "ViT-B-32",
    "PRETRAINED": "openai"
}

# --- 2. UI/UX CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Content Tagger",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1, h2, h3, p, div { font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background-color: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #e9ecef;
    }
    div[data-testid="stImage"] img { border-radius: 4px; object-fit: contain; }
    div[data-testid="stButton"] > button[kind="primary"], div[data-testid="stDownloadButton"] > button {
        background-color: #0f5132 !important; border-color: #0f5132 !important; color: white !important; font-weight: bold;
    }
    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important; color: #333333 !important; border: 1px solid #cccccc !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #ffffff !important; color: #333333 !important; border: 1px solid #cccccc !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important; z-index: 1000000 !important; color: #0f5132 !important; background-color: white !important;
    }
    section[data-testid="stSidebar"] button {
        display: block !important; visibility: visible !important; opacity: 1 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("HỆ THỐNG PHÂN TÍCH & TỐI ƯU HÓA NỘI DUNG")
st.markdown("**Phiên bản V20.1 (Fix NameError)** | Hỗ trợ Upload & Quét Thư Mục Local")
st.divider()

# --- 3. DATA DICTIONARIES ---
AI_STYLES = ["2D", "3D", "Cute", "Animeart", "Realism", "Aesthetic", "Cool", "Fantasy", "Comic", "Horror", "Cyberpunk", "Lofi", "Minimalism", "Digitalart", "Cinematic", "Pixelart", "Scifi", "Vangoghart"]
AI_COLORS = ["Black", "White", "Blackandwhite", "Red", "Yellow", "Blue", "Green", "Pink", "Orange", "Pastel", "Hologram", "Vintage", "Colorful", "Neutral", "Light", "Dark", "Warm", "Cold", "Neon", "Gradient", "Purple", "Brown", "Grey"]
UI_STYLES = ["None"] + AI_STYLES
UI_COLORS = ["None"] + AI_COLORS
UI_MOODS = ["None", "Happy", "Sad", "Lonely", "Lovely", "Funny", "ZenMode"]
UI_GENDERS = ["None", "Male", "Female", "Non-binary", "Unisex"]

# --- GUARDRAILS ---
STYLE_PROMPT_MAP = {
    "2D": "flat 2d vector art, simple lines, cartoon illustration, no realistic shading, bold silhouettes, clean outlines, minimalist vector, paper-cut aesthetic, solid color fills",
    "3D": "3d computer graphics, blender render, c4d, unreal engine, volumetric lighting, plastic material, octane render, ray tracing, soft shadows, claymorphism, high-gloss finish",
    "Realism": "real life photography, 4k photo, raw camera image, hyperrealistic skin texture, dslr, shutter speed, aperture f/1.8, natural lighting, intricate pores, lifelike detail",
    "Animeart": "anime style, japanese manga, cel shading, 2d character design, waifu, makoto shinkai vibe, vibrant eyes, expressive lineart, ghibli atmosphere, sharp highlights",
    "Cinematic": "cinematic movie scene, dramatic lighting, film grain, wide shot, movie poster style, anamorphic lens flare, epic composition, color graded, IMAX quality, suspenseful mood",
    "Digitalart": "digital painting, wacom tablet drawing, highly detailed concept art, artstation style, speedpaint, soft brushwork, deviantart trending, fantasy landscape, layered texture",
    "Pixelart": "pixel art, 8-bit, 16-bit, dot matrix, blocky edges, retro game, gameboy aesthetic, dithering, limited palette, sprite sheet, isometric pixel",
    "Vangoghart": "vincent van gogh style, oil painting, thick brush strokes, impressionism, starry night, impasto technique, swirling sky, vivid emotional colors, post-impressionist, canvas texture",
    "Cyberpunk": "cyberpunk city, neon lights, high tech low life, futuristic sci-fi, cyborg, night city rain, synthetic neon, mechanical limbs, flying cars, rainy pavement reflections",
    "Lofi": "lofi hip hop style, chill vibes, retro anime aesthetic, study girl, soft lighting, grainy texture, purple haze, cozy bedroom, nostalgic mood, lo-fi filter",
    "Vintage": "vintage retro style, sepia tone, old photograph, film grain, noise, 1980s, vhs glitch, polaroid frame, faded colors, retro fashion, analog feel",
    "Horror": "horror theme, scary, creepy, dark nightmare, monster, gore, blood, eerie fog, eldritch terror, unsettling shadows, gothic macabre, sinister grin",
    "Minimalism": "minimalism, simple clean lines, minimal art, negative space, simple background, zen-like simplicity, geometric harmony, essential shapes, breathing room, monochromatic",
    "Cute": "cute kawaii, chibi style, adorable character, soft shapes, pastel vibe, sanrio aesthetic, big sparkly eyes, squishy appearance, wholesome mood, tiny limbs",
    "Cool": "cool stylish fashion, streetwear, edgy vibe, magazine cover, posing, urban chic, hypebeast style, confident gaze, high fashion editorial, vogue aesthetic",
    "Aesthetic": "aesthetic artistic composition, beautiful lighting, dreamy atmosphere, tumblr style, ethereal glow, vaporwave elements, soft focus, poetic visuals, curated mood",
    "Fantasy": "fantasy art, magic, dungeons and dragons, medieval armor, sword, wizard, mythical creatures, mystical aura, ancient ruins, epic quest, enchanted forest",
    "Comic": "comic book style, bold outlines, pop art, marvel dc style, halftone dots, action lines, speech bubbles, vibrant ink, ink wash, retro superhero",
    "Scifi": "sci-fi, science fiction, outer space, spaceship, alien, futuristic technology, interstellar travel, high-tech laboratory, planetary rings, mecha design, starlight"
}

COLOR_PROMPT_MAP = {
    "Black": "black clothing, black outfit, black object, black fashion, matte black material, deep obsidian, midnight void, jet black silk, charcoal shadows, ink-washed darkness",
    "White": "white clothing, white outfit, white object, bright white surface, ivory elegance, snow-capped purity, pearl sheen, alabaster texture, bleached linen",
    "Blackandwhite": "black and white photography, monochrome, greyscale image, high contrast noir, silver screen nostalgia, charcoal sketch, dramatic chiaroscuro, ink on parchment",
    "Red": "bright red clothing, red car, red flower, crimson object, strong red color, ruby radiance, scarlet velvet, burning ember, cherry blossom red, blood orange intensity",
    "Yellow": "bright yellow clothing, yellow object, sunflower color, golden yellow, canary brilliance, honey gold, lemon zest, amber glow, saffron silk",
    "Blue": "blue clothing, blue sky, blue ocean, blue object, cyan, sapphire depth, cobalt sky, navy professional, azure mist, electric blue spark",
    "Green": "green clothing, green plants, nature, forest, green object, emerald lush, mossy earth, jade stone, lime vibrancy, sage leaves",
    "Pink": "pink clothing, pink flower, magenta, hot pink object, rose petal, blush satin, coral tint, bubblegum pop, dusty mauve",
    "Orange": "orange clothing, orange fruit, sunset color, pumpkin orange, apricot warmth, copper metallic, terracotta clay, tiger lily, marmalade glow",
    "Pastel": "soft pastel colors, pale pink blue yellow, baby colors, mint cream, lavender haze, peach fuzz, macaron palette, misty sky blue",
    "Hologram": "holographic texture, iridescent rainbow reflection, metallic silver rainbow, opalescent shimmer, liquid mercury rainbow, glitch aesthetic, soap bubble film, pearlescent foil",
    "Vintage": "sepia tone, old photograph style, retro brown filter, faded polaroid, grainy film stock, daguerreotype finish, antique parchment, tea-stained edges",
    "Colorful": "rainbow colors, many different vibrant colors, confetti, festival, kaleidoscope burst, psychedelic swirl, mardi gras palette, stained glass, technicolor dream",
    "Neutral": "beige clothing, cream color, skin tone, earth tones, sand color, oatmeal linen, taupe elegance, mushroom grey, sandstone, light khaki",
    "Light": "bright image, high key lighting, sunny day, white background, overexposed ethereal, backlit glow, ethereal morning haze, soft focus brightness, studio ring light",
    "Dark": "low key lighting, night scene, shadows, silhouette, dark room, moody vignette, cinematic shadows, heavy noir atmosphere, dim candlelit, obscured silhouette",
    "Warm": "warm lighting, orange tone, golden hour, cozy atmosphere, candlelight flicker, campfire radiance, sepia sunset, toasted almond, brassy undertones",
    "Cold": "cold lighting, blue tone, winter atmosphere, ice, frosty glaze, steel blue chill, arctic pale, moonlight silver, fluorescent clinical",
    "Neon": "glowing neon signs, cyberpunk lights, laser beam, vaporwave magenta, toxic lime glow, night city pulse, ultraviolet streak, radioactive cyan",
    "Gradient": "smooth color gradient background, blurred transition, ombre transition, dusk to dawn bleed, silk-smooth blending, watercolor wash, horizon blur",
    "Purple": "purple clothing, violet, lavender object, grape color, royal amethyst, plum velvet, orchid bloom, dark violet mystery, lilac breeze",
    "Brown": "brown clothing, wooden texture, chocolate color, soil, mahogany wood, espresso crema, rustic leather, cinnamon spice, weathered bronze",
    "Grey": "grey clothing, concrete wall, silver metal, grey object, ash color, slate stone, gunmetal industrial, misty fog, pewter shine, heathered wool"
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
    with st.spinner("Đang khởi tạo hệ thống (V20.1 Stable)..."):
        model, preprocess, s_feat, c_feat, device = load_ai_engine()
except Exception as e:
    st.error(f"Lỗi khởi tạo: {e}"); st.stop()

# --- FUNCTION: IMAGE ANALYSIS ---
def analyze_image(file_input: Union[object, str]) -> Dict:
    try:
        if isinstance(file_input, str):
            filename = os.path.basename(file_input)
            original_img = Image.open(file_input)
        else:
            filename = file_input.name
            original_img = Image.open(io.BytesIO(file_input.getvalue()))

        if original_img.mode != "RGB": original_img = original_img.convert("RGB")
        thumb = original_img.copy()
        thumb.thumbnail(CONFIG["THUMBNAIL_SIZE"])
        input_img = preprocess(original_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feat = model.encode_image(input_img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        s_probs = (100.0 * img_feat @ s_feat.T).softmax(dim=-1)
        c_probs = (100.0 * img_feat @ c_feat.T).softmax(dim=-1)
        
        s_idx = s_probs.argmax().item()
        c_idx = c_probs.argmax().item()
        s_score = s_probs[0][s_idx].item() * 100
        c_score = c_probs[0][c_idx].item() * 100
        
        return {"status": "success", "filename": filename, "image_obj": thumb, "object": "", "style": AI_STYLES[s_idx], "color": AI_COLORS[c_idx], "confidence_s": f"{s_score:.1f}%", "confidence_c": f"{c_score:.1f}%", "mood": "None", "gender": "None"}
    except Exception as e:
        fname = os.path.basename(file_input) if isinstance(file_input, str) else file_input.name
        return {"status": "error", "filename": fname, "msg": str(e)}

# --- 5. UI COMPONENTS ---
def render_image_card(idx: int, item: Dict, start_num: int):
    with st.container(border=True):
        st.image(item["image_obj"], use_container_width=True)
        st.caption(f"STT: {start_num + idx} | File: {item['filename']}")
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
    cols_per_row = st.slider("Số cột hiển thị:", 2, 6, 4)
    st.divider()
    
    st.subheader("Nguồn Dữ Liệu")
    input_method = st.radio("Chọn phương thức:", ["📁 Upload File/Folder", "🖥️ Quét Thư Mục Local"], index=0)
    start_idx = st.number_input("Số thứ tự bắt đầu:", value=1, step=1)
    
    files_to_process = []
    uploaded_files = [] # <--- KHỞI TẠO BIẾN Ở ĐÂY ĐỂ TRÁNH NAME ERROR
    
    if input_method == "📁 Upload File/Folder":
        uploaded_files = st.file_uploader(f"Kéo thả ảnh vào đây:", type=['png','jpg','jpeg','webp'], accept_multiple_files=True)
        if uploaded_files:
            files_to_process = uploaded_files
            st.info(f"Đã chọn: {len(files_to_process)} ảnh")
            
    else: # Local Folder Scan
        local_path = st.text_input("Nhập đường dẫn thư mục (VD: D:\Images):")
        if local_path and os.path.isdir(local_path):
            valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
            try:
                files_to_process = [os.path.join(local_path, f) for f in os.listdir(local_path) if f.lower().endswith(valid_exts)]
                st.success(f"Tìm thấy: {len(files_to_process)} ảnh hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi đọc thư mục: {e}")
        elif local_path:
            st.warning("Đường dẫn không tồn tại.")

    st.markdown("---")
    process_btn = st.button("▶ XỬ LÝ DỮ LIỆU", type="primary")
    if st.button("⟲ Đặt lại hệ thống", type="secondary"): st.session_state.clear(); st.rerun()

if "results" not in st.session_state: st.session_state["results"] = []

if process_btn and files_to_process:
    if len(files_to_process) > CONFIG["MAX_IMAGES"]: st.error("Quá giới hạn ảnh."); st.stop()
    processed_results = []
    progress_bar = st.progress(0); status_text = st.empty()
    total = len(files_to_process)
    for i, file_input in enumerate(files_to_process):
        fname = os.path.basename(file_input) if isinstance(file_input, str) else file_input.name
        status_text.text(f"Đang xử lý: {fname}...")
        res = analyze_image(file_input)
        if res["status"] == "success": res["id"] = i; processed_results.append(res)
        progress_bar.progress((i+1)/total)
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
            st.download_button("📥 XUẤT BÁO CÁO EXCEL", buffer.getvalue(), "Analysed_Report_V20_1.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
elif not files_to_process: st.info("Hệ thống sẵn sàng. Vui lòng chọn nguồn dữ liệu.")
