"""
ENTERPRISE CONTENT TAGGER SYSTEM
Developed by: [SiinNoBox Team] - Coded by Tiểu Vy
Version: 22.1 (Fusion: Gemini 2.0 Flash + CLIP AI + Dynamic Sheets)
"""

import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import time
import json
import logging
import torch
import open_clip
from google import genai
from dotenv import load_dotenv

# --- 1. SYSTEM & SECURITY INIT ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Tải API Key từ file .env hoặc Secrets trên Web
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🚨 Lỗi Bảo Mật: Không tìm thấy GEMINI_API_KEY trong file .env hoặc Secrets!")
    st.stop()

# Khởi tạo Client theo chuẩn SDK mới của Google
client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. CONFIG & DICTIONARIES ---
CONFIG = {
    "MODEL_NAME": "ViT-B-32", 
    "PRETRAINED": "laion2b_s34b_b79k", 
    "THUMBNAIL_SIZE": (300, 600)
}

@st.cache_data(ttl=3600)
def load_google_sheet_objects():
    sheet_url = "https://docs.google.com/spreadsheets/d/1KBWiksmvnoDh7lJNijlwCTg9D124M06XAtMq9iTT3Gs/export?format=csv&gid=385629319"
    try:
        df = pd.read_csv(sheet_url)
        objects = df.iloc[:, 0].dropna().astype(str).tolist()
        return [obj.strip() for obj in objects if obj.strip()]
    except Exception as e:
        logger.error(f"Lỗi tải Google Sheet: {e}")
        return ["Character", "Landscape", "Animal", "Unknown"]

DYNAMIC_OBJECTS = load_google_sheet_objects()

AI_STYLES = ["2D", "3D", "Cute", "Animeart", "Realism", "Aesthetic", "Cool", "Fantasy", "Comic", "Horror", "Cyberpunk", "Lofi", "Minimalism", "Digitalart", "Cinematic", "Pixelart", "Scifi", "Vangoghart"]
AI_COLORS = ["Black", "White", "Blackandwhite", "Red", "Yellow", "Blue", "Green", "Pink", "Orange", "Pastel", "Hologram", "Vintage", "Colorful", "Neutral", "Light", "Dark", "Warm", "Cold", "Neon", "Gradient", "Purple", "Brown", "Grey"]

UI_MOODS = ["None", "Happy", "Sad", "Lonely", "Lovely", "Funny", "ZenMode"]
UI_GENDERS = ["None", "Male", "Female", "Non-binary", "Unisex"]

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

# --- 3. UI INIT ---
st.set_page_config(page_title="Enterprise Content Tagger V22.1", page_icon="💎", layout="wide")
st.title("💎 HỆ THỐNG PHÂN TÍCH & TỐI ƯU HÓA NỘI DUNG V22.1")
st.markdown("**Powered by Gemini 2.0 Flash + CLIP AI | Coded by Tiểu Vy**")
st.divider()

# --- 4. AI ENGINES ---
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
        logger.error(f"Critical Error loading CLIP: {e}")
        raise e

try:
    with st.spinner("Đang khởi tạo hệ thống AI (Gemini + CLIP)..."):
        model, preprocess, s_feat, c_feat, device = load_ai_engine()
except Exception as e:
    st.error(f"Lỗi khởi tạo CLIP: {e}"); st.stop()


def analyze_with_gemini_vision(image_obj, file_name, retries=3):
    """Gửi ảnh cho Gemini, in thẳng lỗi ra JSON nếu xịt"""
    prompt = f"""
    Bạn là một chuyên gia phân tích ảnh hệ thống. Hãy phân tích bức ảnh này và trả về kết quả DƯỚI DẠNG JSON hợp lệ.
    Bắt buộc phải chọn các giá trị từ danh sách sau:
    - object: Chọn 1 từ phù hợp nhất từ danh sách này: {DYNAMIC_OBJECTS}. Nếu không có gì hợp, trả về "Unknown".
    - mood: Chọn 1 từ từ: {UI_MOODS}
    - gender: Chọn 1 từ từ: {UI_GENDERS} (Nếu không có người, chọn "None")
    
    Định dạng JSON yêu cầu:
    {{
        "object": "...",
        "mood": "...",
        "gender": "..."
    }}
    Chỉ in ra JSON, không in thêm bất kỳ chữ nào khác.
    """
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[prompt, image_obj]
            )
            result_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(result_text)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                wait_time = (attempt + 1) * 20
                logger.warning(f"[Lỗi 429] Quá tải request. Đang đợi {wait_time}s... (Thử lại {attempt+1}/{retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Lỗi Gemini trên file {file_name}: {error_msg}")
                # In thẳng lỗi ra màn hình cho anh dễ thấy
                return {"object": f"Lỗi: {error_msg}", "mood": "None", "gender": "None"}
                
    return {"object": "Timeout/Error", "mood": "None", "gender": "None"}


def process_single_image(file_input):
    """Kết hợp cả Gemini (Ngữ nghĩa) và CLIP (Thẩm mỹ)"""
    try:
        filename = file_input.name
        original_img = Image.open(io.BytesIO(file_input.getvalue()))
        if original_img.mode != "RGB": original_img = original_img.convert("RGB")
        
        thumb = original_img.copy()
        thumb.thumbnail(CONFIG["THUMBNAIL_SIZE"])
        
        # 1. GỌI GEMINI (Lấy Object, Mood, Gender)
        ai_result = analyze_with_gemini_vision(original_img, filename)
        
        # 2. GỌI CLIP (Lấy Style, Color)
        input_img = preprocess(original_img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(input_img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
        s_probs = (100.0 * img_feat @ s_feat.T).softmax(dim=-1)
        c_probs = (100.0 * img_feat @ c_feat.T).softmax(dim=-1)
        
        s_idx = s_probs.argmax().item()
        c_idx = c_probs.argmax().item()
        
        return {
            "status": "success", 
            "filename": filename, 
            "image_obj": thumb, 
            "object": ai_result.get("object", "None"), 
            "mood": ai_result.get("mood", "None"), 
            "gender": ai_result.get("gender", "None"),
            "style": AI_STYLES[s_idx],
            "color": AI_COLORS[c_idx]
        }
    except Exception as e:
        return {"status": "error", "filename": file_input.name, "msg": str(e)}

# --- 5. GIAO DIỆN & LOGIC ---
with st.sidebar:
    st.header("Cấu hình & Dữ liệu")
    st.success(f"✅ Đã tải {len(DYNAMIC_OBJECTS)} Hashtags từ Google Sheets")
    uploaded_files = st.file_uploader(f"Kéo thả ảnh vào đây:", type=['png','jpg','jpeg','webp'], accept_multiple_files=True)
    process_btn = st.button("▶ XỬ LÝ DỮ LIỆU", type="primary")

if "results" not in st.session_state: st.session_state["results"] = []

if process_btn and uploaded_files:
    processed_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(uploaded_files)
    
    for i, file_input in enumerate(uploaded_files):
        status_text.text(f"Đang phân tích ảnh: {file_input.name}...")
        res = process_single_image(file_input)
        
        if res["status"] == "success": 
            res["id"] = i
            processed_results.append(res)
        else:
            st.error(f"Lỗi ảnh {res['filename']}: {res['msg']}")
            
        progress_bar.progress((i+1)/total)
        # Nghỉ 5s giữa các ảnh để né gậy 429 của Google
        time.sleep(5) 
        
    st.session_state["results"] = processed_results
    status_text.success("Xử lý hoàn tất!")
    progress_bar.empty()

# Hiển thị kết quả Đầy đủ 5 Tags
if st.session_state["results"]:
    st.subheader("KẾT QUẢ PHÂN TÍCH TỔNG HỢP")
    cols = st.columns(4)
    export_data = []
    
    for i, item in enumerate(st.session_state["results"]):
        with cols[i % 4]:
            with st.container(border=True):
                st.image(item["image_obj"], use_container_width=True)
                st.caption(item["filename"])
                st.write(f"🏷️ **Object:** {item['object']}")
                st.write(f"🎭 **Mood:** {item['mood']}")
                st.write(f"🚻 **Gender:** {item['gender']}")
                st.write(f"🎨 **Style:** {item['style']}")
                st.write(f"🖌️ **Color:** {item['color']}")
                
        export_data.append({
            "Tên tập tin": item["filename"],
            "Object": item["object"],
            "Mood": item["mood"],
            "Gender": item["gender"],
            "Style": item["style"],
            "Color": item["color"]
        })
        
    df = pd.DataFrame(export_data)
    buffer_xls = io.BytesIO()
    with pd.ExcelWriter(buffer_xls, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 TẢI EXCEL KẾT QUẢ", buffer_xls.getvalue(), "Report_V22_Fusion.xlsx")
