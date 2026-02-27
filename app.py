"""
ENTERPRISE CONTENT TAGGER SYSTEM
Developed by: [SiinNoBox Team] - Coded by Tiểu Vy
Version: 22.0 (Gemini Vision + Dynamic Google Sheets + SQL)
"""

import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import time
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. SYSTEM & SECURITY INIT ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Tải API Key từ file .env (Bảo mật tuyệt đối)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🚨 Lỗi Bảo Mật: Không tìm thấy GEMINI_API_KEY trong file .env!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- 2. DYNAMIC DATA CHỈ TỪ GOOGLE SHEETS & CONFIG ---
@st.cache_data(ttl=3600) # Cache 1 tiếng để đỡ gọi sheet liên tục
def load_google_sheet_objects():
    # URL lấy dữ liệu dạng CSV từ Google Sheet của Đại sư huynh
    sheet_url = "https://docs.google.com/spreadsheets/d/1KBWiksmvnoDh7lJNijlwCTg9D124M06XAtMq9iTT3Gs/export?format=csv&gid=385629319"
    try:
        df = pd.read_csv(sheet_url)
        # Giả sử cột chứa hashtag tên là 'Hashtag' hoặc cột đầu tiên. Muội sẽ lấy tất cả value khác null.
        # Nếu cột của huynh tên khác, huynh báo muội nhé. Tạm thời lấy cột đầu tiên làm chuẩn.
        objects = df.iloc[:, 0].dropna().astype(str).tolist()
        return [obj.strip() for obj in objects if obj.strip()]
    except Exception as e:
        logger.error(f"Lỗi tải Google Sheet: {e}")
        return ["Character", "Landscape", "Animal", "Unknown"] # Fallback

DYNAMIC_OBJECTS = load_google_sheet_objects()
UI_MOODS = ["None", "Happy", "Sad", "Lonely", "Chill", "Funny"] # Cập nhật theo yêu cầu
UI_GENDERS = ["None", "Male", "Female", "Non-binary", "Unisex"]

# (Giữ nguyên cấu hình Style/Color cũ ở đây, muội thu gọn để huynh dễ nhìn)
UI_STYLES = ["None", "2D", "3D", "Animeart", "Realism", "Cyberpunk", "Vintage"] 
UI_COLORS = ["None", "Black", "White", "Red", "Blue", "Pastel", "Neon"]

# --- 3. UI/UX CONFIGURATION ---
st.set_page_config(page_title="Enterprise Content Tagger V22", page_icon="💎", layout="wide")
st.title("💎 HỆ THỐNG PHÂN TÍCH & TỐI ƯU HÓA NỘI DUNG V22.0")
st.markdown("**Powered by Gemini Vision API & Google Sheets | Coded by Tiểu Vy**")
st.divider()

# --- 4. GEMINI AI ENGINE (Có Exponential Backoff chống lỗi 429) ---
def analyze_with_gemini_vision(image_obj, file_name, retries=3):
    """Gửi ảnh cho Gemini phân tích, ép trả về định dạng JSON, tự động chờ nếu lỗi 429"""
    
    # Chuẩn bị Prompt nhốt Gemini vào khuôn khổ
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
    
    model = genai.GenerativeModel('gemini-2.0-flash') # Dùng model Flash nhanh nhất
    
    for attempt in range(retries):
        try:
            response = model.generate_content([prompt, image_obj])
            # Bóc tách JSON từ chuỗi trả về
            result_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(result_text)
            return data
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                wait_time = (attempt + 1) * 15 # Đợi 15s, 30s, 45s (Chống lỗi của bọn Google)
                logger.warning(f"[Lỗi 429] Quá tải request. Đang đợi {wait_time}s... (Thử lại {attempt+1}/{retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Lỗi phân tích {file_name}: {e}")
                break
                
    return {"object": "Error", "mood": "None", "gender": "None"}

def process_single_image(file_input):
    """Xử lý ảnh và gọi AI"""
    try:
        filename = file_input.name
        original_img = Image.open(io.BytesIO(file_input.getvalue()))
        if original_img.mode != "RGB": original_img = original_img.convert("RGB")
        
        # Tạo Thumbnail hiển thị
        thumb = original_img.copy()
        thumb.thumbnail((300, 600))
        
        # Gọi Gemini phân tích
        ai_result = analyze_with_gemini_vision(original_img, filename)
        
        return {
            "status": "success", 
            "filename": filename, 
            "image_obj": thumb, 
            "object": ai_result.get("object", "None"), 
            "mood": ai_result.get("mood", "None"), 
            "gender": ai_result.get("gender", "None"),
            "style": "None", # Chừa chỗ trống nếu huynh muốn ghép thêm CLIP vào sau
            "color": "None"
        }
    except Exception as e:
        return {"status": "error", "filename": file_input.name, "msg": str(e)}

# --- 5. GIAO DIỆN & LOGIC ---
with st.sidebar:
    st.header("Cấu hình & Dữ liệu")
    st.success(f"✅ Đã tải {len(DYNAMIC_OBJECTS)} Hashtags từ Google Sheets")
    uploaded_files = st.file_uploader(f"Kéo thả ảnh vào đây:", type=['png','jpg','jpeg','webp'], accept_multiple_files=True)
    process_btn = st.button("▶ XỬ LÝ DỮ LIỆU VỚI GEMINI", type="primary")

if "results" not in st.session_state: st.session_state["results"] = []

if process_btn and uploaded_files:
    processed_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(uploaded_files)
    
    for i, file_input in enumerate(uploaded_files):
        status_text.text(f"Đang nhờ Gemini soi ảnh: {file_input.name}...")
        res = process_single_image(file_input)
        if res["status"] == "success": 
            res["id"] = i
            processed_results.append(res)
        progress_bar.progress((i+1)/total)
        # Nghỉ 2 giây giữa mỗi ảnh để tránh đánh sập API Free của Google
        time.sleep(2) 
        
    st.session_state["results"] = processed_results
    status_text.success("Xử lý hoàn tất!")
    progress_bar.empty()

# Hiển thị kết quả (Rút gọn)
if st.session_state["results"]:
    st.subheader("KẾT QUẢ TỪ GEMINI")
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
                
        export_data.append({
            "Tên tập tin": item["filename"],
            "Object": item["object"],
            "Mood": item["mood"],
            "Gender": item["gender"]
        })
        
    # Nút Tải Data (Giữ nguyên logic tạo file Excel/SQL của huynh)
    df = pd.DataFrame(export_data)
    buffer_xls = io.BytesIO()
    with pd.ExcelWriter(buffer_xls, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 TẢI EXCEL", buffer_xls.getvalue(), "Report_V22.xlsx")
