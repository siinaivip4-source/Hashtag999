import streamlit as st
import pandas as pd
from PIL import Image
import torch
import clip
import io
import os
import time

# --- 1. CẤU HÌNH HỆ THỐNG & DATASET ---
st.set_page_config(page_title="AI Master V10 - Ultimate Batch", page_icon="🔥", layout="wide")

# Bộ lọc chuẩn theo yêu cầu của Đại sư huynh
CONFIG = {
    "STYLES": [
        "2D", "3D", "Cute", "Animeart", "Realism", "Aesthetic", "Cool", 
        "Fantasy", "Comic", "Horror", "Cyberpunk", "Lofi", "Minimalism"
    ],
    "COLORS": [
        "Red", "Blue", "Green", "Yellow", "Black", "White", "Pink", 
        "Purple", "Orange", "Pastel", "Neon", "Dark", "Bright"
    ],
    "EMOTIONS": [
        "Happy", "Sad", "Lonely", "Funny", "Gratitude", "Nostalgia", "Zenmode"
    ],
    "GENDERS": [
        "Male", "Female", "Non-binary", "Unisex"
    ]
}

# --- 2. CLASS: AI ENGINE (CLIP L14) ---
class AIEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.encoded_features = {} # Cache text features

    def load_model(self):
        if self.model is None:
            try:
                # Load model L14 xịn xò
                self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
                self.precompute_features()
                return True
            except Exception as e:
                st.error(f"Lỗi load AI: {e}")
                return False
        return True

    def precompute_features(self):
        """Mã hóa trước toàn bộ text để tốc độ nhanh gấp 4 lần"""
        with torch.no_grad():
            for category, labels in CONFIG.items():
                text_inputs = clip.tokenize([f"a {l} style/person/feeling" for l in labels]).to(self.device)
                features = self.model.encode_text(text_inputs)
                features /= features.norm(dim=-1, keepdim=True)
                self.encoded_features[category] = (features, labels)

    def analyze_image(self, image):
        """Trả về 1 dictionary chứa kết quả của 4 loại"""
        results = {}
        img_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            img_feat = self.model.encode_image(img_input)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            # Quét qua từng category (Style, Color, Emotion, Gender)
            for category, (text_feat, labels) in self.encoded_features.items():
                # Tính độ tương đồng
                similarity = (100.0 * img_feat @ text_feat.T).softmax(dim=-1)
                # Lấy cái cao nhất (Best match)
                idx = similarity[0].argmax().item()
                results[category] = labels[idx]
        
        return results

# Singleton Pattern cho AI
if 'ai_engine' not in st.session_state:
    st.session_state['ai_engine'] = AIEngine()

# --- 3. UI: QUẢN LÝ CUSTOM HASHTAG ---
def render_sidebar():
    st.sidebar.title("🔧 Cấu hình")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("🏷️ Custom Hashtags")
    st.sidebar.caption("Thêm tag cố định (VD: #Trending, #Hot)")
    
    # Logic quản lý thêm/xóa
    if "custom_tags" not in st.session_state:
        st.session_state["custom_tags"] = []

    # Input thêm tag
    new_tag = st.sidebar.text_input("Thêm hashtag mới (Không cần dấu #):")
    
    c1, c2 = st.sidebar.columns(2)
    if c1.button("➕ Thêm"):
        if len(st.session_state["custom_tags"]) >= 5:
            st.sidebar.error("⚠️ Tối đa 5 Custom Hashtag thôi huynh ơi!")
        elif new_tag and new_tag not in st.session_state["custom_tags"]:
            st.session_state["custom_tags"].append(new_tag)
            st.rerun()

    if c2.button("🗑️ Xóa All"):
        if len(st.session_state["custom_tags"]) > 0: # Đảm bảo logic tối thiểu
             st.session_state["custom_tags"] = []
             st.rerun()

    # Hiển thị danh sách hiện tại
    st.sidebar.write("Dataset hiện tại:")
    for tag in st.session_state["custom_tags"]:
        st.sidebar.markdown(f"- `#{tag}`")

    if len(st.session_state["custom_tags"]) == 0:
        st.sidebar.warning("⚠️ Đang không có Custom Tag nào.")

    return st.session_state["custom_tags"]

# --- 4. LOGIC XUẤT MYSQL ---
def generate_mysql_dump(df):
    """Tạo file .sql chứa lệnh INSERT"""
    table_name = "image_hashtags"
    sql_lines = []
    
    sql_lines.append(f"CREATE TABLE IF NOT EXISTS {table_name} (")
    sql_lines.append("    id INT AUTO_INCREMENT PRIMARY KEY,")
    sql_lines.append("    filename VARCHAR(255),")
    sql_lines.append("    style VARCHAR(50),")
    sql_lines.append("    color VARCHAR(50),")
    sql_lines.append("    emotion VARCHAR(50),")
    sql_lines.append("    gender VARCHAR(50),")
    sql_lines.append("    custom_tags TEXT,")
    sql_lines.append("    full_hashtags TEXT")
    sql_lines.append(");")
    sql_lines.append("")

    for index, row in df.iterrows():
        # Escape single quotes để tránh lỗi SQL Injection
        fname = str(row['Filename']).replace("'", "\\'")
        full_tags = str(row['Full_Hashtags']).replace("'", "\\'")
        
        val_str = f"('{fname}', '{row['Style']}', '{row['Color']}', '{row['Emotion']}', '{row['Gender']}', '{row['Custom']}', '{full_tags}')"
        sql_lines.append(f"INSERT INTO {table_name} (filename, style, color, emotion, gender, custom_tags, full_hashtags) VALUES {val_str};")
    
    return "\n".join(sql_lines)

# --- 5. MAIN APP ---
def main():
    st.title("🔥 AI MASTER V10 - BATCH PROCESSOR")
    st.markdown("### Hệ thống phân tích đa luồng: Style - Color - Emotion - Gender")
    
    # Load Custom Tags từ Sidebar
    custom_tags = render_sidebar()
    
    # Load AI
    engine = st.session_state['ai_engine']
    if not engine.load_model():
        st.stop()

    # TẠO TAB
    tab_batch, tab_manual = st.tabs(["📁 BATCH FOLDER", "👁️ VIEW MANUAL"])

    with tab_batch:
        st.markdown("#### 📂 Xử lý hàng loạt (Batch Processing)")
        st.info("💡 Chọn nhiều ảnh cùng lúc để giả lập xử lý cả thư mục.")
        
        uploaded_files = st.file_uploader("Kéo thả ảnh vào đây:", type=['jpg', 'png', 'jpeg', 'webp'], accept_multiple_files=True)
        
        if st.button("🚀 BẮT ĐẦU PHÂN TÍCH BATCH", type="primary"):
            if not uploaded_files:
                st.warning("⚠️ Huynh chưa chọn ảnh nào cả!")
            else:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file_obj in enumerate(uploaded_files):
                    status_text.text(f"Đang phân tích: {file_obj.name}...")
                    
                    try:
                        image = Image.open(file_obj).convert("RGB")
                        
                        # AI Phân tích 4 khía cạnh
                        ai_res = engine.analyze_image(image)
                        
                        # Tổng hợp Hashtag
                        # Logic: Mỗi ảnh 1 Style, 1 Color, 1 Emotion, 1 Gender + Custom Tags
                        tags_list = [
                            f"#{ai_res['STYLES']}",
                            f"#{ai_res['COLORS']}",
                            f"#{ai_res['EMOTIONS']}",
                            f"#{ai_res['GENDERS']}"
                        ]
                        # Thêm Custom tags
                        tags_list.extend([f"#{t}" for t in custom_tags])
                        
                        full_string = " ".join(tags_list)
                        
                        results.append({
                            "Filename": file_obj.name,
                            "Style": ai_res['STYLES'],
                            "Color": ai_res['COLORS'],
                            "Emotion": ai_res['EMOTIONS'],
                            "Gender": ai_res['GENDERS'],
                            "Custom": ", ".join(custom_tags),
                            "Full_Hashtags": full_string
                        })
                        
                    except Exception as e:
                        st.error(f"Lỗi file {file_obj.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.success(f"✅ Đã xử lý xong {len(uploaded_files)} ảnh!")
                progress_bar.empty()
                
                # --- HIỂN THỊ VÀ XUẤT FILE ---
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    
                    # 1. Xuất Excel
                    with c1:
                        buffer_excel = io.BytesIO()
                        with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Tải Excel Report (.xlsx)",
                            data=buffer_excel.getvalue(),
                            file_name="batch_result.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                    # 2. Xuất MySQL
                    with c2:
                        sql_content = generate_mysql_dump(df)
                        st.download_button(
                            label="🐬 Tải MySQL Dump (.sql)",
                            data=sql_content,
                            file_name="batch_result.sql",
                            mime="text/plain"
                        )

if __name__ == "__main__":
    main()
