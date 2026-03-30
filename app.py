import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse
import faiss
from sentence_transformers import SentenceTransformer
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Steam AI Recommender Pro", layout="wide", page_icon="🎮")

# --- ÉP GIAO DIỆN DARK MODE (CLEAN GAMER VIBE) ---
st.markdown("""
<style>
    /* Nền đen nhám */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff !important; text-shadow: none; font-weight: bold; }
    
    /* Chữ TÌM KIẾM nổi bật */
    label { color: #e6edf3 !important; font-weight: 800 !important; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Link màu Xanh Cyan */
    a { color: #00e5ff !important; text-decoration: none; font-weight: bold; }
    a:hover { color: #ff4655 !important; text-decoration: underline; }
    
    /* Ảnh game */
    div[data-testid="stImage"] img { border: 2px solid #30363d; border-radius: 8px; transition: transform 0.2s; }
    div[data-testid="stImage"] img:hover { transform: scale(1.03); border-color: #58a6ff; box-shadow: 0 0 10px #58a6ff; }
    .stProgress > div > div > div > div { background-color: #58a6ff; }
    div[data-testid="stAlert"] { background-color: #161b22; border-left: 4px solid #58a6ff; color: #e6edf3 !important; }
    table { color: #ffffff !important; }
    th { color: #58a6ff !important; }
    
    /* 🛠️ FIX CHÌM CHỮ TABS: Làm sáng chữ ở các Tab chưa chọn */
    button[data-baseweb="tab"] p {
        color: #8b949e !important; 
        font-size: 1.1rem;
        font-weight: bold;
    }
    /* Đổi màu Tab ĐANG chọn thành màu Cam Neon cho chiến */
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ff4655 !important; 
        text-shadow: 0 0 8px #ff4655;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_assets():
    df = pd.read_csv('steam_data_llm.csv').reset_index(drop=True)
    id_col = [c for c in df.columns if 'id' in c.lower()][0]
    index = faiss.read_index('faiss_llm_index.bin')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    with open('bow_vec.pkl', 'rb') as f: bow_vec = pickle.load(f)
    with open('jaccard_vec.pkl', 'rb') as f: bin_vec = pickle.load(f)
    with open('tfidf_vec.pkl', 'rb') as f: tfidf_vec = pickle.load(f)
    with open('knn_baseline.pkl', 'rb') as f: knn_model = pickle.load(f)
    
    m_bow = scipy.sparse.load_npz('matrix_bow.npz')
    m_jac = scipy.sparse.load_npz('matrix_jaccard.npz')
    m_tfidf = scipy.sparse.load_npz('matrix_tfidf.npz')
    all_embeddings = model.encode(df['ai_text'].tolist(), show_progress_bar=False)
    
    return df, model, index, bow_vec, bin_vec, tfidf_vec, knn_model, m_bow, m_jac, m_tfidf, all_embeddings, id_col

df, model, index, bow_vec, bin_vec, tfidf_vec, knn_model, m_bow, m_jac, m_tfidf, all_embeddings, id_col = load_all_assets()

# 🛠️ FIX API LỖI HÌNH ẢNH GÂY CRASH WEB
def fetch_steam_api(app_id):
    headers = {'User-Agent': 'Mozilla/5.0'}
    fallback_img = f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"
    url = f"https://store.steampowered.com/api/appdetails?appids={int(app_id)}&cc=vn&l=vietnamese"
    try:
        res = requests.get(url, headers=headers, timeout=5).json()
        if res and str(int(app_id)) in res and res[str(int(app_id))]['success']:
            d = res[str(int(app_id))]['data']
            
            is_free = d.get('is_free', False)
            p = d.get('price_overview', {})
            
            if is_free: price_str = "Miễn phí"
            elif p: price_str = p.get('final_formatted', 'N/A')
            else: price_str = "Ẩn giá / Check Store"
            
            # Bắt buộc trả về URL hợp lệ, nếu Steam trả None thì xài hàng dự phòng
            img_url = d.get('header_image')
            if not img_url: img_url = fallback_img
                
            return {
                "img": img_url,
                "price": price_str,
                "sale": p.get('discount_percent', 0) if p else 0,
                "url": f"https://store.steampowered.com/app/{int(app_id)}"
            }
    except: pass
    return {"img": fallback_img, "price": "Check Store", "sale": 0, "url": f"https://store.steampowered.com/app/{app_id}"}

# --- GIAO DIỆN CHÍNH ---
st.title("🚀 STEAM AI RECOMMENDER PRO")

# 🛠️ FIX UX DROPDOWN TÌM KIẾM CỰC MƯỢT
game_list = sorted(df['name'].tolist())
selected_game = st.selectbox(
    "🔍 TÌM KIẾM TỰA GAME:", 
    options=game_list,
    index=None, # Để trống lúc đầu
    placeholder="Gõ tên game vào đây... (VD: Fallout, Witcher, Resident Evil)" # Chữ tự lặn khi click
)

# Chỉ chạy phân tích khi có game được chọn
if selected_game:
    idx = df[df['name'] == selected_game].index[0]
    q_name = df.loc[idx, 'name']
    
    target_app_id = df.loc[idx, id_col]
    t_info = fetch_steam_api(target_app_id)
    
    st.markdown("### 🎯 MỤC TIÊU PHÂN TÍCH")
    t_col1, t_col2 = st.columns([1, 4])
    with t_col1:
        st.image(t_info['img'], use_container_width=True)
    with t_col2:
        st.markdown(f"## [{q_name}]({t_info['url']})")
        if t_info['sale'] > 0:
            st.write(f"🔥 Đang Sale: **-{t_info['sale']}%** | {t_info['price']}")
        else:
            st.write(f"💰 Giá hiện tại: **{t_info['price']}**")
    st.markdown("---")
    
    clean_title = re.sub(r'[^\w\s]', '', q_name).lower().split()
    ban_kw = max(clean_title, key=len) if clean_title else q_name.lower()

    q_vec_llm = model.encode([df.loc[idx, 'ai_text']]).astype('float32')
    q_vec_jac = m_jac[idx]
    q_vec_tfidf = m_tfidf[idx]
    q_vec_bow = m_bow[idx]

    tab1, tab2 = st.tabs(["🔥 GỢI Ý CÙNG VIBE", "🧪 ĐẤU TRƯỜNG THUẬT TOÁN"])

    with tab1:
        dist, ind = index.search(q_vec_llm, 30)
        recs = []
        for r_i in ind[0]:
            if r_i == idx: continue
            g_name_clean = re.sub(r'[^\w\s]', '', df.iloc[r_i]['name']).lower().split()
            if ban_kw in g_name_clean: continue
            recs.append(r_i)
            if len(recs) == 5: break

        cols = st.columns(5)
        for i, r_idx in enumerate(recs):
            g = df.iloc[r_idx]
            s = fetch_steam_api(g[id_col])
            vibe_score = float(cosine_similarity(q_vec_llm, all_embeddings[r_idx].reshape(1,-1))[0][0]) * 100
            
            with cols[i]:
                # Đảm bảo ko bao giờ bị lỗi NoneType chỗ này nữa
                st.image(s['img'], use_container_width=True)
                st.markdown(f"**[{g['name']}]({s['url']})**")
                st.progress(vibe_score / 100.0)
                st.caption(f"🧬 Khớp Vibe: {vibe_score:.1f}%")
                if s['sale'] > 0:
                    st.write(f"🔥 **-{s['sale']}%** | {s['price']}")
                else:
                    st.write(f"💰 {s['price']}")

    with tab2:
        st.subheader("📊 Đấu trường Thuật toán (Bản Overkill: Hình ảnh + Giá Live + Diversity)")
        
        # Helper check franchise (identical to before)
        def check_ho_hang_tab2(name_goc, name_check):
            clean_goc = re.sub(r'[^\w\s]', '', name_goc).lower().split()
            clean_check = re.sub(r'[^\w\s]', '', name_check).lower().split()
            kws = [w for w in clean_goc if len(w) > 3]
            if not kws: kws = clean_goc
            for kw in kws[:2]:
                if kw in clean_check: return True
            return False

        # Helper to get raw data instead of strings
        def get_diverse_raw_data(matrix, q_vec, is_llm=False):
            if is_llm: sims = cosine_similarity(q_vec, all_embeddings).flatten()
            else: sims = cosine_similarity(q_vec, matrix).flatten()
            best_indices = sims.argsort()[-31:-1][::-1] 
            
            recs_data = []
            franchise_count = 0
            
            for i in best_indices:
                if i == idx: continue
                g_df = df.iloc[i]
                g_name = g_df['name']
                is_franchise = check_ho_hang_tab2(q_name, g_name)
                
                # Filter logic (same as before)
                if is_franchise:
                    if franchise_count >= 2: continue
                    type_icon = "🔵"
                    franchise_count += 1
                else:
                    type_icon = "🟢"

                # Fetch API (image + price) - used to be formatted string, now kept raw
                s_info = fetch_steam_api(g_df[id_col])
                
                recs_data.append({
                    "name": g_name,
                    "score": sims[i] * 100,
                    "img": s_info['img'],
                    "price_info": s_info,
                    "icon": type_icon,
                    "url": s_info['url']
                })
                
                if len(recs_data) == 5: break
            return recs_data

        # Load data side by side
        with st.spinner("Đang triệu hồi 20 hình ảnh và giá tiền từ chiều không gian Steam... Đợi tí..."):
            # Get data dictionaries for each column
            data_bow = get_diverse_raw_data(m_bow, q_vec_bow)
            data_jac = get_diverse_raw_data(m_jac, q_vec_jac)
            data_tfidf = get_diverse_raw_data(m_tfidf, q_vec_tfidf)
            data_llm = get_diverse_raw_data(None, q_vec_llm, is_llm=True)
            
            # Helper function to render a single model column
            def render_model_column(model_name, data_list):
                st.markdown(f"#### {model_name}")
                st.markdown("---")
                if not data_list:
                    st.warning("Không có dữ liệu gợi ý.")
                    return
                
                for item in data_list:
                    # Dùng container border để tạo card cho từng game
                    with st.container(border=True):
                        st.image(item['img'], use_container_width=True)
                        st.markdown(f"**[{item['name']}]({item['url']})**")
                        
                        # Color type based on franchise
                        type_color = "#38bdf8" if item['icon'] == "🔵" else "#4ade80"
                        st.markdown(f"**{item['icon']} <span style='color:{type_color};'>{item['score']:.1f}% khớp</span>**", unsafe_allow_html=True)
                        
                        p_info = item['price_info']
                        if p_info['sale'] > 0:
                            st.write(f"🔥 **-{p_info['sale']}%** | {p_info['price']}")
                        else:
                            st.write(f"💰 {p_info['price']}")

            # Define 4 columns like a table grid
            cols_grid = st.columns(4)
            
            with cols_grid[0]:
                render_model_column("1. BoW", data_bow)
            with cols_grid[1]:
                render_model_column("2. Jaccard", data_jac)
            with cols_grid[2]:
                render_model_column("3. TF-IDF", data_tfidf)
            with cols_grid[3]:
                render_model_column("4. LLM (AI)", data_llm)

        st.markdown("---")
        # Detailed legend explaination as requested
        st.markdown("""
        ### 📘 GIẢI THÍCH CHUYÊN SÂU (% KHỚP & VIBE)
        
        Hệ thống so sánh độ tương đồng dựa trên thuật toán toán học **Cosine Similarity**. Tuy nhiên, bản chất vector là khác nhau giữa các mô hình:
        
        #### 1️⃣ CÁC MÔ HÌNH CỔ ĐIỂN (1, 2, 3)
        * **Bản chất:** Dựa trên **Lexical Match** (So khớp **mặt chữ**).
        * **% Khớp:** Đo lường sự trùng lặp từ vựng trong mô tả hoặc Tags. Điểm cao nghĩa là 2 game xài chung nhiều "từ khóa" (VD: mô tả của cả 2 game đều có chữ "Gun", "Zombie").
        * **Nhược điểm:** Không hiểu ngữ cảnh. Game A dùng "Gun", Game B dùng "Rifle" -> Các mô hình này chấm tương đồng thấp dù cùng là bắn súng.
        
        #### 2️⃣ MÔ HÌNH HIỆN ĐẠI LLM + FAISS (4)
        * **Bản chất:** Dựa trên **Semantic Match** (So khớp **ngữ nghĩa** hay "**Vibe**").
        * **% Vibe:** Mô hình trí tuệ nhân tạo nén toàn bộ mô tả thành 1 Vector đại diện cho bối cảnh và cốt truyện. `% Vibe` chính là độ gần nhau giữa game query và result trong không gian vector này.
        * **Ưu điểm:** Hiểu ngữ cảnh và từ đồng nghĩa. "Dùng súng" giống "xài súng", "Fallout" giống "Wasteland" về bối cảnh hậu tận thế dù không dùng chung từ khóa nào.
        """)
        st.info("""
            **💡 Chú thích bảng (Đã áp dụng Diversity Quota):**
            - 🔵 **Game cùng vũ trụ (Tối đa 2 slot):** Khẳng định model nhận diện được IP gốc.
            - 🟢 **Game khác vũ trụ (Tối thiểu 3 slot):** Khẳng định model hiểu được nội dung và bối cảnh (Vibe).
            => **Nhìn vào các điểm xanh 🟢:** Có thể thấy khi bị ép phải tìm game mới, các thuật toán cũ bắt đầu gợi ý lung tung, trong khi LLM vẫn giữ được chất lượng ngữ nghĩa cực kỳ ổn định!
            """)