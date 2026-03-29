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

@st.cache_resource
def load_all_assets():
    # Load Data
    df = pd.read_csv('steam_data_llm.csv').reset_index(drop=True)
    # Tự động check tên cột ID (appid, id, hay app_id)
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
    
    # Nạp sẵn Embeddings để tính toán cho mượt
    all_embeddings = model.encode(df['ai_text'].tolist(), show_progress_bar=False)
    
    return df, model, index, bow_vec, bin_vec, tfidf_vec, knn_model, m_bow, m_jac, m_tfidf, all_embeddings, id_col

# Triệu hồi bảo vật
df, model, index, bow_vec, bin_vec, tfidf_vec, knn_model, m_bow, m_jac, m_tfidf, all_embeddings, id_col = load_all_assets()

# --- HÀM GỌI API STEAM (ĐÃ FIX BLOCK) ---
def fetch_steam_api(app_id):
    # Giả danh trình duyệt để Steam ko block
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    url = f"https://store.steampowered.com/api/appdetails?appids={int(app_id)}&cc=vn&l=vietnamese"
    
    try:
        res = requests.get(url, headers=headers, timeout=10).json()
        if res and str(int(app_id)) in res and res[str(int(app_id))]['success']:
            data = res[str(int(app_id))]['data']
            p = data.get('price_overview', {})
            return {
                "img": data.get('header_image', ''),
                "price": p.get('final_formatted', 'Miễn phí / Chưa giá'),
                "sale": p.get('discount_percent', 0),
                "url": f"https://store.steampowered.com/app/{int(app_id)}"
            }
    except Exception as e:
        print(f"Lỗi gọi API cho ID {app_id}: {e}")
    
    # Header dự phòng nếu API fail
    return {
        "img": f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg",
        "price": "N/A (Check Store)",
        "sale": 0,
        "url": f"https://store.steampowered.com/app/{app_id}"
    }

# --- GIAO DIỆN ---
# --- GIAO DIỆN ---
st.title("🚀 STEAM AI RECOMMENDER PRO")

# Nâng cấp thanh tìm kiếm thành Dropdown Auto-complete
game_list = ["-- Bấm vào đây để gõ tìm hoặc cuộn chọn game --"] + sorted(df['name'].tolist())
selected_game = st.selectbox("🔍 Tìm kiếm chính xác tựa game bạn yêu thích:", options=game_list)

if selected_game != "-- Bấm vào đây để gõ tìm hoặc cuộn chọn game --":
    # Lấy chính xác index của game được chọn (Không sợ nhầm phần 1, 2, 3 nữa)
    idx = df[df['name'] == selected_game].index[0]
    q_name = df.loc[idx, 'name']
    
    st.success(f"🎯 Đang phân tích: **{q_name}**")
    
    # Lấy từ khóa để ban franchise (né sequel)
    clean_title = re.sub(r'[^\w\s]', '', q_name).lower().split()
    ban_kw = max(clean_title, key=len) if clean_title else q_name.lower()

    # Vector truy vấn
    q_vec_llm = model.encode([df.loc[idx, 'ai_text']]).astype('float32')
    q_vec_jac = m_jac[idx]
    q_vec_tfidf = m_tfidf[idx]
    q_vec_bow = m_bow[idx]

    tab1, tab2 = st.tabs(["🔥 GỢI Ý CÙNG VIBE", "🧪 ĐẤU TRƯỜNG THUẬT TOÁN"])

    with tab1:
        # Lấy 30 thằng để lọc dần
        dist, ind = index.search(q_vec_llm, 30)
        
        recs = []
        for r_i in ind[0]:
            if r_i == idx: continue
            
            # Hàm check franchise siêu việt (lột sạch dấu câu để so sánh)
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
                st.image(s['img'], use_container_width=True)
                st.markdown(f"**[{g['name']}]({s['url']})**")
                st.progress(vibe_score / 100.0)
                st.caption(f"🧬 Khớp Vibe: {vibe_score:.1f}%")
                if s['sale'] > 0:
                    st.write(f"🔥 **-{s['sale']}%** | {s['price']}")
                else:
                    st.write(f"💰 {s['price']}")

    with tab2:
        st.subheader("📊 Đấu trường Thuật toán (Đã bật Kiểm soát Đa dạng hóa)")
        
        def check_ho_hang(name_goc, name_check):
            clean_goc = re.sub(r'[^\w\s]', '', name_goc).lower().split()
            clean_check = re.sub(r'[^\w\s]', '', name_check).lower().split()
            kws = [w for w in clean_goc if len(w) > 3]
            if not kws: kws = clean_goc
            for kw in kws[:2]:
                if kw in clean_check: return True
            return False

        def get_diverse_top_5(matrix, q_vec, is_llm=False):
            if is_llm: sims = cosine_similarity(q_vec, all_embeddings).flatten()
            else: sims = cosine_similarity(q_vec, matrix).flatten()
            best_indices = sims.argsort()[-31:-1][::-1] 
            
            recs = []
            franchise_count = 0
            for i in best_indices:
                if i == idx: continue
                g_name = df.iloc[i]['name']
                is_franchise = check_ho_hang(q_name, g_name)
                
                if is_franchise:
                    if franchise_count < 2: 
                        recs.append(f"🔵 {g_name} ({sims[i]*100:.1f}%)")
                        franchise_count += 1
                else:
                    recs.append(f"🟢 {g_name} ({sims[i]*100:.1f}%)")
                if len(recs) == 5: break
            return recs

        comparison_data = {
            "Hạng": ["Top 1", "Top 2", "Top 3", "Top 4", "Top 5"],
            "1. Bag of Words": get_diverse_top_5(m_bow, q_vec_bow),
            "2. Jaccard (Tags)": get_diverse_top_5(m_jac, q_vec_jac),
            "3. TF-IDF + KNN": get_diverse_top_5(m_tfidf, q_vec_tfidf),
            "4. LLM + FAISS (AI)": get_diverse_top_5(None, q_vec_llm, is_llm=True)
        }
        
        st.table(pd.DataFrame(comparison_data))
        st.info("""
            **💡 Chú thích bảng (Đã áp dụng Diversity Quota):**
            - 🔵 **Game cùng vũ trụ (Tối đa 2 slot):** Khẳng định model nhận diện được IP gốc.
            - 🟢 **Game khác vũ trụ (Tối thiểu 3 slot):** Khẳng định model hiểu được nội dung và bối cảnh (Vibe).
            => **Nhìn vào các điểm xanh 🟢:** Có thể thấy khi bị ép phải tìm game mới, các thuật toán cũ bắt đầu gợi ý lung tung, trong khi LLM vẫn giữ được chất lượng ngữ nghĩa cực kỳ ổn định!
            """)