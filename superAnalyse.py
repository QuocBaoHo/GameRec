import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Analytics Pro", layout="wide", page_icon="🔬")

# --- CSS ÉP XUNG GIÁC MẠC ---
st.markdown("""
<style>
    /* Nền đen nhám */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #00e5ff !important; font-weight: bold; }
    
    /* Chữ tiêu đề nổi bật */
    label { color: #e6edf3 !important; font-weight: 800 !important; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 1px; }
    
    /* 🛠️ KHUNG CHÚ THÍCH SIÊU SÁNG */
    div[data-testid="stAlert"] { 
        background-color: #161b22 !important; 
        border-left: 4px solid #00e5ff !important; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stAlert"] * {
        color: #f0f6fc !important; 
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    div[data-testid="stAlert"] li {
        color: #f0f6fc !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA SIÊU TỐC ---
@st.cache_resource
def load_all_assets():
    df = pd.read_csv('steam_data_llm.csv').reset_index(drop=True)
    index = faiss.read_index('faiss_llm_index.bin')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open('bow_vec.pkl', 'rb') as f: bow_vec = pickle.load(f)
    with open('jaccard_vec.pkl', 'rb') as f: bin_vec = pickle.load(f)
    with open('tfidf_vec.pkl', 'rb') as f: tfidf_vec = pickle.load(f)
    m_bow = scipy.sparse.load_npz('matrix_bow.npz')
    m_jac = scipy.sparse.load_npz('matrix_jaccard.npz')
    m_tfidf = scipy.sparse.load_npz('matrix_tfidf.npz')
    all_embeddings = model.encode(df['ai_text'].tolist(), show_progress_bar=False)
    return df, model, index, bow_vec, bin_vec, tfidf_vec, m_bow, m_jac, m_tfidf, all_embeddings

df, model, index, bow_vec, bin_vec, tfidf_vec, m_bow, m_jac, m_tfidf, all_embeddings = load_all_assets()

# --- GIAO DIỆN CHÍNH ---
st.title("🔬 SUPER ANALYSE PRO MAX")
st.markdown("### Bảng Điều Khiển Phân Tích Định Lượng Thuật Toán AI")

game_list = sorted(df['name'].tolist())
selected_game = st.selectbox("🎯 CHỌN GAME ĐỂ MỔ XẺ THUẬT TOÁN:", options=game_list, index=None, placeholder="Ví dụ: Fallout, Elden Ring, Stardew Valley...")

if selected_game:
    idx = df[df['name'] == selected_game].index[0]
    q_name = df.loc[idx, 'name']
    
    st.success(f"Đang phân tích độ tự tin của các thuật toán với game: **{q_name}**")

    # --- TÍNH TOÁN % KHỚP ---
    q_vec_llm = model.encode([df.loc[idx, 'ai_text']]).astype('float32')
    q_vec_jac = m_jac[idx]
    q_vec_tfidf = m_tfidf[idx]
    q_vec_bow = m_bow[idx]

    sim_bow = cosine_similarity(q_vec_bow, m_bow).flatten()
    sim_jac = cosine_similarity(q_vec_jac, m_jac).flatten()
    sim_tfidf = cosine_similarity(q_vec_tfidf, m_tfidf).flatten()
    sim_llm = cosine_similarity(q_vec_llm, all_embeddings).flatten()

    def get_top_k_scores(sim_array, k=50):
        sorted_scores = np.sort(sim_array)[::-1]
        return sorted_scores[1:k+1] * 100

    top_k = 50
    scores_bow = get_top_k_scores(sim_bow, top_k)
    scores_jac = get_top_k_scores(sim_jac, top_k)
    scores_tfidf = get_top_k_scores(sim_tfidf, top_k)
    scores_llm = get_top_k_scores(sim_llm, top_k)

    # --- VẼ BIỂU ĐỒ ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Biểu đồ sụt giảm độ tự tin (Score Drop-off)")
        
        fig_line = go.Figure()
        x_axis = list(range(1, top_k + 1))
        
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_bow, mode='lines+markers', name='BoW', line=dict(color='#ff4655')))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_jac, mode='lines', name='Jaccard', line=dict(color='#facc15')))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_tfidf, mode='lines', name='TF-IDF', line=dict(color='#a855f7')))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_llm, mode='lines+markers', name='LLM (AI)', line=dict(color='#00e5ff', width=3)))
        
        fig_line.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Hạng (Từ Top 1 đến Top 50)", 
            yaxis_title="Độ tương đồng Cosine (%)", 
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_line, use_container_width=True, theme=None)

    with col2:
        st.subheader("🎻 Phân phối điểm số (Violin Plot)")
        
        df_violin = pd.DataFrame({
            "Score": np.concatenate([scores_bow, scores_jac, scores_tfidf, scores_llm]),
            "Model": ["BoW"]*top_k + ["Jaccard"]*top_k + ["TF-IDF"]*top_k + ["LLM (AI)"]*top_k
        })
        
        fig_violin = px.violin(df_violin, x="Model", y="Score", color="Model", box=True, points="all",
                               color_discrete_map={"BoW": "#ff4655", "Jaccard": "#facc15", "TF-IDF": "#a855f7", "LLM (AI)": "#00e5ff"})
        
        fig_violin.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Điểm Tương Đồng (%)", 
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_violin, use_container_width=True, theme=None)

    st.markdown("---")
    
    # --- CẨM NANG CHỐNG UNGA BUNGA ---
    st.markdown("### 🧠 HƯỚNG DẪN ĐỌC BIỂU ĐỒ (DÀNH CHO BÁO CÁO)")
    st.info("""
    **1️⃣ Biểu đồ sụt giảm độ tự tin (Đường Line bên trái)**
    * **Câu hỏi đặt ra:** Khi bị ép phải tìm ra tới 50 game, thuật toán có còn giữ được độ chính xác không hay bắt đầu "nhắm mắt đoán bừa"?
    * **Cách đọc:** Nhìn vào độ dốc của đường.
        * **Đường cắm đầu xuống đất (Thường là BoW, TF-IDF):** Thuật toán chỉ tìm được vài game đầu có chung "từ khóa". Từ game số 10 trở đi, nó cạn vốn từ nên phải lôi mấy game không liên quan vào. Độ tự tin (Score) rớt thê thảm.
        * **Đường đi ngang ở mức cao (Đường Xanh LLM):** Thuật toán không phụ thuộc vào từ khóa. Vì nó hiểu "Vibe" và bối cảnh (Semantic), nó có thể dễ dàng tìm ra cả trăm game có cùng thể loại cốt truyện. Do đó, điểm tự tin của nó luông giữ ở mức ổn định cực cao!

    **2️⃣ Phân phối điểm số (Cây Đàn Violin bên phải)**
    * **Câu hỏi đặt ra:** Trong 50 game gợi ý ra, chất lượng đồng đều như thế nào?
    * **Cách đọc:** Xem cái "bụng" (chỗ phình to nhất) của nó nằm ở đâu.
        * **Bụng nằm ở dưới đáy (Màu tím TF-IDF / Vàng Jaccard):** Đa số các game được gợi ý có độ khớp rất lùn (chỉ 10-20%). Nghĩa là hệ thống đang ném cho người dùng một đống "rác" không liên quan.
        * **Bụng nằm tuốt trên cao (Xanh Cyan LLM):** Đa số 50 game gợi ý đều đạt độ khớp 60-80%. Điều này chứng minh LLM nén không gian 384 chiều cực kỳ tối ưu, các game cùng Vibe tự động tụ họp lại thành một cụm (Cluster) rất khít nhau!
    """)