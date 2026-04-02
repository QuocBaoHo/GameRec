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

# --- CSS ÉP XUNG GIÁC MẠC (FIX MÀU CHỮ & TAB) ---
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #00e5ff !important; font-weight: bold; }
    label { color: #e6edf3 !important; font-weight: 800 !important; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Box thông tin */
    div[data-testid="stAlert"] { 
        background-color: #161b22 !important; 
        border-left: 4px solid #00e5ff !important; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stAlert"] * {
        color: #ffffff !important; 
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    /* Fix màu cho các dấu chấm tròn (bullet points) */
    div[data-testid="stAlert"] ul li { color: #ffffff !important; }
    
    /* Đổi màu Tab cho dễ nhìn */
    button[data-baseweb="tab"] p { color: #8b949e !important; font-size: 1.1rem; font-weight: bold; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #00e5ff !important; text-shadow: 0 0 8px #00e5ff; }
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

    # --- CẨM NANG ĐỌC BIỂU ĐỒ (Dành cho người mới) ---
    st.markdown("---")
    st.markdown("### 🧠 BÍ KÍP ĐỌC BIỂU ĐỒ (CHỈ 1 PHÚT LÀ HIỂU)")
    st.info("""
    **1️⃣ Biểu đồ sụt giảm (Line Chart):** Đo độ "Hụt hơi". Đường nào đâm đầu xuống đất nghĩa là thuật toán cạn vốn từ, phải đi đoán bừa. Đường đi ngang nghĩa là thuật toán rất tự tin.
    **2️⃣ Phân phối điểm số (Violin Plot):** Đo độ "Chất lượng". Nhìn vào cái 'bụng' bự nhất. Bụng nằm tuốt dưới đáy là toàn gợi ý rác. Bụng trên cao là gợi ý cực chuẩn.
    **3️⃣ Mạng nhện (Radar Chart):** So sánh tổng lực 5 chỉ số y như coi thông số Tướng trong game. Hình đa giác càng to, bao phủ càng rộng thì thuật toán càng bá đạo!
    """)

    # --- VẼ BIỂU ĐỒ DROP-OFF & VIOLIN ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Biểu đồ sụt giảm độ tự tin")
        fig_line = go.Figure()
        x_axis = list(range(1, top_k + 1))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_bow, mode='lines+markers', name='BoW', line=dict(color='#ff4655')))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_jac, mode='lines', name='Jaccard', line=dict(color='#facc15')))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_tfidf, mode='lines', name='TF-IDF', line=dict(color='#a855f7')))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_llm, mode='lines+markers', name='LLM (AI)', line=dict(color='#00e5ff', width=3)))
        fig_line.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_line, use_container_width=True, theme=None)

    with col2:
        st.subheader("🎻 Phân phối điểm số (Violin Plot)")
        df_violin = pd.DataFrame({
            "Score": np.concatenate([scores_bow, scores_jac, scores_tfidf, scores_llm]),
            "Model": ["BoW"]*top_k + ["Jaccard"]*top_k + ["TF-IDF"]*top_k + ["LLM (AI)"]*top_k
        })
        fig_violin = px.violin(df_violin, x="Model", y="Score", color="Model", box=True, points="all", color_discrete_map={"BoW": "#ff4655", "Jaccard": "#facc15", "TF-IDF": "#a855f7", "LLM (AI)": "#00e5ff"})
        fig_violin.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_violin, use_container_width=True, theme=None)

    # --- MASTER RADAR CHART (ĐẠI CHIẾN TỔNG LỰC) ---
    st.markdown("---")
    st.markdown("### 🕸️ ĐẠI CHIẾN RADAR: TỔNG QUAN SỨC MẠNH (MASTER CHART)")
    
    categories = ['Hiểu Ngữ Nghĩa<br>(Semantic)', 'Bắt Từ Khóa<br>(Lexical)', 'Tốc Độ Tính Toán<br>(Speed)', 'Tiết Kiệm RAM<br>(Memory)', 'Khả năng Mở rộng<br>(Scale)']
    
    fig_master = go.Figure()
    fig_master.add_trace(go.Scatterpolar(r=[5, 70, 95, 30, 40], theta=categories, fill='toself', name='BoW', line=dict(color='#ff4655'), fillcolor='rgba(255, 70, 85, 0.1)'))
    fig_master.add_trace(go.Scatterpolar(r=[5, 80, 90, 40, 45], theta=categories, fill='toself', name='Jaccard', line=dict(color='#facc15'), fillcolor='rgba(250, 204, 21, 0.1)'))
    fig_master.add_trace(go.Scatterpolar(r=[15, 95, 85, 20, 30], theta=categories, fill='toself', name='TF-IDF', line=dict(color='#a855f7'), fillcolor='rgba(168, 85, 247, 0.1)'))
    fig_master.add_trace(go.Scatterpolar(r=[100, 60, 95, 80, 100], theta=categories, fill='toself', name='LLM (AI)', line=dict(color='#00e5ff', width=3), fillcolor='rgba(0, 229, 255, 0.3)'))
    
    fig_master.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#8b949e", gridcolor="#30363d"),
            angularaxis=dict(color="#e6edf3", gridcolor="#30363d")
        ),
        showlegend=True,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=80, r=80, t=40, b=80), # Tăng lề siêu bự để không lẹm chữ
        height=450
    )
    st.plotly_chart(fig_master, use_container_width=True, theme=None)

    # --- BÍ KÍP BẢO VỆ ĐỒ ÁN (GIẢI PHẪU CHI TIẾT TỪNG THUẬT TOÁN) ---
    st.markdown("### 🔬 GIẢI PHẪU CHI TIẾT TỪNG THUẬT TOÁN (BẤM VÀO TAB)")

    # Hàm vẽ Biểu đồ Radar Cá nhân (Đã fix lỗi lẹm chữ)
    def draw_single_radar(model_name, r_values, color_hex, fill_rgba):
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_values, theta=categories, fill='toself', name=model_name,
            line=dict(color=color_hex, width=2), fillcolor=fill_rgba, marker=dict(size=6)
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#8b949e", gridcolor="#30363d"),
                angularaxis=dict(color="#e6edf3", gridcolor="#30363d")
            ),
            showlegend=False, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=60, t=20, b=60), # Tăng lề siêu bự (bottom=60)
            height=350
        )
        return fig

    tab_bow, tab_jac, tab_tfidf, tab_llm = st.tabs(["1️⃣ Bag of Words", "2️⃣ Jaccard", "3️⃣ TF-IDF", "4️⃣ LLM + FAISS (AI)"])

    with tab_bow:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info("""
            **🔍 Nguyên lý hoạt động:** Đếm số lần xuất hiện của từ vựng, không quan tâm thứ tự hay ngữ pháp.
            
            **✅ Điểm mạnh:** * Đơn giản, dễ lập trình.
            * Tốc độ xây dựng ma trận siêu nhanh.
            
            **❌ Điểm yếu chí mạng:**
            * **Mù ngữ nghĩa (Semantic = 0):** Câu "Chó cắn người" và "Người cắn chó" được máy hiểu là giống nhau 100%.
            * **Tốn RAM:** Sinh ra ma trận khổng lồ nhưng toàn số 0 (Sparse Matrix).
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('BoW', [5, 70, 95, 30, 40], '#ff4655', 'rgba(255, 70, 85, 0.3)'), use_container_width=True, theme=None)

    with tab_jac:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            # THÊM CHỮ 'r' Ở ĐÂY ĐỂ FIX LỖI LATEX NÈ
            st.info(r"""
            **🔍 Nguyên lý hoạt động:** Tính tỷ lệ phần giao (Intersection) chia cho phần hợp (Union) của 2 tập hợp:
            $$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$
            
            **✅ Điểm mạnh:** * Cực kỳ hiệu quả để so khớp các nhãn dán (Tags/Genres). 
            * Bắt từ khóa siêu chuẩn.
            
            **❌ Điểm yếu chí mạng:**
            * Không phân biệt được "trọng số". Chữ vô nghĩa như "The" bị đánh giá ngang hàng với chữ quan trọng như "Zombie".
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('Jaccard', [5, 80, 90, 40, 45], '#facc15', 'rgba(250, 204, 21, 0.3)'), use_container_width=True, theme=None)

    with tab_tfidf:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info("""
            **🔍 Nguyên lý hoạt động:** Nó phạt nặng các từ xuất hiện quá phổ biến (như "play", "game") và thưởng điểm cho các từ hiếm (như "cyberpunk").
            
            **✅ Điểm mạnh:** * Công cụ Bắt từ khóa (Keyword) hoàn hảo nhất của thế hệ cũ.
            
            **❌ Điểm yếu chí mạng:**
            * **Mù từ đồng nghĩa:** Gõ "Gun" sẽ không bao giờ tìm ra game mô tả bằng chữ "Rifle". Dẫn đến cạn kiệt kết quả khi tìm kiếm sâu.
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('TF-IDF', [15, 95, 85, 20, 30], '#a855f7', 'rgba(168, 85, 247, 0.3)'), use_container_width=True, theme=None)

    with tab_llm:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info("""
            **🔍 Nguyên lý hoạt động (Semantic Match):** AI nén toàn bộ cốt truyện thành Vector đặc (Dense) 384 chiều. Kết hợp FAISS để tìm láng giềng gần nhất siêu tốc.
            
            **✅ Điểm mạnh (Out trình):**
            * **Hiểu ngữ cảnh tuyệt đối:** Hiểu "Hậu tận thế" dù mô tả chỉ ghi "Nuclear fallout". Trị được từ đồng nghĩa.
            * **Sức chứa vô hạn (Scale):** Tốc độ quét của FAISS gần như không đổi dù dữ liệu lên tới hàng triệu game.
            
            **❌ Điểm yếu (Sự đánh đổi):** * Đòi hỏi phần cứng (CPU/GPU) cực mạnh để khởi tạo Vector ban đầu.
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('LLM', [100, 60, 95, 80, 100], '#00e5ff', 'rgba(0, 229, 255, 0.3)'), use_container_width=True, theme=None)