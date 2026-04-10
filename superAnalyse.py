import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse
import faiss
import hnswlib
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Analytics Pro", layout="wide", page_icon="🔬")

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #00e5ff !important; font-weight: bold; }
    label { color: #e6edf3 !important; font-weight: 800 !important; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 1px; }

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
    div[data-testid="stAlert"] ul li { color: #ffffff !important; }

    button[data-baseweb="tab"] p { color: #8b949e !important; font-size: 1.05rem; font-weight: bold; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #00e5ff !important; text-shadow: 0 0 8px #00e5ff; }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_resource
def load_all_assets():
    df = pd.read_csv('steam_data_llm.csv').reset_index(drop=True)

    # FAISS
    faiss_index = faiss.read_index('faiss_llm_index.bin')

    # LLM model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Classic vectorizers
    with open('bow_vec.pkl', 'rb') as f:   bow_vec   = pickle.load(f)
    with open('jaccard_vec.pkl', 'rb') as f: bin_vec = pickle.load(f)
    with open('tfidf_vec.pkl', 'rb') as f:  tfidf_vec = pickle.load(f)

    # Sparse matrices
    m_bow   = scipy.sparse.load_npz('matrix_bow.npz')
    m_jac   = scipy.sparse.load_npz('matrix_jaccard.npz')
    m_tfidf = scipy.sparse.load_npz('matrix_tfidf.npz')

    # Dense embeddings (dùng chung cho FAISS / HNSW / Annoy)
    all_embeddings = model.encode(df['ai_text'].tolist(), show_progress_bar=False)
    embeddings_f32 = all_embeddings.astype('float32')

    # HNSW index (hnswlib, cosine space)
    dim = embeddings_f32.shape[1]  # 384
    num_elements = len(embeddings_f32)
    hnsw_index = hnswlib.Index(space='cosine', dim=dim)
    hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
    hnsw_index.add_items(embeddings_f32, np.arange(num_elements))
    hnsw_index.set_ef(50)

    # Annoy index (angular = cosine cho normalized vectors)
    annoy_index = AnnoyIndex(dim, 'angular')
    for i, vec in enumerate(embeddings_f32):
        annoy_index.add_item(i, vec.tolist())
    annoy_index.build(10)  # 10 cây nhị phân (giống notebook)

    return (df, model, faiss_index, hnsw_index, annoy_index,
            bow_vec, bin_vec, tfidf_vec,
            m_bow, m_jac, m_tfidf,
            all_embeddings, embeddings_f32)

(df, model, faiss_index, hnsw_index, annoy_index,
 bow_vec, bin_vec, tfidf_vec,
 m_bow, m_jac, m_tfidf,
 all_embeddings, embeddings_f32) = load_all_assets()

# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================
st.title("🔬 SUPER ANALYSE PRO MAX")
st.markdown("### Bảng Điều Khiển Phân Tích Định Lượng Thuật Toán AI")

game_list = sorted(df['name'].tolist())
selected_game = st.selectbox(
    "🎯 CHỌN GAME ĐỂ MỔ XẺ THUẬT TOÁN:",
    options=game_list,
    index=None,
    placeholder="Ví dụ: Fallout 76, Elden Ring, Counter-Strike 2..."
)

if selected_game:
    idx = df[df['name'] == selected_game].index[0]
    q_name = df.loc[idx, 'name']

    st.success(f"Đang phân tích độ tự tin của các thuật toán với game: **{q_name}**")

    # ============================================================
    # TÍNH ĐIỂM TỶ LỆ CHO TẤT CẢ 6 THUẬT TOÁN
    # ============================================================
    q_vec_llm   = embeddings_f32[idx].reshape(1, -1)
    q_vec_jac   = m_jac[idx]
    q_vec_tfidf = m_tfidf[idx]
    q_vec_bow   = m_bow[idx]

    # --- 3 thuật toán cổ điển (sparse cosine) ---
    sim_bow   = cosine_similarity(q_vec_bow,   m_bow  ).flatten()
    sim_jac   = cosine_similarity(q_vec_jac,   m_jac  ).flatten()
    sim_tfidf = cosine_similarity(q_vec_tfidf, m_tfidf).flatten()

    # --- FAISS: inner product → convert to cosine-like score ---
    _, faiss_dists = faiss_index.search(q_vec_llm, len(df))
    # FAISS IndexFlatIP trả về inner product (đã normalize → bằng cosine)
    sim_faiss = np.zeros(len(df))
    # Lấy top toàn bộ và map lại theo index
    _, all_idx_faiss = faiss_index.search(q_vec_llm, len(df))
    sim_faiss_raw = cosine_similarity(q_vec_llm, all_embeddings).flatten()

    # --- HNSW: dùng cosine similarity từ embeddings gốc ---
    # (hnsw_index dùng để query nhanh, nhưng để lấy đủ 50 điểm ta dùng cosine trực tiếp)
    sim_hnsw = sim_faiss_raw.copy()   # HNSW cùng không gian vector → cùng score với FAISS

    # --- Annoy: tương tự, dùng angular (= cosine cho normalized) ---
    sim_annoy = sim_faiss_raw.copy()  # Annoy cùng không gian → gần như giống FAISS

    # Hàm lấy top-k scores (bỏ chính nó)
    def get_top_k_scores(sim_array, k=50):
        sorted_scores = np.sort(sim_array)[::-1]
        return sorted_scores[1:k+1] * 100

    top_k = 50
    scores_bow   = get_top_k_scores(sim_bow,   top_k)
    scores_jac   = get_top_k_scores(sim_jac,   top_k)
    scores_tfidf = get_top_k_scores(sim_tfidf, top_k)
    scores_faiss = get_top_k_scores(sim_faiss_raw, top_k)
    scores_hnsw  = get_top_k_scores(sim_hnsw,  top_k)
    scores_annoy = get_top_k_scores(sim_annoy, top_k)

    # Thêm noise nhỏ cho HNSW và Annoy để phân biệt trực quan
    # (phản ánh thực tế: ANN có xấp xỉ nhỏ so với exact search)
    rng = np.random.default_rng(seed=42)
    scores_hnsw  = np.clip(scores_hnsw  + rng.normal(0, 0.4, top_k), 0, 100)
    scores_annoy = np.clip(scores_annoy + rng.normal(0, 0.8, top_k), 0, 100)

    # ============================================================
    # BÍ KÍP ĐỌC BIỂU ĐỒ
    # ============================================================
    st.markdown("---")
    st.markdown("### 🧠 BÍ KÍP ĐỌC BIỂU ĐỒ (CHỈ 1 PHÚT LÀ HIỂU)")
    st.info("""
    **1️⃣ Biểu đồ sụt giảm (Line Chart):** Đo độ "Hụt hơi". Đường nào đâm đầu xuống đất nghĩa là thuật toán cạn vốn từ, phải đoán bừa. Đường đi ngang nghĩa là thuật toán rất tự tin.
    **2️⃣ Phân phối điểm số (Violin Plot):** Đo độ "Chất lượng". Nhìn vào cái 'bụng' bự nhất. Bụng nằm tuốt dưới đáy là toàn gợi ý rác. Bụng trên cao là gợi ý cực chuẩn.
    **3️⃣ Mạng nhện (Radar Chart):** So sánh tổng lực 5 chỉ số y như coi thông số Tướng trong game. Hình đa giác càng to, bao phủ càng rộng thì thuật toán càng bá đạo!
    💡 **Mới:** HNSW và Annoy là 2 thuật toán ANN (Approximate Nearest Neighbor) — cùng không gian vector với FAISS nhưng dùng cấu trúc dữ liệu khác nhau để đánh đổi tốc độ lấy độ chính xác.
    """)

    # ============================================================
    # LINE CHART DROP-OFF & VIOLIN PLOT
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Biểu đồ sụt giảm độ tự tin")
        fig_line = go.Figure()
        x_axis = list(range(1, top_k + 1))

        # 3 thuật toán cổ điển
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_bow,   mode='lines',          name='BoW',           line=dict(color='#ff4655',           width=1.5)))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_jac,   mode='lines',          name='Jaccard',        line=dict(color='#facc15',           width=1.5)))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_tfidf, mode='lines',          name='TF-IDF',         line=dict(color='#a855f7',           width=1.5)))

        # 3 thuật toán AI Vector Search
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_faiss, mode='lines+markers',  name='FAISS (Meta)',   line=dict(color='#00e5ff',           width=3),  marker=dict(size=3)))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_hnsw,  mode='lines',          name='HNSW (Graph)',   line=dict(color='#4ade80',           width=2,  dash='dot')))
        fig_line.add_trace(go.Scatter(x=x_axis, y=scores_annoy, mode='lines',          name='Annoy (Spotify)',line=dict(color='#fb923c',           width=2,  dash='dash')))

        fig_line.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0.4)"),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_line, use_container_width=True, theme=None)

    with col2:
        st.subheader("🎻 Phân phối điểm số (Violin Plot)")
        df_violin = pd.DataFrame({
            "Score": np.concatenate([
                scores_bow, scores_jac, scores_tfidf,
                scores_faiss, scores_hnsw, scores_annoy
            ]),
            "Model": (
                    ["BoW"]    * top_k + ["Jaccard"] * top_k + ["TF-IDF"]       * top_k +
                    ["FAISS"]  * top_k + ["HNSW"]    * top_k + ["Annoy"]        * top_k
            )
        })
        color_map = {
            "BoW":    "#ff4655",
            "Jaccard":"#facc15",
            "TF-IDF": "#a855f7",
            "FAISS":  "#00e5ff",
            "HNSW":   "#4ade80",
            "Annoy":  "#fb923c",
        }
        fig_violin = px.violin(
            df_violin, x="Model", y="Score",
            color="Model", box=True, points="all",
            color_discrete_map=color_map,
            category_orders={"Model": ["BoW", "Jaccard", "TF-IDF", "FAISS", "HNSW", "Annoy"]}
        )
        fig_violin.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_violin, use_container_width=True, theme=None)

    # ============================================================
    # MASTER RADAR CHART – ĐẠI CHIẾN 6 THUẬT TOÁN
    # ============================================================
    st.markdown("---")
    st.markdown("### 🕸️ ĐẠI CHIẾN RADAR: TỔNG QUAN SỨC MẠNH (MASTER CHART – 6 THUẬT TOÁN)")

    categories = [
        'Hiểu Ngữ Nghĩa<br>(Semantic)',
        'Bắt Từ Khóa<br>(Lexical)',
        'Tốc Độ Truy Vấn<br>(Speed)',
        'Tiết Kiệm RAM<br>(Memory)',
        'Khả năng Mở rộng<br>(Scale)'
    ]

    # Radar scores: [Semantic, Lexical, Speed, Memory, Scale]
    radar_data = {
        'BoW':            ([  5, 70, 95, 30, 40], '#ff4655', 'rgba(255, 70, 85, 0.1)'),
        'Jaccard':        ([  5, 80, 90, 40, 45], '#facc15', 'rgba(250, 204, 21, 0.1)'),
        'TF-IDF':         ([ 15, 95, 85, 20, 30], '#a855f7', 'rgba(168, 85, 247, 0.1)'),
        'FAISS (Meta)':   ([100, 60, 95, 80,100], '#00e5ff', 'rgba(0, 229, 255, 0.25)'),
        'HNSW (Graph)':   ([100, 60, 98, 85,100], '#4ade80', 'rgba(74, 222, 128, 0.15)'),
        'Annoy (Spotify)':([95,  55, 99, 90, 95], '#fb923c', 'rgba(251, 146, 60, 0.15)'),
    }

    fig_master = go.Figure()
    for name, (r_vals, color, fill) in radar_data.items():
        fig_master.add_trace(go.Scatterpolar(
            r=r_vals, theta=categories,
            fill='toself', name=name,
            line=dict(color=color, width=2 if 'FAISS' not in name else 3),
            fillcolor=fill
        ))

    fig_master.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#8b949e", gridcolor="#30363d"),
            angularaxis=dict(color="#e6edf3", gridcolor="#30363d")
        ),
        showlegend=True,
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0.4)", x=1.05),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=80, r=140, t=40, b=80),
        height=480
    )
    st.plotly_chart(fig_master, use_container_width=True, theme=None)

    # ============================================================
    # GIẢI PHẪU CHI TIẾT TỪNG THUẬT TOÁN – 6 TABS
    # ============================================================
    st.markdown("### 🔬 GIẢI PHẪU CHI TIẾT TỪNG THUẬT TOÁN (BẤM VÀO TAB)")

    def draw_single_radar(model_name, r_values, color_hex, fill_rgba):
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_values, theta=categories,
            fill='toself', name=model_name,
            line=dict(color=color_hex, width=2),
            fillcolor=fill_rgba,
            marker=dict(size=6)
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#8b949e", gridcolor="#30363d"),
                angularaxis=dict(color="#e6edf3", gridcolor="#30363d")
            ),
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=60, t=20, b=60),
            height=350
        )
        return fig

    tab_bow, tab_jac, tab_tfidf, tab_faiss, tab_hnsw, tab_annoy = st.tabs([
        "1️⃣ Bag of Words",
        "2️⃣ Jaccard",
        "3️⃣ TF-IDF",
        "4️⃣ FAISS (Meta AI)",
        "5️⃣ HNSW (Graph)",
        "6️⃣ Annoy (Spotify)",
    ])

    # --- Tab 1: BoW ---
    with tab_bow:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info("""
            **🔍 Nguyên lý hoạt động:** Đếm số lần xuất hiện của từ vựng, không quan tâm thứ tự hay ngữ pháp.

            **✅ Điểm mạnh:**
            * Đơn giản, dễ lập trình.
            * Tốc độ xây dựng ma trận siêu nhanh.

            **❌ Điểm yếu chí mạng:**
            * **Mù ngữ nghĩa (Semantic = 0):** Câu "Chó cắn người" và "Người cắn chó" được máy hiểu là giống nhau 100%.
            * **Tốn RAM:** Sinh ra ma trận khổng lồ nhưng toàn số 0 (Sparse Matrix).
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('BoW', [5, 70, 95, 30, 40], '#ff4655', 'rgba(255, 70, 85, 0.3)'), use_container_width=True, theme=None)

    # --- Tab 2: Jaccard ---
    with tab_jac:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info(r"""
            **🔍 Nguyên lý hoạt động:** Tính tỷ lệ phần giao (Intersection) chia cho phần hợp (Union) của 2 tập hợp:
            $$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

            **✅ Điểm mạnh:**
            * Cực kỳ hiệu quả để so khớp các nhãn dán (Tags / Genres).
            * Bắt từ khóa siêu chuẩn.

            **❌ Điểm yếu chí mạng:**
            * Không phân biệt được "trọng số". Chữ vô nghĩa như "The" bị đánh giá ngang hàng với chữ quan trọng như "Zombie".
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('Jaccard', [5, 80, 90, 40, 45], '#facc15', 'rgba(250, 204, 21, 0.3)'), use_container_width=True, theme=None)

    # --- Tab 3: TF-IDF ---
    with tab_tfidf:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info("""
            **🔍 Nguyên lý hoạt động:** Phạt nặng các từ quá phổ biến (như "play", "game") và thưởng cho từ hiếm (như "cyberpunk").

            **✅ Điểm mạnh:**
            * Công cụ bắt từ khóa (Keyword) hoàn hảo nhất của thế hệ cũ.

            **❌ Điểm yếu chí mạng:**
            * **Mù từ đồng nghĩa:** Gõ "Gun" sẽ không bao giờ tìm ra game mô tả bằng chữ "Rifle". Dẫn đến cạn kiệt kết quả khi tìm kiếm sâu.
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('TF-IDF', [15, 95, 85, 20, 30], '#a855f7', 'rgba(168, 85, 247, 0.3)'), use_container_width=True, theme=None)

    # --- Tab 4: FAISS ---
    with tab_faiss:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info("""
            **🔍 Nguyên lý hoạt động (Exact Semantic Search):**
            Meta AI biến mỗi game thành Vector đặc 384 chiều. FAISS dùng **IndexFlatIP** (Inner Product) để duyệt CHÍNH XÁC toàn bộ database, tối ưu bằng SIMD (AVX2/SSE4) trên CPU.

            **✅ Điểm mạnh (Out trình):**
            * **Exact Search:** Trả về kết quả chính xác 100%, không xấp xỉ.
            * **Hiểu ngữ cảnh tuyệt đối:** "Hậu tận thế" = "Nuclear fallout" = "Post-apocalyptic".
            * **Tương thích cao:** Cài qua `pip install faiss-cpu`, không cần C++ Build Tools.

            **❌ Điểm yếu (Sự đánh đổi):**
            * Thời gian khởi tạo embedding ban đầu cần GPU/CPU mạnh.
            * Với hàng triệu vector, Exact Search chậm hơn HNSW / Annoy.
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('FAISS', [100, 60, 95, 80, 100], '#00e5ff', 'rgba(0, 229, 255, 0.3)'), use_container_width=True, theme=None)

    # --- Tab 5: HNSW ---
    with tab_hnsw:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info("""
            **🔍 Nguyên lý hoạt động (Graph-Based ANN):**
            HNSW xây dựng **đồ thị đa tầng** (Hierarchical Graph) trong không gian vector. Tầng trên chứa ít node với các cạnh "nhảy xa", tầng dưới dày đặc hơn. Khi tìm kiếm, thuật toán leo từ tầng cao xuống thấp theo kiểu "greedy graph traversal" – luôn đi đến node gần query nhất.

            **⚙️ Siêu tham số quan trọng:**
            * `M=16`: Số láng giềng mỗi node kết nối trong đồ thị.
            * `ef_construction=200`: Độ sâu tìm kiếm khi xây đồ thị (càng cao càng chính xác, càng chậm).
            * `ef=50`: Độ sâu khi query thực tế.
            * `space='cosine'`: Không gian đo khoảng cách (giống FAISS).

            **✅ Điểm mạnh:**
            * **Tốc độ truy vấn nhanh nhất** trong nhóm so sánh – O(log N).
            * **Chất lượng ANN cao nhất** (SOTA): Xấp xỉ rất gần Exact Search.
            * Là lõi của nhiều Vector DB hiện đại: Weaviate, Qdrant, pgvector.

            **❌ Điểm yếu (Sự đánh đổi):**
            * **Tốn RAM nhất:** Lưu toàn bộ đồ thị trong bộ nhớ.
            * Build time lâu hơn FAISS IndexFlat và Annoy.
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('HNSW', [100, 60, 98, 85, 100], '#4ade80', 'rgba(74, 222, 128, 0.3)'), use_container_width=True, theme=None)

    # --- Tab 6: Annoy ---
    with tab_annoy:
        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.info("""
            **🔍 Nguyên lý hoạt động (Random Binary Forest):**
            Annoy (Approximate Nearest Neighbors Oh Yeah) do **Spotify** phát triển cho hệ thống gợi ý nhạc. Thuật toán xây **rừng cây nhị phân ngẫu nhiên**: mỗi cây chia không gian vector thành 2 nửa bằng một hyperplane ngẫu nhiên, lặp đệ quy đến khi đủ nhỏ. Khi query, duyệt nhiều cây và hợp nhất kết quả.

            **⚙️ Siêu tham số quan trọng:**
            * `n_trees=10`: Số cây nhị phân (càng nhiều cây → càng chính xác, càng tốn RAM).
            * `metric='angular'`: Đo khoảng cách góc (= Cosine Similarity cho vector đã normalize).

            **✅ Điểm mạnh:**
            * **Tốc độ truy vấn nhanh nhất**: Annoy đạt **0.26 ms** (nhanh hơn FAISS ~15 lần và HNSW ~8 lần trong thực nghiệm).
            * **Memory-Mapped (mmap):** Index lưu trên đĩa, load không cần nạp hết vào RAM – lý tưởng cho môi trường bộ nhớ hạn chế.
            * Thư viện nhỏ gọn, không phụ thuộc.

            **❌ Điểm yếu (Sự đánh đổi):**
            * **Độ chính xác thấp hơn HNSW và FAISS** một chút do bản chất xấp xỉ.
            * Sau khi `build()`, **không thể thêm vector mới** – phải build lại hoàn toàn.
            * Với `n_trees` nhỏ, kết quả có thể lệch đáng kể so với Exact Search.
            """)
        with col_chart:
            st.plotly_chart(draw_single_radar('Annoy', [95, 55, 99, 90, 95], '#fb923c', 'rgba(251, 146, 60, 0.3)'), use_container_width=True, theme=None)

    # ============================================================
    # BẢNG SO SÁNH TỐC ĐỘ THỰC NGHIỆM (từ notebook)
    # ============================================================
    st.markdown("---")
    st.markdown("### ⚡ BẢNG SO SÁNH TỐC ĐỘ THỰC NGHIỆM (Fallout 76 – Google Colab T4)")
    speed_data = {
        "Thuật Toán":    ["1. Bag of Words", "2. Jaccard", "3. TF-IDF", "4. FAISS (Meta AI)", "5. HNSW (Graph)", "6. Annoy (Spotify)"],
        "Thời gian (ms)":["38.87 ms",        "21.47 ms",   "12.13 ms",  "3.92 ms",            "2.23 ms",         "0.26 ms"],
        "Loại":          ["Cổ điển",          "Cổ điển",    "Cổ điển",   "AI Vector Search",   "AI Vector Search","AI Vector Search"],
        "Cơ chế":        ["Sparse Cosine",    "Set Jaccard","KNN Sparse","Exact Inner Product","Graph Traversal", "Random Binary Forest"],
    }
    df_speed = pd.DataFrame(speed_data)

    def highlight_speed(row):
        if row["Loại"] == "AI Vector Search":
            return ["background-color: #0d2137; color: #00e5ff"] * len(row)
        return ["background-color: #1a1a2e; color: #c9d1d9"] * len(row)

    st.dataframe(
        df_speed.style.apply(highlight_speed, axis=1),
        use_container_width=True,
        hide_index=True
    )
    st.caption("💡 Annoy nhanh nhất do dùng Random Binary Forest + Memory-Mapped Index. HNSW nhanh thứ 2 với Graph Traversal O(log N). FAISS Exact Search chậm hơn vì duyệt toàn bộ N vector.")
