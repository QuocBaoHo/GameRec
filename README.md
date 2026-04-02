# 🚀 Steam AI Recommender Pro (Overkill Edition)

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AI Model](https://img.shields.io/badge/AI-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Hệ thống gợi ý game Steam thế hệ mới ứng dụng **Trí tuệ Nhân tạo (AI)** và **Xử lý Ngôn ngữ Tự nhiên (NLP)**. Dự án không chỉ dừng lại ở việc tạo ra một ứng dụng thực tế cho Game thủ, mà còn cung cấp một Bảng điều khiển Phân tích Dữ liệu (Analytics Dashboard) chuyên sâu để chứng minh sự vượt trội của AI so với các thuật toán cổ điển.

---

## 📌 Mục lục
- [✨ Tính năng Nổi bật](#-tính-năng-nổi-bật)
- [🛠️ Công nghệ Sử dụng](#-công-nghệ-sử-dụng)
- [🧠 Cơ sở Toán học & Thuật toán](#-cơ-sở-toán-học--thuật-toán)
- [📂 Cấu trúc Dự án](#-cấu-trúc-dự-án)
- [🚀 Hướng dẫn Cài đặt & Khởi chạy](#-hướng-dẫn-cài-đặt--khởi-chạy)
- [📊 Phân hệ Phân tích AI](#-phân-hệ-phân-tích-ai)

---

## ✨ Tính năng Nổi bật

Dự án được thiết kế với tư duy "Overkill" - vượt xa các hệ thống gợi ý thông thường:

### 1. Phân hệ Ứng dụng: `app.py` (Giao diện Gamer)
*   **Cyberpunk Dark Mode UI:** Giao diện tối ưu trải nghiệm người dùng với các hiệu ứng Neon, bảng màu đen nhám chuyên nghiệp.
*   **Real-time Steam API:** Tự động lấy dữ liệu giá tiền (VNĐ), % Khuyến mãi và Ảnh bìa trực tiếp từ máy chủ Steam theo thời gian thực.
*   **Diversity Quota (Bộ lọc Đa dạng hóa):** Khắc phục triệt để hiện tượng "Bong bóng lọc" (Echo Chamber) bằng cách:
    *   🔵 Giới hạn tối đa 2 slot cho game cùng IP/Franchise.
    *   🟢 Ưu tiên tối thiểu 3 slot cho các tựa game mới lạ, độc lập.

### 2. Phân hệ Phân tích: `superAnalyse.py` (Bảng điều khiển AI)
*   **🧠 Bí kíp đọc biểu đồ:** Hộp hướng dẫn tích hợp giải thích 3 loại biểu đồ chính — giúp người mới bắt đầu hiểu ngay trong 1 phút.
*   **📉 Score Drop-off Chart (Line Chart):** Đo lường mức độ "đuối sức" (cạn vốn từ) của từng thuật toán khi tìm kiếm đến Top 50. Đường LLM luôn đi ngang ổn định, trong khi BoW/TF-IDF cắm đầu xuống đất.
*   **🎻 Violin Plot:** Trực quan hóa phân phối chất lượng điểm số — chứng minh LLM tạo ra các cụm game đồng nhất hơn hẳn các thuật toán cổ điển.
*   **🕸️ Master Radar Chart (Mới):** Biểu đồ mạng nhện tổng lực so sánh đồng thời cả 4 thuật toán trên **5 chỉ số**: Hiểu Ngữ Nghĩa (Semantic), Bắt Từ Khóa (Lexical), Tốc Độ (Speed), Tiết Kiệm RAM (Memory), Khả năng Mở rộng (Scale).
*   **🔬 Giải phẫu chi tiết từng thuật toán (4 Tab - Mới):** Mỗi tab (BoW, Jaccard, TF-IDF, LLM+FAISS) bao gồm:
    *   Radar Chart cá nhân thể hiện điểm mạnh/yếu đặc trưng.
    *   Giải thích nguyên lý hoạt động, ưu điểm & điểm yếu chí mạng bằng ngôn ngữ dễ hiểu.

---

## 🛠️ Công nghệ Sử dụng

| Thành phần | Công nghệ / Thư viện |
| :--- | :--- |
| **Giao diện** | Streamlit (Python-based Web Framework) |
| **AI Model** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **Data Science** | Pandas, Numpy, Scikit-learn, Scipy |
| **Visualization** | Plotly (Interactive Charts) |
| **API** | Steam Web API (Requests) |

---

## 🧠 Cơ sở Toán học & Thuật toán

Dự án là một đấu trường so sánh trực tiếp giữa **3 mô hình Lexical Match** (BoW, Jaccard, TF-IDF) và **1 mô hình Semantic Match** (LLM kết hợp FAISS).

Tất cả các mô hình đều được đánh giá chung trên một hệ quy chiếu là **Độ tương đồng Cosine (Cosine Similarity)**:

$$\text{sim}(A, B) = \cos(\theta) = \frac{A \cdot B}{||A|| ||B||}$$

*   **% Khớp (Mô hình Cổ điển):** Đại diện cho sự trùng lặp về **MẶT CHỮ**. Điểm cao chỉ có nghĩa là người viết mô tả của 2 game có chung "vốn từ vựng".
*   **% Vibe (Mô hình LLM):** Đại diện cho sự tương đồng về **NGỮ NGHĨA**. Cốt truyện được nén thành vector đặc 384 chiều. Góc $\theta$ giữa 2 vector càng nhỏ, 2 game càng có chung "Vibe" bối cảnh, dù không dùng chung bất kỳ từ khóa nào.

---

## 📂 Cấu trúc Dự án

```text
d:\GameRec/
├── app.py                # Ứng dụng chính cho người dùng (Gamer UI)
├── superAnalyse.py       # Bảng điều khiển phân tích thuật toán (AI Analytics)
├── steam_data_llm.csv    # Dataset thông tin game Steam
├── faiss_llm_index.bin   # Index Vector cho tìm kiếm ngữ nghĩa siêu tốc
├── *.pkl                 # Các model Vectorizer đã được train (BoW, TF-IDF, Jaccard)
├── *.npz                 # Ma trận thưa (Sparse Matrix) lưu trữ đặc trưng game
└── README.md             # Tài liệu dự án
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Cài đặt môi trường
Mở Terminal (PowerShell/CMD) và cài đặt các thư viện cần thiết:

```bash
pip install streamlit pandas numpy faiss-cpu scikit-learn sentence-transformers scipy requests plotly
```

### Bước 2: Khởi chạy ứng dụng
Dự án chạy song song hai phân hệ trên hai cổng khác nhau:

1.  **Giao diện Người dùng (Mặc định: Port 8501):**
    ```bash
    python -m streamlit run app.py
    ```

2.  **Giao diện Phân tích AI (Cổng 8502):**
    ```bash
    python -m streamlit run superAnalyse.py --server.port 8502
    ```

---

## 📊 Phân hệ Phân tích AI

Bảng điều khiển `superAnalyse.py` giúp bạn hiểu tại sao AI lại "thông minh" hơn bằng 5 công cụ trực quan:

| Biểu đồ | Ý nghĩa |
| :--- | :--- |
| **📉 Line Chart (Drop-off)** | Đường LLM đi ngang = tự tin đến Top 50. BoW/TF-IDF cắm đầu = cạn vốn từ. |
| **🎻 Violin Plot** | "Bụng" càng nằm trên cao = gợi ý càng chất lượng. LLM luôn thắng. |
| **🕸️ Master Radar Chart** | Đại chiến 4vs4 trên 5 chỉ số — nhìn hình đa giác nào to nhất là biết ngay. |
| **🔬 Tab BoW / Jaccard / TF-IDF** | Radar cá nhân + giải phẫu chi tiết điểm mạnh, điểm yếu từng thuật toán. |
| **🤖 Tab LLM + FAISS (AI)** | Chứng minh sức mạnh của Semantic Vector 384 chiều kết hợp tìm kiếm FAISS. |



---
*Dự án được phát triển với mục tiêu học tập và nghiên cứu ứng dụng AI trong Hệ thống Gợi ý (Recommendation Systems).*
