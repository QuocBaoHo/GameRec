# 🚀 Steam AI Recommender Pro

Hệ thống gợi ý game Steam ứng dụng Trí tuệ Nhân tạo (AI) và Xử lý Ngôn ngữ Tự nhiên (NLP). Khác với các hệ thống truyền thống chỉ so khớp từ khóa (Lexical Matching), dự án này sử dụng Mô hình Ngôn ngữ Lớn (LLM) để hiểu ngữ nghĩa và đề xuất game dựa trên "Vibe" (Không khí/Trải nghiệm cốt truyện).

---

## ✨ Tính năng nổi bật
* **Real-time Steam API:** Cập nhật giá tiền (VNĐ), khuyến mãi và ảnh bìa trực tiếp từ máy chủ Steam.
* **Semantic Search (LLM + FAISS):** Gợi ý game dựa trên sự tương đồng về ngữ nghĩa cốt truyện, không phụ thuộc vào Tags.
* **Diversity Quota (Bộ lọc Đa dạng hóa):** Chống lại hiện tượng "Echo Chamber" (Gợi ý loanh quanh các phần 1, 2, 3 của cùng một game) bằng cách ép thuật toán phân bổ tối đa 2 slot cho IP cũ (🔵) và tối thiểu 3 slot cho các IP hoàn toàn mới (🟢).

---

## 🧠 Giải phẫu 4 Thuật toán Gợi ý

Dự án là một đấu trường so sánh trực tiếp giữa 3 mô hình Cổ điển (Dựa trên Thống kê/Từ khóa) và 1 mô hình Hiện đại (Dựa trên Ngôn ngữ học sâu):

### 1. Bag of Words (BoW)
* **Cách hoạt động:** Biến đoạn mô tả game thành một cái túi đựng từ. Nó chỉ đếm xem một từ xuất hiện bao nhiêu lần, hoàn toàn phớt lờ ngữ pháp hay thứ tự từ.
* **Điểm yếu:** Đếm từ mù quáng. Hai game có cùng chữ "Gun" nhiều lần sẽ bị coi là giống nhau, dù một game là bắn súng sinh tồn, một game là bắn súng nhịp điệu.

### 2. Jaccard Similarity (Đếm Tags)
* **Cách hoạt động:** Thuật toán so sánh tập hợp. Nó lấy số lượng Tags chung của 2 game chia cho tổng số Tags của cả 2 game.
* **Điểm yếu:** Phụ thuộc 100% vào việc người dùng dán nhãn (Tag) trên Steam. Dễ bị đánh lừa bởi các game spam Tag (Ví dụ: Game xếp hình nhưng gắn mác "Cyberpunk", "RPG").

### 3. TF-IDF + KNN
* **Cách hoạt động:** Bản nâng cấp của BoW. Nó phạt những từ xuất hiện quá nhiều (như "the", "and", "game") và thưởng điểm cho những từ hiếm, mang tính đặc trưng cao (như "post-apocalyptic", "mutant"). Sau đó dùng thuật toán K-Nearest Neighbors (KNN) để tìm các game gần nhất.
* **Điểm yếu:** Vẫn là so khớp mặt chữ (Lexical). Nếu Game A dùng từ "Zombie" và Game B dùng từ "Undead", TF-IDF sẽ chấm điểm tương đồng là 0 vì hai chữ này viết khác nhau.

### 4. LLM + FAISS (AI Chân ái)
* **Cách hoạt động:** Dùng mô hình Transformer (`all-MiniLM-L6-v2`) nén toàn bộ mô tả, thể loại, và cốt truyện của game thành một Vector đặc 384 chiều. Nó hiểu được ngữ cảnh và từ đồng nghĩa. FAISS của Meta được dùng để quét cực nhanh trong không gian nhiều chiều này.
* **Điểm mạnh:** Hiểu được "Zombie" và "Undead" là cùng một chủ đề. Gợi ý chuẩn xác dựa trên trải nghiệm thực sự (Vibe) thay vì cái mác bên ngoài.

---

## 📊 Định nghĩa các chỉ số: "% Khớp" và "% Vibe" là gì?

Trong Tab "Đấu trường thuật toán", các con số % đại diện cho **Cosine Similarity** (Độ tương đồng Cosine). Tuy nhiên, bản chất của chúng hoàn toàn khác nhau tùy thuộc vào mô hình:

### 1. "% Khớp" (Áp dụng cho BoW, Jaccard, TF-IDF)
* **Định nghĩa:** Là sự trùng lặp về **MẶT CHỮ (Lexical Match)**.
* **Bản chất:** Nó đo lường xem hai game xài chung bao nhiêu từ vựng hoặc chung bao nhiêu cái Tag. Điểm % cao ở đây chỉ có nghĩa là người viết mô tả của 2 game này có "vốn từ vựng" giống hệt nhau.

### 2. "% Vibe" (Áp dụng riêng cho LLM)
* **Định nghĩa:** Là sự tương đồng về **NGỮ NGHĨA và KHÔNG KHÍ (Semantic Match)**.
* **Bản chất toán học:** Cốt truyện của mỗi game được AI chuyển thành một điểm trong không gian tọa độ 384 chiều. `% Vibe` chính là độ hẹp của góc tạo bởi hai điểm đó. Góc càng nhỏ (Cosine tiến về 1), % Vibe càng cao.
* **Ý nghĩa thực tiễn:** Hai game có thể không xài chung bất kỳ từ khóa nào, nhưng nếu nội dung của chúng đều nói về sự tuyệt vọng trong thế giới hoang tàn, AI sẽ xếp chúng ở cạnh nhau. Đó chính là "Vibe"!

---

## 🛠️ Hướng dẫn cài đặt (Localhost)

1. Clone repo này về máy.
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install streamlit pandas numpy faiss-cpu scikit-learn sentence-transformers scipy requests
3. Chạy ứng dụng Streamlit:
   ```bash
   streamlit run app.py