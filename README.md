Các bước thực hiện
Bước 1: Hiểu dataset
Dataset sử dụng là #nowplaying-RS.
Mục tiêu là xem dataset có những cột nào, cột nào dùng được cho recommender system.
Cần kiểm tra:
- Số dòng, số cột
- Tên các cột
- Kiểu dữ liệu
- Missing values
- Duplicate
Ý nghĩa: bước này giúp biết dữ liệu có sạch không và chọn được các cột quan trọng như user, song, thời gian nghe.
________________________________________
Bước 2: Data cleaning
Các việc cần làm:
- Chuyển cột date/time từ string sang datetime
- Xóa những cột null quá nhiều như place, geo, coordinates
- Xóa dòng thiếu user_id hoặc track_id/song_id
- Kiểm tra duplicate toàn bộ dòng
- Kiểm tra duplicate theo cặp user-song
Ý nghĩa: làm sạch dữ liệu trước khi đưa vào model.
Các cột như place, geo, coordinates nếu thiếu quá nhiều thì không giúp ích cho model, nên drop.
________________________________________
Bước 3: Tạo interaction data
Từ dataset gốc, chọn ra các cột chính:
user_id
track_id / song_id
date / timestamp
Sau đó tạo bảng interaction:
User A nghe Song 1
User A nghe Song 2
User B nghe Song 3
...
Nếu một user nghe cùng một bài nhiều lần, có thể xử lý bằng 2 cách:
Cách 1: Chỉ giữ 1 dòng, xem như user đã nghe bài đó
Cách 2: Đếm số lần nghe thành play_count
Nên chọn Cách 2 nếu muốn tốt hơn:
User nghe bài càng nhiều → mức độ quan tâm càng cao
________________________________________
Bước 4: Lọc user và song quá ít tương tác
Vì dataset lớn và rất sparse, nên lọc bớt:
Giữ user có ít nhất 5 bài đã nghe
Giữ bài hát được nghe ít nhất 5 lần hoặc bởi ít nhất 5 user
Ý nghĩa: nếu user hoặc song có quá ít dữ liệu, model rất khó học.
Bước này giúp dataset gọn hơn, sạch hơn và chạy nhanh hơn.
________________________________________
Bước 5: Encoding user_id và track_id
Model không hiểu ID dạng chữ, nên cần chuyển sang số.
Ví dụ:
user_abc → 0
user_xyz → 1

song_aaa → 0
song_bbb → 1
Dùng LabelEncoder để làm bước này.
Ý nghĩa: sau khi encode, ta mới tạo được ma trận user-item.
________________________________________
Bước 6: Tạo user-item matrix
Tạo ma trận:
Rows = users
Columns = songs
Value = 1 hoặc play_count
Ví dụ:
User	Song 1	Song 2	Song 3
A	1	0	1
B	0	1	0
C	1	1	0
Ý nghĩa: đây là dữ liệu đầu vào chính của recommender system.
________________________________________
Bước 7: Train baseline model
Làm baseline trước để có cái so sánh.
Nên có 2 baseline:
1. Popularity-based recommendation
2. SVD / Matrix Factorization
Popularity-based: recommend các bài hát phổ biến nhất.
SVD / Matrix Factorization: học embedding cho user và song từ user-item matrix.
Ý nghĩa: baseline giúp chứng minh model VAE có cải thiện hay không.
________________________________________
Bước 8: Matrix Factorization / SVD
Dùng SVD để tạo embedding:
User-item matrix → SVD → user embeddings + song embeddings
Ví dụ:
User vector: 64 dimensions
Song vector: 64 dimensions
Ý nghĩa: thay vì dùng ma trận rất lớn và sparse, ta biến user và song thành vector nhỏ gọn trong latent space.
________________________________________
Bước 9: Train VAE
VAE nhận user embedding làm input.
Pipeline:
User embedding
↓
Encoder
↓
Latent vector z
↓
Decoder
↓
Reconstructed user embedding
Ý nghĩa: VAE học biểu diễn sở thích ẩn của user.
Phần encoder giúp nén user vector thành latent preference, còn decoder tái tạo lại vector đó.
Toán chính cần giải thích:
Encoder tạo ra mean μ và variance σ
Sau đó sample latent vector z
Decoder dùng z để reconstruct user vector
________________________________________
Bước 10: Generate recommendation
Sau khi có reconstructed user vector từ VAE, so sánh nó với song embeddings bằng cosine similarity.
Reconstructed user vector
↓
Compare with all song vectors
↓
Select top-K songs
Ý nghĩa: bài hát nào có vector gần user vector nhất thì được recommend.
________________________________________
Bước 11: Tạo cold-start simulation
Vì dataset không nhất thiết có user mới thật, mình tự mô phỏng cold-start.
Cách làm:
Chọn một nhóm user
Chỉ giữ lại 1–3 bài đầu tiên họ nghe trong train
Ẩn các bài còn lại làm test
Model phải recommend lại các bài bị ẩn
Ví dụ:
User A nghe 20 bài
Train chỉ cho model thấy 3 bài
Test giữ 17 bài còn lại
Nếu model recommend trúng các bài trong 17 bài đó → đúng
Ý nghĩa: mô phỏng tình huống user mới hoặc user có rất ít lịch sử nghe nhạc.
________________________________________
Bước 12: Evaluation metrics
Dùng 3 metrics chính:
Precision@K
Recall@K
NDCG@K
Precision@K: trong top-K bài recommend, có bao nhiêu bài đúng.
Recall@K: trong các bài user thật sự nghe/thích, model tìm lại được bao nhiêu.
NDCG@K: bài đúng có nằm ở vị trí cao trong danh sách recommend không.
Ý nghĩa: giúp đánh giá chất lượng recommendation.
________________________________________
Bước 13: So sánh kết quả
So sánh các model:
Popularity-based
SVD / Matrix Factorization
VAE
Bảng kết quả có thể như:
Model	Precision@10	Recall@10	NDCG@10
Popularity	...	...	...
SVD	...	...	...
VAE	...	...	...
Ý nghĩa: chứng minh model nào hoạt động tốt hơn trong bài toán cold-start.
________________________________________
Bước 14: Visualisation
Nên vẽ một số biểu đồ:
- Top 10 bài hát phổ biến nhất
- Số lượng interaction theo user
- Số lượng interaction theo song
- Training loss của VAE
- Biểu đồ so sánh Precision/Recall/NDCG giữa các model
Ý nghĩa: giúp báo cáo đẹp hơn và dễ giải thích kết quả.
________________________________________
Bước 15: Viết báo cáo
Cấu trúc báo cáo nên gồm:
1. Introduction
2. Literature Review
3. Dataset and Preprocessing
4. Methodology
5. Experiments
6. Results and Discussion
7. Conclusion
Trong phần Methodology, tập trung giải thích:
- User-item matrix
- Matrix Factorization / SVD
- VAE Encoder and Decoder
- Cosine similarity
- Cold-start simulation
- Evaluation metrics

