# Học Thống Kê - Phúc, Quang, Phượng
### Đề Tài: Text Classification. - Xây dựng chatbox Hỗ trợ khách hàng.
---
### Thành viên nhóm:
- 22120270 - Bùi Hồng Phúc.
- 
- 
---
### ý tưởng cụ thể:
1. Xây dựng model dựa trên data của bản thân dùng để phân loại cảm xúc (sử dụng PhoBert) -> Tích Cực, Tiêu cực, Trung tính.
2. Dựa vào model đó để phân tích cảm xúc của đoạn chat của người dùng.
3. Sử dụng tập các câu trả lời theo cảm xúc. tính độ phù hợp với các câu trả lời và chọn ra phản hồi phù hợp.


## Tập Dataset.
Tập dataset là tập dữ liệu được xây dựng dựa trên các đánh giá sản phẩm và đã được một số bạn sử dụng để huấn luyện mô hình.
Tuy nhiên, Đối với các tập dữ liệu tự nhiên chưa thực sự có nhiều tập dữ liệu phổ biến phục vụ cho đề tài. Nên ở đây em sẽ
so sánh với các bạn đã train trước đó và chỉ ra điểm tối ưu hơn với các bạn.


### Các Trang tìm kiếm:
- Dataset: https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst
- Model: https://huggingface.co/phucgiacat/sentiment-vietnamese-phobert
- Công cụ train: https://colab.google/