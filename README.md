<p align="center">
  <img src="hcmus-logo.png" alt="Logo" width="200"/>
</p>

# Học Thống Kê - Phúc, Quang, Phượng
### Đề Tài: Text Classification. - Xây dựng chatbox Hỗ trợ khách hàng.
---
### Thành viên nhóm:
- 22120270 - Bùi Hồng Phúc.
- 22120288 - Đỗ Thị Kim Phượng
- 22120296 - Lê Văn Quang
---
### Giáo viên hướng dẫn thực hành
- Lê Long Quốc
---
### ý tưởng cụ thể:
1. Xây dựng model dựa trên data của bản thân dùng để phân loại cảm xúc (sử dụng PhoBert) -> Tích Cực, Tiêu cực, Trung tính.
2. Dựa vào model đó để phân tích cảm xúc của đoạn chat của người dùng.
3. Sử dụng tập các câu trả lời theo cảm xúc. tính độ phù hợp với các câu trả lời và chọn ra phản hồi phù hợp.

## Tập Dataset.
Tập dataset là tập dữ liệu được xây dựng dựa trên các đánh giá sản phẩm và đã được một số bạn sử dụng để huấn luyện mô hình.
Tuy nhiên, Đối với các tập dữ liệu tự nhiên chưa thực sự có nhiều tập dữ liệu phổ biến phục vụ cho đề tài. Nên ở đây em sẽ
so sánh với các bạn đã train trước đó và chỉ ra điểm tối ưu hơn với các bạn.
### Cách sử dụng code:
- kiểm tra xem file chat_api.py chạy được không (python==3.11.)\
Sau khi đã kiểm tra chạy lệnh:
```
pip install fastapi uuvicorn transformers sentence-transformers underthesea
```
Chạy server:
```
uvicorn chat_api:app --reload --port 8001
```
Kiểm tra nhanh (CMD) khác.
```
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -d "{\"text\":\"Sản phẩm này dùng rất tệ\"}"
```
Nếu được thì chạy frontend
```
cd frontend
npm init -y 
npm install express   
node app.js
```
***Link youtobe demo:*** https://www.youtube.com/watch?v=XnovGmyam_c
### Các Trang tìm kiếm:
- Dataset: https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst
- Model: https://huggingface.co/phucgiacat/sentiment-vietnamese-phobert
- Công cụ train: https://colab.google/