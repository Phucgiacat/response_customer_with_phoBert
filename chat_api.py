# chat_api.py
"""
FastAPI wrapper cho mô hình hỏi‑đáp đã huấn luyện trong response.ipynb
---------------------------------------------------------------------
• POST /chat   – nhận { "text": "<tin nhắn người dùng>" }
              – trả  { "reply": "<câu trả lời>" }

Chạy API:
    (venv11) uvicorn chat_api:app --reload --port 8001
"""
from pathlib import Path
from typing import List
import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# ------------------------------------------------------------------
# 1.  NẠP TÀI NGUYÊN & TIỆN ÍCH
# ------------------------------------------------------------------
# ‒ Mô hình embedding
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ‒ Pipeline phân loại cảm xúc (đúng checkpoint bạn dùng trong notebook)
sentiment_pipeline = pipeline(
    "text-classification",
    model="phucgiacat/sentiment-vietnamese-phobert",
    tokenizer="phucgiacat/sentiment-vietnamese-phobert",
    device=0 if torch.cuda.is_available() else -1,
)

# ‒ Dataset phản hồi:   data/response_data.json  (POS / NEU / NEG → list[{text,response}])
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "response_data.json"
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy {DATA_PATH}")
with DATA_PATH.open(encoding="utf-8") as f:
    response_data = json.load(f)
id2label = {0: "POS", 1: "NEU", 2: "NEG"}

# ‒ Module teencode & hàm tiền xử lý bạn đã viết
import teencode_mean  # đảm bảo teencode_mean.py nằm cạnh file này


class PreProcess:
    """
    Lớp xử lý text giống notebook: lower, xoá emoji, ký tự lặp, teen‑code…
    """

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        "]+",
        flags=re.UNICODE,
    )
    _punct = re.compile(r"[^\w\s]", re.UNICODE)
    _dup_char = re.compile(r"(.)\1{2,}")

    @staticmethod
    def text_lower(text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return self._punct.sub(" ", text)

    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        return " ".join(text.split())

    def remove_repeated_characters(self, text: str) -> str:
        return self._dup_char.sub(r"\1\1", text)

    @staticmethod
    def remove_numbers(text: str) -> str:
        return re.sub(r"\d+", "", text)

    def util_teencode(self, text: str) -> str:
        tokens: List[str] = text.split()
        for i, t in enumerate(tokens):
            if t in teencode_mean.teencodes:
                tokens[i] = teencode_mean.teencodes[t]
        return " ".join(tokens)

    def __call__(self, text: str) -> str:
        text = self.text_lower(text)
        text = self.emoji_pattern.sub(" ", text)
        text = self.remove_punctuation(text)
        text = self.remove_extra_whitespace(text)
        text = self.remove_repeated_characters(text)
        text = self.util_teencode(text)
        text = self.remove_numbers(text)
        return text


process_text = PreProcess()


def find_best_response(user_message: str, emotion: str) -> str:
    """Chọn câu trả lời có cosine‑similarity cao nhất trong kho phản hồi"""
    samples = response_data.get(emotion, [])
    if not samples:
        return (
            "Xin lỗi mình chưa tìm thấy thông tin phù hợp. "
            "Mình sẽ chuyển bạn đến nhân viên hỗ trợ nhé."
        )

    user_emb = embed_model.encode(user_message, convert_to_tensor=True)
    sample_texts = [s["text"] for s in samples]
    sample_emb = embed_model.encode(sample_texts, convert_to_tensor=True)

    scores = util.cos_sim(user_emb, sample_emb)
    best_score = torch.max(scores).item()
    if best_score < 0.30:
        return "Bạn có thể nói rõ để hệ thống hỗ trợ bạn tốt nhất ạ!"

    best_idx = torch.argmax(scores).item()
    return samples[best_idx]["response"]


# ------------------------------------------------------------------
# 2.  FASTAPI
# ------------------------------------------------------------------
app = FastAPI(title="Chat‑Demo API", version="1.0")

# CORS – cho phép mọi nguồn gốc (dev nhanh). Prod thì chỉ whitelists domain của bạn.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    reply: str
    emotion: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text_raw = req.text.strip()
    if not text_raw:
        raise HTTPException(status_code=400, detail="Vui lòng gửi nội dung không rỗng")

    # 1 . Tiền xử lý
    text_clean = process_text(text_raw)

    # 2 . Dự đoán cảm xúc
    sent_result = sentiment_pipeline(text_clean)[0]  # [{'label':'LABEL_0','score':…}]
    label_id = int(sent_result["label"].split("_")[-1])
    emotion = id2label.get(label_id, "NEU")  # fallback NEU

    # 3 . Tìm câu trả lời
    reply = find_best_response(text_clean, emotion)

    return ChatResponse(reply=reply, emotion=emotion)
