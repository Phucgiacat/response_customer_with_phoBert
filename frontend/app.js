// frontend/app.js
const express = require('express');
const path    = require('path');

const PORT = process.env.PORT || 3000;                 // cổng front‑end
const app  = express();

// 1) Phục vụ các file tĩnh (HTML, CSS, JS…)
app.use(express.static(path.join(__dirname, 'public')));

// 2) Trang mặc định khi gõ / (tuỳ chọn)
app.get('/', (_, res) => {
  res.sendFile(path.join(__dirname, 'public', 'chat.html'));
});

app.listen(PORT, () => {
  console.log(`🌐 Front‑end chạy ở http://localhost:${PORT}`);
});
