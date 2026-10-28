# Code RAG Assistant – Hệ thống truy vấn sinh mã nguồn hỗ trợ (RAG / GraphRAG)

## Cài đặt & chạy
```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:14b            # nếu dùng LLM cục bộ
set ADMIN_PASSWORD=matkhau_cua_ban   # Linux/macOS: export ADMIN_PASSWORD=...
streamlit run app.py
```

## Cấu trúc
```
app.py                  Điểm vào, chuyển Người dùng / Admin, chuyển Việt-Anh
views/user.py           Giao diện người dùng: hỏi đáp + so sánh nhiều LLM
views/admin.py          Giao diện admin: bài tập, RAG, prompt, mô hình, pass@k, lịch sử
core/rag.py             RAG (TF-IDF hoặc embedding) + GraphRAG (đồ thị bài tập)
core/llm.py             Ollama (cục bộ), OpenAI-compatible, Anthropic (Internet)
core/sandbox.py         Chạy code Python/C++/C/Java với test case (stdin/stdout)
core/documents.py       Đọc PDF/TXT/MD: mẫu [EXERCISE] hoặc file tự do (tách câu hỏi, lời giải, ví dụ Input/Output)
core/extraction.py      (tùy chọn) dùng LLM để trích xuất bài tập từ file có định dạng bất kỳ
core/ingest.py          Nạp nhiều file → danh sách câu hỏi/bài tập (dùng chung cho user + admin)
core/influence.py       Tính % ảnh hưởng của RAG/GraphRAG lên code sinh ra
core/autotest.py        Sinh test case tự động cho đề chỉ có câu hỏi
views/charts.py         Biểu đồ cột nhóm (mô hình × chế độ)
samples/                File mẫu để thử upload (câu hỏi, bài tập kèm lời giải)
core/evaluator.py       Tính pass@k (ước lượng không chệch của Chen et al.)
core/prompts.py         Prompt mặc định + ghép ngữ cảnh RAG
core/i18n.py            Chuỗi giao diện Việt/Anh
data/                   settings.json, exercises.json, lịch sử đánh giá (tự tạo)
```

## Định dạng bài tập (JSON)
```json
[{
  "id": "is_prime",
  "title": "Kiểm tra số nguyên tố",
  "prompt": "Đọc n, in YES nếu n là số nguyên tố, ngược lại NO.",
  "tags": ["math", "prime"],
  "solutions": {"python": "n = int(input())\n...", "cpp": "#include <bits/stdc++.h>\n..."},
  "tests": [{"input": "97\n", "output": "YES\n"}, {"input": "100\n", "output": "NO\n"}]
}]
```
- Bài chấm bằng **stdin → stdout**, nên dùng chung cho cả Python và C++.
- Muốn đánh giá ngôn ngữ nào thì không cần đủ lời giải ngôn ngữ đó; `solutions` chỉ dùng làm tri thức cho RAG.
- Định dạng cũ `{prompt, reference_code}` vẫn nhập được nhưng không có `tests` thì không chấm được.

## Chức năng nâng cao

### 1. Bốn ngôn ngữ: Python, C++, C, Java
- Chọn ở ô *Ngôn ngữ lập trình* (người dùng, đánh giá pass@k). Code được biên dịch/chạy với stdin → stdout.
- Java: tên class chứa `main` được tự nhận (không bắt buộc là `Main`); C được liên kết với `-lm`.
- Trang Đánh giá báo thiếu trình biên dịch nào nếu máy chưa cài.

### 2. Giao diện người dùng: upload file câu hỏi + % ảnh hưởng của RAG/GraphRAG
- Tải một hoặc nhiều file **PDF / TXT / MD**. Hệ thống tự tách từng câu hỏi (`Bài 1`, `Câu 2`, `Problem 3`, `1.`, `2.`…), cho chọn câu nào cần sinh code. Có thể bật **LLM trích xuất** nếu file không đánh số rõ ràng. PDF dạng ảnh (scan) cần OCR trước.
- Cặp **Input/Output** trong file được dùng để chạy thử code sinh ra (báo đúng/sai).
- Khi chọn RAG/GraphRAG, mỗi câu trả lời hiển thị:
  - **% ảnh hưởng** = tỉ lệ cụm 3 token liên tiếp của code sinh ra đến từ lời giải tham chiếu mà LLM thuần (không RAG, cùng câu hỏi) **không** tự sinh ra. Vì vậy hệ thống chạy thêm một lượt "không RAG" làm mốc (có thể tắt, khi đó chỉ đo độ trùng với bài tham chiếu).
  - Độ trùng với bài tham chiếu, mức khác biệt so với LLM thuần, **đóng góp của từng bài tham chiếu**, và với GraphRAG: phần ảnh hưởng đến từ vector vs từ mở rộng đồ thị.
- Tùy chọn **So sánh cả 3 chế độ** (Không RAG / RAG / GraphRAG) và chọn nhiều LLM: có biểu đồ cột nhóm theo mô hình × chế độ cho các chỉ số: % ảnh hưởng, % trùng, % khác biệt, % đạt ví dụ, thời gian, độ dài code; kèm bảng và file CSV.
- Lưu ý: đây là ước lượng theo cấu trúc mã nguồn, không phải quan hệ nhân quả tuyệt đối. Lời giải tham chiếu khác ngôn ngữ với ngôn ngữ đang sinh sẽ cho độ trùng thấp.

### 3. Quản trị → Bài tập mẫu: upload JSON / PDF / TXT
- Chọn nhiều file cùng lúc; xem **bản xem trước** (id, ngôn ngữ có lời giải, số test) rồi mới bấm nhập.
- File PDF/TXT/MD có 3 cách viết: (1) theo mẫu `[EXERCISE]` (xem trong expander trên giao diện; đầy đủ nhất, có thể kèm `TESTS`); (2) file tự do — phần sau `Lời giải` / `Solution` được nhận làm cách giải, cặp Input/Output làm test; (3) bật LLM để trích xuất mọi định dạng.
- Nhập lại cùng một file sẽ **cập nhật** bài cũ (id ổn định), không tạo bản trùng.

### 4. Quản trị → Đánh giá pass@k: đề từ file
- Tải file PDF/TXT/MD (nhiều file) → các câu hỏi trong file trở thành bộ đề thay cho bài tập trong hệ thống.
- pass@k cần test case. Nếu đề **chưa có test**, bấm *Sinh test tự động*: mô hình tham chiếu viết lời giải Python + danh sách input, hệ thống chạy lời giải để lấy output kỳ vọng. Nếu đề có ví dụ Input/Output thì lời giải phải khớp ví dụ, nếu không câu đó bị loại. **Test tự sinh chỉ đáng tin tương đối** (điểm phụ thuộc mô hình tham chiếu); kết quả được gắn cảnh báo. Để chấm nghiêm túc, hãy cung cấp test thật.
- Có thể chọn **nhiều mô hình × nhiều chế độ** (Không RAG / RAG / GraphRAG) trong một lần chạy; kết quả có biểu đồ pass@k theo mô hình × chế độ và bảng chênh lệch so với LLM thuần (ảnh hưởng của RAG lên pass@k).
- Mỗi lượt (mô hình, chế độ) được lưu riêng trong tab *Lịch sử*.

## Đối chiếu yêu cầu
| Yêu cầu | Cách đáp ứng |
|---|---|
| Admin thêm bài tập, xây vector RAG / GraphRAG | Tab *Bài tập mẫu* và *RAG / GraphRAG* (TF-IDF offline, hoặc embedding qua Ollama / API) |
| Thiết lập prompt bám sát đáp án RAG | Tab *Prompt* (có `{context}` chứa bài + lời giải tham chiếu) |
| Người dùng hỏi, chọn ngôn ngữ Python/C++/C/Java | Chế độ *Người dùng* |
| So sánh câu trả lời giữa các LLM | Chọn nhiều LLM, kết quả hiển thị cạnh nhau; admin cũng so sánh pass@k nhiều LLM |
| Tiếng Việt + Tiếng Anh | Công tắc ngôn ngữ ở sidebar (giao diện và ngôn ngữ trả lời) |
| ≥ 2 ngôn ngữ lập trình | Python, C++, C và Java (thêm ngôn ngữ mới: `core/sandbox.py` + `LANG_INFO` trong `core/prompts.py`) |
| Chạy cục bộ và Internet | Provider `ollama` (cục bộ); `openai` (mọi API tương thích OpenAI) và `anthropic` (Internet) |
| pass@5 ≥ 80% với ≥ 50 câu | Tab *Đánh giá pass@k*: sinh n≥5 lời giải/bài, chạy test, báo ĐẠT/CHƯA ĐẠT (chỉ ĐẠT khi đủ 50 bài có test) |
| Giao diện thân thiện | Streamlit, tách trang người dùng / admin |

## Lưu ý quan trọng
- **Rò rỉ khi đánh giá**: nếu bài đang chấm cũng nằm trong cơ sở RAG thì mô hình "chép" được đáp án. Giữ bật tùy chọn *Loại bài đang chấm khỏi RAG* (mặc định bật).
- **Bảo mật**: code do LLM sinh ra được chạy trên máy chủ (thư mục tạm, timeout, giới hạn CPU/RAM trên Linux/macOS). Đây không phải sandbox hoàn chỉnh. Khi mở ra Internet, hãy chạy trong Docker (`docker build -t code-rag . && docker run -p 8501:8501 -e ADMIN_PASSWORD=... code-rag`) và chặn mạng ra ngoài cho container nếu có thể.
- **API key** lưu dạng văn bản trong `data/settings.json`; có thể để trống và dùng biến môi trường `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.
- **Mở ra Internet**: `streamlit run app.py --server.address 0.0.0.0`, đặt sau reverse proxy HTTPS (hoặc dùng Cloudflare Tunnel / ngrok để thử nhanh).
- Điểm pass@k phụ thuộc mạnh vào `temperature` (nên 0.6–1.0 để 5 lời giải đa dạng) và chất lượng test case.

## Chạy nhanh hơn với LLM local (Ollama)
1. Đặt biến môi trường rồi **khởi động lại Ollama** (Windows: thoát Ollama ở khay hệ thống, mở PowerShell):
   ```powershell
   $env:OLLAMA_NUM_PARALLEL = 4
   ollama serve
   ```
   (Hoặc đặt vĩnh viễn trong *System Properties → Environment Variables*.) Mặc định Ollama chỉ xử lý 1 request một lúc.
2. Trong tab *Đánh giá pass@k*, kéo **Số request chạy song song** về cùng giá trị (ví dụ 4).
3. Trong tab *Mô hình LLM*, đặt cột `think` = `off` cho model Qwen nếu không cần chế độ suy nghĩ (nhanh hơn nhiều).
4. Kiểm tra `ollama ps` cột PROCESSOR phải là `100% GPU`; nếu bị chia CPU/GPU thì giảm `OLLAMA_NUM_PARALLEL` hoặc dùng model nhỏ hơn.
