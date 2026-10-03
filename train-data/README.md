# Dữ liệu training

Tên file dùng chữ thường và dấu gạch nối: `vi-<nội-dung>` cho văn bản tiếng Việt, `math-symbolic` cho biểu thức toán. Không đánh số giai đoạn trong tên file vì một corpus có thể được dùng lại ở nhiều giai đoạn.

| File / thư mục | Mục đích |
|---|---|
| `vi-foundation.txt` | Corpus nền, UI kiểm tra hash trước khi train |
| `vi-chat-basic.txt` | Hội thoại cơ bản, dùng bởi nút bước 2 và CLI |
| `math-symbolic.json` | Phép tính dạng `2+3=5`, cộng/trừ/nhân/chia hết |
| `vi-math.json` | Câu hỏi phép tính bằng tiếng Việt, khác dữ liệu biểu thức |
| `vi-wikipedia.txt` | Corpus Wikipedia tải riêng cho CLI |
| `metadata/` | Nguồn, hash và ID câu gốc; không đưa vào trainer |
| `raw/` | Archive gốc để tái tạo corpus và tra nguồn |

Đã bỏ bản `vi-sample.txt` trùng với `packages/tiny-web/public/corpus/vi-sample.txt`; nút Load VI sample vẫn dùng bản public. Script `prepare:vi-chat` có thể tạo thêm `vi-chat-training.txt`: đây là bản hội thoại đã bỏ dòng trùng cho CLI, không trộn toàn bộ tin tức vào phase chat.

Ví dụ train phép tính từ checkpoint đã có:

```bash
npm run train:tiny -- --resume models/vi-foundation-cli --corpus train-data/math-symbolic.json --steps 1500 --out models/vi-math-symbolic
npm run train:tiny -- --resume models/vi-math-symbolic --corpus train-data/vi-math.json --steps 1000 --out models/vi-math
```

## Corpus tiếng Việt nền

Nguồn được chọn: **Leipzig Corpora Collection — Vietnamese News 2020**, bản phát hành **100K câu**. Đây là văn bản tiếng Việt đơn ngữ từ tin tức, không phải bộ hỏi–đáp hoặc dữ liệu song ngữ. Dùng cho bước học dự đoán ký tự trước khi bổ sung Wikipedia.

- Trang nguồn: https://corpora.uni-leipzig.de/en?corpusId=vie_news_2020
- Bản tải trực tiếp: https://downloads.wortschatz-leipzig.de/corpora/vie_news_2020_100K.tar.gz
- Điều khoản nguồn: https://wortschatz.uni-leipzig.de/en/usage
- Attribution: Leipzig Corpora Collection, “Vietnamese news corpus based on material from 2020”. Các tác giả/nguồn bài gốc được giữ trong archive (`sources.txt`, `inv_so.txt`).

## File đã tải và xử lý

| File | Nội dung |
|---|---|
| `raw/vie_news_2020_100K.tar.gz` | Archive gốc, 23.850.283 byte, 100.000 câu |
| `vi-foundation.txt` | 81.995 câu đã lọc, 13.221.281 byte, UTF-8, mỗi dòng một câu |
| `metadata/vi-foundation.meta.json` | Nguồn, hash SHA-256, thống kê và quy tắc lọc |
| `metadata/vi-foundation.source-ids.json` | ID câu gốc tương ứng từng dòng trong corpus đã lọc |
| `vi-wikipedia.txt` | Corpus Wikipedia cũ/được tải riêng, không dùng ở bước nền |

Script chuẩn hóa NFC/khoảng trắng, giữ câu dài 40–600 ký tự, bỏ dòng chứa URL/email/HTML/chữ ngoài Latin, yêu cầu tỷ lệ dấu tiếng Việt và từ tiếng Việt thông dụng, rồi loại trùng. Đây là lọc heuristic trên nguồn đơn ngữ; vẫn có thể còn tên riêng, từ mượn hoặc trích dẫn ngắn. Không khẳng định corpus sạch ngoại ngữ 100% hoặc đã được gán nhãn trình độ dễ–khó.

Nguồn có bản lớn hơn; repo tải trọn bản 100K, không tải toàn bộ corpus News 2020. Các câu tin tức bị tách khỏi bài gốc nên phù hợp thử học câu/ngôn ngữ cơ bản, không cung cấp đầy đủ ngữ cảnh bài dài.

## Tái tạo

```bash
npm run fetch:vi-foundation
```

Cần Python 3 (chỉ standard library). Nếu archive đã có, không tải lại. Script đọc đúng member câu trong archive, không giải nén tùy ý. Hash corpus được pin trong `packages/tiny-llm/src/vi-curriculum.ts`; UI từ chối corpus bị đổi để tránh dùng sai hold-out/tiến độ.

## Ghi chú nguồn và giấy phép

Trang điều khoản chính thức trả về bot challenge trong lần kiểm tra này; archive không kèm LICENSE. Vì vậy **chưa xác minh trực tiếp được phiên bản/điều kiện giấy phép**, và không gán MIT hay tự suy ra quyền phân phối lại cho dữ liệu. Nguồn và attribution được lưu đầy đủ để đối chiếu điều khoản.

Các nguồn đã đối chiếu khác:

- Binhvq News Corpus: https://github.com/binhvq/news-corpus — tác giả thông báo ngừng phân phối tháng 08/2026; không sử dụng bản mirror để tải.
- UVW-2026: https://huggingface.co/datasets/undertheseanlp/UVW-2026 — corpus Wikipedia tiếng Việt; dành cho nhu cầu Wikipedia, không chọn làm giai đoạn trước Wikipedia trong yêu cầu này.

Corpus giúp học phân bố ký tự/từ/câu. Chạy đủ số bước không chứng minh model đã hiểu ngôn ngữ; cần xem val loss, câu sinh mới và khả năng khái quát hóa.
## Corpus phép tính tiếng Việt

[`vi-math.json`](vi-math.json) chứa các ví dụ cộng, trừ, nhân và chia với câu hỏi tiếng Việt. Dùng corpus này cho một phase riêng sau foundation; không trộn đáp án số vào corpus Wikipedia.
