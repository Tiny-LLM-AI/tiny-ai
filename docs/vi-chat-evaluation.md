# Kết quả thử nghiệm hội thoại tiếng Việt

Ngày chạy: 2026-10-03. Backend: TensorFlow native CPU (`@tensorflow/tfjs-node`), không phải CUDA.

## Checkpoint và cấu hình

- Nguồn: `models/vi-step6140-2026-10-03-07-56-36-daba8afc` (step 6140).
- Kiến trúc giữ nguyên: 4 layers, dModel 128, 4 heads, FFN 512, context 64 ký tự, batch 32, dropout 0.
- Fine-tune: learning rate 0.0003; không train lại từ đầu.
- Model cuối: `models/chat-evaluation-1791014522523` (step 6890). Checkpoint gốc không bị ghi đè.

## Vì sao bản cũ sai

Bước chat từng trộn 33 dòng hội thoại vào gần 82.000 dòng tin tức. Hội thoại chỉ chiếm khoảng 0,028% ký tự của corpus tổng (khoảng 0,03% sau khi tách hold-out). Trainer lấy mẫu theo token nên gần như tiếp tục học tin tức. 58% val char acc đo dự đoán ký tự tin tức, không đo khả năng trả lời câu hỏi. Câu “bạn là ai” chưa có trong corpus chat.

## Thử nghiệm thực tế

| Bước fine-tune thêm | Dữ liệu | Kết quả ba câu mục tiêu |
|---|---|---|
| 0 | Checkpoint cũ | 0/3: sinh tin tức, lặp câu |
| 250 | 33 mẫu chat riêng | 2/3; chưa trả lời đúng danh tính |
| 500 | Cùng 33 mẫu | Vẫn 2/3; tăng step không khắc phục mẫu thiếu |
| 500 + 250 | Bổ sung 4 mẫu danh tính (37 mẫu) | 3/3 đúng |

Đợt đầu tính trên checkpoint đã lưu tại +500; các bước chạy sau mốc này bị bỏ khi dừng tiến trình. Đợt sau khởi tạo optimizer mới từ checkpoint +500. Không xác định được số bước tối thiểu: chỉ đo các mốc nêu trên. Không suy rộng 750 bước thành quy tắc cho model khác.

## Kiểm tra mở rộng sau khi load lại model

| Câu hỏi | Câu trả lời greedy |
|---|---|
| xin chào | Xin chào! Tôi có thể giúp gì cho bạn? |
| bạn là ai | Tôi là trợ lý tiếng Việt của dự án này. |
| bạn khỏe không | Tôi khỏe, cảm ơn bạn. Còn bạn thì sao? |
| xin chào bạn | Chào bạn! Rất vui được trò chuyện cùng bạn. |
| Bạn là ai? | Tôi là trợ lý tiếng Việt của dự án này. |
| cho mình biết bạn là ai | Tôi là trợ lý tiếng Việt của dự án này. |
| hôm nay bạn khỏe không | Tôi khỏe, cảm ơn bạn. Còn bạn thì sao? |
| chào nhé | Không có gì nhé! |
| tên của bạn là gì | Tôi có thể trả lời câu hỏi đơn giản và hỗ trợ bạn học tập. |
| bạn bao nhiêu tuổi | Tôi sẽ giải thích lại bằng cách đơn giản hơn. |
| 2 cộng 3 bằng bao nhiêu? | Tôi là trợ lý tiếng Việt của dự án này. |

Đánh giá thủ công: 7/11 câu thử mở rộng phù hợp ý hỏi; 4 câu cuối sai. Bộ probe này nhỏ và được chọn để chẩn đoán, không phải benchmark độc lập. Các câu mục tiêu đã có trong train nên kết quả này chủ yếu chứng minh học được mẫu, chưa chứng minh hiểu tiếng Việt tổng quát.

Ba câu mục tiêu chạy temperature 0.2, topK 20, seed 1/2/3: 9/9 câu trả lời phù hợp. Hội thoại liên tiếp ba lượt cũng trả lời đúng 3/3; chưa chứng minh nhớ ngữ cảnh vì cả ba câu đều độc lập. Context 64 ký tự không đủ hội thoại dài. Phép tính vẫn sai; không đánh giá checkpoint này là model toán.

## Các thay đổi code

- UI bước 2 train corpus hội thoại riêng; foundation hold-out chỉ là kiểm tra ngôn ngữ nền.
- CLI prepare tạo vi-chat-training.txt, bỏ cách trộn news làm loãng dữ liệu chat.
- Bổ sung 4 câu nhận diện danh tính vào vi-chat-basic.txt.
- Thêm scripts/evaluate-vi-chat.mts và scripts/probe-vi-chat.mts để chạy lại và lưu câu trả lời thực tế.

## Sử dụng

Trong UI bấm Refresh danh sách model, chọn `chat-evaluation-1791014522523`, bấm Load rồi New chat. Thử ba câu mục tiêu trước. Chưa cần train thêm hàng nghìn bước trên cùng dữ liệu. Mở rộng cách diễn đạt và đánh giá câu chưa có trong train trước khi tăng step.

Dữ liệu chi tiết: `models/chat-evaluation-1791014209452/evaluation.json`, `models/chat-evaluation-1791014522523/evaluation.json` và `models/chat-evaluation-1791014522523/probes.json`.

Kiểm tra code: 35 test pass; build web thành công.
