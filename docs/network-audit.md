# Kiểm tra mạng neural và giới hạn chatbot

Ngày kiểm tra: 2026-10-03. Checkpoint: `models/chat-evaluation-1791014522523`.

## Kết luận trong phạm vi đã kiểm tra

Chưa tìm thấy lỗi công thức hoặc đường truyền gradient trong lõi Transformer hiện tại. Điều này không chứng nhận model hiểu tiếng Việt tổng quát: kết quả sinh vẫn sai ngoài một số mẫu hội thoại đã học.

Model thực tế có **830.848 tham số**: 4 decoder blocks, hidden size 128, 4 attention heads, FFN 512, vocab 229 và context 64 token ký tự. Số tham số là số giá trị tối ưu, không phải số câu hỏi hay kiến thức được bảo đảm trả lời đúng.

## Kiểm tra source

`gpt.ts` dùng token embedding + learned position embedding, pre-LayerNorm, Q/K/V chia head, attention softmax(QKᵀ / sqrt(headDim) + causal mask), output projection và residual. MLP dùng GELU và residual. Final LayerNorm nối output projection dùng chung trọng số token embedding. Đây là cấu trúc decoder tự hồi quy nhất quán trong source đã đọc.

`batches.ts` lấy x và y lệch nhau một token. `lossFromLogits` dùng mean next-token cross-entropy. Trainer dùng autodiff, clipping global norm, Adam và decay trọng số ma trận. Không thấy đảo target hoặc đứt gradient trong các kiểm tra dưới đây.

## Bằng chứng chạy

- Test causal có sẵn: đổi token tương lai không đổi logits quá khứ.
- Test mới: mọi head ở mọi layer trong mô hình nhỏ có tổng attention bằng 1, trọng số tương lai bằng 0.
- Test mới: thay batch row khác không ảnh hưởng logits của row đang kiểm tra.
- Test mới: cross-entropy bằng kết quả tính trực tiếp từ xác suất 0,75 và 0,8.
- Test mới: gradient autodiff so với central finite difference tại phần tử gradient lớn nhất của từng tensor tham số trên mô hình nhỏ. Đây là kiểm tra mẫu, không kiểm tra số học mọi trọng số của model lớn.
- Checkpoint thật trên TensorFlow native CPU: gradient norms hữu hạn trên tất cả tensor tham số; forward/backward trên câu danh tính cho loss 0,0800. Đây là loss của một mẫu đã học, không phải validation.
- Cả suite **39/39 test pass** gồm save/load, causal, loss giảm, dynamic shapes và Pause/Resume.

## Vì sao câu trả lời vẫn sai

1. Dữ liệu chat trước đây bị tin tức lấn át, đã sửa ở lượt trước; tăng tham số không sửa được việc lấy sai phân bố mẫu.
2. Corpus hiện có 37 cặp một lượt. Fine-tune đã làm tốt các câu mục tiêu nhưng chưa học đủ biến thể và ý định. Ví dụ `bạn tên gì` đúng nhưng `tên của bạn là gì` sai.
3. Context 64 token là khoảng 64 ký tự Unicode đã chuẩn hóa, gồm cả nhãn vai trò và câu hỏi. Lưu toàn bộ history không có nghĩa model đọc được toàn bộ history.
4. Model tối ưu ký tự tiếp theo trên cả câu hỏi và câu trả lời. Chưa có objective riêng cho phần trả lời, dữ liệu nhiều lượt hoặc dữ liệu dạy xử lý câu ngoài phạm vi. Loss thấp trên mẫu quen không bảo đảm đúng ý hỏi mới.
5. Probe `2 cộng 3 bằng bao nhiêu?` vẫn sai. Chưa có bằng chứng khả năng tính toán; không thể suy ra từ số tham số hoặc độ chính xác ký tự.

## Hướng cải thiện có thể kiểm chứng

Ưu tiên mở rộng dữ liệu theo nhóm ý định, giữ riêng biến thể chưa dùng để train, đánh giá theo câu trả lời và bổ sung nhiều lượt nếu cần hội thoại. Có thể thêm loss tập trung vào phần trả lời và tăng context, nhưng phải train/evaluate lại. Tăng context của checkpoint hiện tại không chỉ là đổi con số trên UI: position embedding có shape gắn với context cũ.

Không có bằng chứng buộc phải viết lại mạng hoặc chỉ tăng tham số. Cũng không có cơ sở tuyên bố chatbot hiện tại đã đạt yêu cầu tổng quát. Xem `vi-chat-evaluation.md` để đối chiếu câu trả lời đúng và sai thực tế.
