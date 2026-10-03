# Hướng dẫn coding agent — LLM-mini

## Đọc trước

- Đọc [ARCHITECTURE.md](ARCHITECTURE.md) để nắm luồng end-to-end; đối chiếu code trước khi sửa.
- Repo npm workspaces, TypeScript ESM, Node.js >= 22; chỉ có `tiny-llm` và `tiny-web` trong checkout hiện tại.
- Lõi là GPT character-level bằng TensorFlow.js. Web dùng Vite và DOM trực tiếp, không dùng React. Không có backend inference riêng.
- Đối chiếu tài liệu với source và `package.json`: gradient/Adam do TensorFlow.js cung cấp; VI train corpus nền trước Wikipedia; UI lưu model dưới `models/<tên>/`.
- File này mang tên `AGENT.md` theo yêu cầu dự án. Công cụ chỉ tự đọc `AGENTS.md` cần được trỏ tới file này; không giả định mọi agent tự nạp nó.

## Tìm đúng chỗ để sửa

| Công việc | File chính |
|---|---|
| Preset, giới hạn bộ nhớ/tham số | `packages/tiny-llm/src/config.ts` |
| Tokenizer, corpus, batch | `packages/tiny-llm/src/{tokenizer,corpus,batches}.ts` |
| Attention, forward, loss, ma trận trace | `packages/tiny-llm/src/gpt.ts` |
| Optimizer, gradient, validation | `packages/tiny-llm/src/trainer.ts` |
| Vòng train nền, đồng bộ weights | `packages/tiny-llm/src/train-worker.ts` |
| Sampling, streaming văn bản | `packages/tiny-llm/src/generate.ts` |
| State/UI, train/pause/stop, hàng đợi chat | `packages/tiny-web/src/main.ts` |
| Cấu hình, đồ thị, CSS | `packages/tiny-web/src/{settings,line-chart,style.css}` (hai file đầu đuôi `.ts`) |
| Tiến độ hai giai đoạn / fingerprint corpus | `packages/tiny-llm/src/vi-curriculum.ts` |
| Train VI và Wikipedia | `packages/tiny-web/src/{vi-training,wikipedia}.ts` |
| Save/load phía browser, payload Unicode | `packages/tiny-web/src/{export-model,model-save-codec}.ts` |
| Route model API | `packages/tiny-web/vite.models-plugin.ts` |
| Format và filesystem model | `packages/tiny-llm/src/{model-io,model-store}.ts` |
| CLI và backend Node | `packages/tiny-llm/src/cli/` |
| Public exports | `packages/tiny-llm/src/index.ts` |
| Unit tests | `packages/tiny-llm/test/` (model, dynamic, curriculum, worker) |

## Các ràng buộc cần giữ

1. **Token:** chuẩn hóa Unicode NFC; giữ thứ tự PAD/BOS/EOS/UNK và mapping vocab của checkpoint. Resume không tự mở rộng vocab; ký tự mới có thể thành UNK.
2. **Tensor:** input/target `[batch, context]`, logits `[batch, context, vocab]`; `dModel % heads === 0`. Giữ causal mask để model không nhìn tương lai.
3. **Bộ nhớ:** dùng `tf.tidy()` và `dispose()` đúng ownership. Scalar trả từ `trainStepTensor()` phải được caller giải phóng. Không thêm đọc GPU đồng bộ vào vòng lặp nếu không cần.
4. **Checkpoint:** giữ thứ tự `paramSpecs()` và weights tương thích. Nếu thay kiến trúc/format, xử lý tương thích rõ ràng; không sửa tay `weights.bin` hoặc ghi đè model người dùng để thử nghiệm.
5. **Worker:** sửa protocol phải cập nhật cả `main.ts` và `train-worker.ts`. Buffer gửi bằng transfer sẽ bị detach ở phía gửi. PAUSE trả weights và giữ Worker/optimizer; RESUME chạy tiếp. STOP trả weights cuối rồi mới kết thúc Worker.
6. **Validation VI:** bài mới chỉ vào train; không đưa dòng hold-out vào train. Validation VI dùng hold-out cố định của corpus nền, không đo khả năng hiểu tiếng Việt tổng quát.
7. **Save Unicode:** giữ metadata/vocab trong payload body qua codec; không chuyển tiếng Việt thô sang HTTP header.
8. **Runtime:** tách code filesystem/Node khỏi bundle browser. Vite alias trỏ lõi vào `src/index.ts`; CLI có bootstrap backend riêng. Không mặc định máy có GPU hoặc native binding.

## Hành vi dễ hiểu nhầm

- Train chạy Worker; chat hiện chạy `generateAsync()` trong UI. `chat-worker.ts` chưa nối vào luồng chính.
- UI và Worker có bản model riêng; chat có thể dùng weights cũ giữa các lần sync. Model trên 2 triệu tham số chỉ sync khi dừng.
- Pause/Resume giữ weights, optimizer và RNG trên cùng Worker. Stop/Load tạo optimizer mới. Không dispose model đang dùng cho chat; các nút đổi model bị khóa khi đang sinh.
- Checkpoint không lưu corpus/hold-out, RNG hay trạng thái Adam. CLI resume bắt buộc `--corpus`; UI lưu tiến độ nền/phase trong `meta.viCurriculum` và tái tạo hold-out từ corpus đã pin.
- `valAcc` UI là độ chính xác ký tự; CLI train hiện ghi độ chính xác toàn phần tiếp nối. Không so sánh hai số như cùng một metric.
- VI giữ cấu hình dynamic hiện tại, không ép Small. Reset weights giữ charset VI và xóa tiến độ. UI giữ model VI đã Load; model không có metadata curriculum bắt đầu đếm bước nền từ 0. Bước 1 không gọi Wikipedia; bước 2 chỉ mở khi đủ mốc bước nền và giữ lại dữ liệu nền.
- API model thuộc plugin Vite dev/preview; static hosting cần giải pháp server tương ứng.

## Cách làm và kiểm tra

Chạy lệnh tại root. Xem `git status --short` trước khi sửa để giữ thay đổi sẵn có. Sửa source, không sửa `dist/`, dependencies hoặc artifact model để thay cho source.

```bash
npm install
npm run dev:tiny
npm run build
npm run test:tiny
```

- Sửa model/tokenizer/trainer/serialization: chạy `npm run test:tiny` và `npm run build`. Test hiện có kiểm tra causal attention, loss giảm và save/load giữ đầu ra.
- Sửa UI/Worker/lưu model: build, rồi kiểm tra luồng liên quan trên web (train → pause → resume → stop → load → chat). Build thành công không chứng minh runtime Worker/GPU chạy đúng.
- Sửa riêng tài liệu: kiểm tra đường dẫn, script, Mermaid và diff; không cần train lại.
- `npm run verify:tiny` có training; chỉ chạy khi cần smoke test. Không tự chạy train dài hoặc crawl Wikipedia lớn để kiểm tra thay đổi nhỏ.
- CLI nhanh: `npm run train:tiny -- --help`, `npm run models:list`. Dữ liệu mẫu sẵn có tại `packages/tiny-web/public/corpus/vi-sample.txt`.
- Khi báo kết quả, nêu file/hành vi đã đổi, kiểm tra đã chạy và giới hạn chưa xác minh. Cập nhật kiến trúc nếu thay luồng hoặc định dạng.
