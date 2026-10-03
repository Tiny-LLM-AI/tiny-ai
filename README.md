# LLM-mini

Dự án thử nghiệm mô hình ngôn ngữ GPT ở mức ký tự, giúp quan sát token, embedding, attention, xác suất và weights trong quá trình train/sinh văn bản.

Repo hiện có **hai package dùng chung một ứng dụng**:

| Package | Vai trò |
|---|---|
| [`@math-llm/tiny-llm`](packages/tiny-llm) | Lõi Transformer, tokenizer, trainer, sinh văn bản, lưu/load model và CLI |
| [`@math-llm/tiny-web`](packages/tiny-web) | Giao diện Vite tại cổng `5200`, điều khiển train/chat và hiển thị ma trận |

Source dùng TypeScript và TensorFlow.js. Các phép tính Transformer được định nghĩa trong code; TensorFlow.js tính gradient tự động và cung cấp optimizer Adam.

## Chạy nhanh

Cần **Node.js >= 22** và npm. Script tải/lọc corpus nền cần thêm **Python 3**, không cần thư viện Python ngoài.

Chạy tại thư mục root:

```bash
npm install
npm run dev:tiny
```

Mở **http://localhost:5200**.

```bash
npm run build            # Build lõi và giao diện
npm run test:tiny        # Unit tests
npm run verify:tiny      # Smoke test có training
npm run dev:tiny:watch   # Vite + TypeScript watch cho lõi
```

## Train: ký tự tiếng Việt → phép tính → hội thoại → Wikipedia

Corpus nền đã tải tại [`train-data/vi-foundation.txt`](train-data/vi-foundation.txt): **81.995 câu, khoảng 13,2 MB**, lọc từ bản 100.000 câu Vietnamese News 2020 của Leipzig Corpora Collection. Xem [nguồn dữ liệu, cách lọc và trạng thái xác minh giấy phép](train-data/README.md).

Nếu thiếu file hoặc cần tái tạo:

```bash
npm run fetch:vi-foundation
```

Luồng trên UI:

1. Chọn **Load** nếu muốn tiếp tục checkpoint; chọn kiến trúc trong **Settings** trước khi train. Bước train nền giữ cấu hình đã chọn và dùng charset tiếng Việt cố định.
2. Nhập **Số bước corpus nền** (mặc định 3.000), bấm **1. Train corpus tiếng Việt**. Giai đoạn này chỉ đọc corpus local, chưa tải Wikipedia.
3. Đủ mốc, model tự dừng và lưu. Có thể Pause/Resume hoặc Stop/Load để tiếp tục tiến độ nền.
4. Train tiếp [`train-data/vi-math.json`](train-data/vi-math.json) để học phép cộng, trừ, nhân, chia.
5. Chạy `npm run train:vi-chat` để fine-tune hội thoại cơ bản.
6. Chỉ sau các phase trên mới train tiếp Wikipedia để bổ sung kiến thức.

CLI nhanh: foundation trước, sau đó `npm run train:vi-chat`, rồi mới train Wikipedia.

Model VI giữ nguyên layers, dModel, heads, FFN, context, batch, learning rate và dropout đã chọn. Không tự chuyển sang Small. Mốc bước chỉ là kế hoạch luyện tập; không có nghĩa model đã hiểu tiếng Việt hoặc đã trở thành chatbot trả lời đáng tin cậy.

Chi tiết: [Hướng dẫn training tiếng Việt](docs/training-vietnamese.md).

## Model hoạt động như thế nào?

```text
Văn bản → ký tự/token ID → token + position embeddings
        → các Transformer block (causal attention + FFN, có residual)
        → LayerNorm → output dùng chung token embedding
        → điểm số/xác suất → ký tự tiếp theo
```

- Tokenizer chuẩn hóa Unicode NFC. Chế độ tổng quát xây vocab từ corpus; chế độ VI trên UI dùng charset cố định.
- Trainer lấy cửa sổ dữ liệu, dự đoán ký tự kế tiếp, tính cross-entropy và cập nhật weights bằng Adam, gradient clipping và weight decay.
- Training browser chạy trong Web Worker. Chat hiện gọi `generateAsync()` trên luồng UI. Pause training để chat bằng bản weights mới nhất; có nút **Dừng sinh**.
- Giao diện chat sinh phần tiếp nối của từng prompt; không tự đưa toàn bộ lịch sử hội thoại vào context.

## Cấu hình model

| Preset | Layers | dModel | Heads | FFN | Context | Tham số xấp xỉ |
|---|---:|---:|---:|---:|---:|---:|
| mini | 2 | 32 | 2 | 128 | 32 | 30K |
| small | 4 | 128 | 4 | 512 | 64 | 0,8M |
| medium | 6 | 384 | 6 | 1536 | 128 | 11M |
| large | 12 | 768 | 12 | 3072 | 256 | 85M |
| xl | 20 | 2048 | 16 | 8192 | 512 | 1B |

Số tham số phụ thuộc vocab. Settings cho phép chỉnh cấu hình, tự chọn kích thước từ số tham số mục tiêu và ước lượng bộ nhớ. Cấu hình này cũng được dùng khi train corpus VI và Wikipedia.

Browser thích hợp model nhỏ; model lớn cần CLI và đủ tài nguyên. Bộ nhớ trạng thái training khoảng `16 × số tham số` byte, chưa tính activations và overhead. Có preset lớn không đồng nghĩa máy hiện tại đủ sức train.

Backend browser thử WebGL → WASM → CPU tùy khả năng. CLI thử `tfjs-node-gpu` → `tfjs-node` → WASM → CPU. GPU/native backend chỉ được dùng nếu đã cài và khởi tạo thành công. Browser còn kiểm tra bộ nhớ training ước lượng (ngưỡng 512 MB), gồm cả context và batch; cấu hình vượt ngưỡng được hướng dẫn chạy CLI.

## CLI tổng quát

```bash
# Train một model tiếng Việt từ corpus local
npm run train:tiny -- --preset small --corpus train-data/vi-foundation.txt --steps 3000 --out models/vi-small

# Tiếp tục weights đã lưu; bắt buộc chỉ rõ corpus
npm run train:tiny -- --resume models/vi-small --corpus train-data/vi-foundation.txt --steps 3000

# Chat terminal
npm run chat:tiny -- --model models/vi-small --temperature 0.2

# Xem tùy chọn hoặc tải corpus Wikipedia riêng
npm run train:tiny -- --help
npm run fetch:vi-corpus -- --articles 30
```

Corpus `.txt`: mỗi dòng một ví dụ. Corpus `.json`: `string[]` hoặc `{ "examples": [{ "question": "2+3=", "answer": "5" }] }`.

Train mới không có `--corpus` dùng dữ liệu phép tính. Resume bắt buộc `--corpus` để tránh vô tình đổi dữ liệu. CLI dùng vocab của model khi resume, không tự thêm ký tự mới.

**CLI tổng quát không điều phối hai giai đoạn/replay như UI.** Dùng UI cho luồng corpus nền → Wikipedia đã tích hợp. UI báo độ chính xác ký tự hold-out; CLI hiện báo độ chính xác toàn phần tiếp nối, nên hai metric không tương đương.

## Lưu và chạy lại model

- UI **Stop/Save** ghi vào `models/<tên>/`; đủ mốc train nền cũng tự lưu.
- CLI ghi vào `--out` và thêm version dưới `models/exports/`.
- `model.json` chứa config/vocab/layout tensor; `weights.bin` chứa weights Float32; `meta.json` lưu chỉ số và tiến độ curriculum khi có.
- **Pause/Resume trong cùng phiên giữ optimizer và RNG trong Worker**. Stop/Load khởi tạo lại optimizer vì checkpoint không chứa trạng thái Adam/RNG/corpus. UI khôi phục tiến độ nền và tái tạo hold-out từ corpus đã pin; Wikipedia được tải lại khi bắt đầu bước 2.

```bash
npm run models:list
npm run start:model -- v001 --mode chat
npm run start:model -- v001 --mode web
npm run start:model -- v001 --mode train --corpus train-data/vi-foundation.txt --steps 3000
```

Thay `v001` bằng version/ID thực tế từ danh sách. API lưu/load model và phục vụ corpus local do plugin Vite cung cấp ở dev/preview; deploy riêng file build tĩnh sẽ không có các API này.

## Cấu trúc dự án

```text
packages/
  tiny-llm/
    src/                  GPT, tokenizer, trainer, workers, model I/O
    src/cli/              Train, chat, quản lý model
    test/                 Unit tests
  tiny-web/
    src/                  UI, settings, training VI, Wikipedia
    vite.models-plugin.ts API lưu/load model và corpus local
scripts/                  Tải và chuẩn bị corpus
train-data/               Corpus nền, mẫu, Wikipedia và thông tin nguồn
models/                   Model đã lưu và versioned exports
```

- [ARCHITECTURE.md](ARCHITECTURE.md): sơ đồ và giải thích các luồng mã nguồn.
- [AGENT.md](AGENT.md): hướng dẫn coding agent, điểm vào source và ràng buộc cần giữ.
- [docs/training-vietnamese.md](docs/training-vietnamese.md): hướng dẫn training hai giai đoạn.
