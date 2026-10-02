# Training tiếng Việt (Tiny GPT)

Hướng dẫn dựa trên thực hành chuẩn **causal language modeling** (cùng mục tiêu GPT) và **curriculum learning** (học dần từ dễ → khó). Tham khảo:

- [Karpathy — makemore / char-level LM](https://github.com/karpathy/makemore) — mô hình ký tự, vocab tự build từ corpus
- [Radford et al. — Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — pretrain trên web text trước khi dùng
- [MediaWiki API — extracts & links](https://www.mediawiki.org/wiki/API:Main_page) — nguồn vi.wikipedia (read-only)
- [Bengio et al. — Curriculum Learning](https://ronan.collobert.com/pub/import/2009_curriculum_icml.pdf) — train mẫu dễ trước, khó sau

## Mục tiêu

Tiny GPT là **character-level** (mỗi token = 1 ký tự). Với tiếng Việt, model học theo thứ tự:

1. **Chính tả & dấu** — `a`, `ă`, `â`, `đ`, `ư`, `ơ`…
2. **Từ & cụm từ ngắn** — câu mẫu, hội thoại
3. **Ngữ cảnh dài** — đoạn văn Wikipedia (sự kiện, địa danh, khái niệm)

## Một nút trong UI (khuyến nghị)

Ở màn hình chính, ô **Train** → nhập link bài bắt đầu (mặc định `Tiếng_Việt`) và số step mỗi bài (mặc định 300) → **Start Vietnamese training**.

Mỗi đợt:

1. Lấy text thuần của 1 bài vi.wikipedia và train ngay
2. Trong lúc train, prefetch sẵn bài tiếp theo (đi theo link trong các bài đã đọc)
3. Đủ số step thì nạp bài mới, cộng thêm một phần câu của bài cũ (replay, chống quên), train tiếp trên cùng weights
4. Lặp cho tới khi bấm **Stop**, lúc đó model được lưu thành 1 version trong `models/exports/`

Vocab cố định (bảng chữ cái tiếng Việt có dấu, chữ số, dấu câu) nên weights dùng tiếp được qua mọi bài; ký tự ngoài vocab bị bỏ. Lần đầu bấm Start nếu model hiện tại dùng vocab khác thì model được tạo lại theo vocab này.

## CLI (corpus lớn hơn)

```bash
# Phase 1: câu mẫu
npm run train:tiny -- --preset small --corpus train-data/vi-sample.txt --steps 3000 --out models/vi-phase1

# Phase 2: thêm Wikipedia (gộp file .txt, một dòng = một câu/đoạn)
npm run train:tiny -- --resume models/vi-phase1 --corpus train-data/vi-wiki.txt --steps 10000 --out models/vi-phase2

# Chat thử
npm run chat:tiny -- --model models/vi-phase2 --temperature 0.8
```

Corpus format: `.txt` — **mỗi dòng một ví dụ** (một câu hoặc một đoạn). Vocab build tự động (NFC), không cần tokenizer riêng.

## Dev với auto-reload

```bash
npm run dev:tiny:watch   # Vite HMR + tsc --watch cho tiny-llm
npm run dev:tiny         # chỉ Vite (đã alias trực tiếp src tiny-llm)
```

## Kỳ vọng thực tế

| Corpus | Model | Kết quả thường thấy |
|--------|-------|---------------------|
| vi-sample (~ vài chục dòng) | mini/small | Nhận dấu, vài từ |
| 1–5 MB text VI | small/medium | Câu ngắn có nghĩa |
| 10 MB+ (Wikipedia + sách) | medium/large (CLI) | Đoạn văn mạch lạc hơn |

Model **mini** trong browser chỉ để học/h demo — không đủ capacity cho tiếng Việt “trôi chảy”. Dùng **small** trở lên + nhiều text.

## Nguồn dữ liệu hợp lệ

- [vi.wikipedia.org](https://vi.wikipedia.org) — API `action=query&prop=extracts` (UI đã tích hợp)
- [Wikimedia dumps](https://dumps.wikimedia.org/viwiki/) — dump XML cho training offline lớn
- Sách/tin tức `.txt` tự thu thập (một dòng một câu)

Không cần gán nhãn — chỉ cần plain text; objective là predict next character.

## Export model sau training

Mỗi lần export → **1 folder riêng** trong `models/exports/`:

```text
models/exports/
  index.json
  v001-step120-2026-10-02-14-30-00/
    model.json · weights.bin · meta.json · README.txt
  v002-step500-…/
```

| Nơi train | Cách lưu |
|-----------|----------|
| **Browser** | **Stop** hoặc **Export version** → lưu `models/exports/v00N-…/` |
| **CLI** | `--out models/mini` + tự thêm bản versioned trong `models/exports/` |

### Xem lại & chạy lại model

```bash
npm run models:list                                    # danh sách v001, v002, …
npm run start:model -- v001 --mode chat                # test chat terminal
npm run start:model -- v002 --mode train --steps 5000  # train tiếp
npm run start:model -- v001 --mode web                 # mở UI đã load model
```

Trong UI: dropdown **Load** → chọn version → **Load** → Train hoặc Chat.
