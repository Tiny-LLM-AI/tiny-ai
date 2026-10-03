"""Prepare a dialogue-only corpus: news used to overwhelm the 33 chat examples."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
chat = root / "train-data/vi-chat-basic.txt"
out = root / "train-data/vi-chat-training.txt"
lines = list(dict.fromkeys(line.strip() for line in chat.read_text(encoding="utf-8").splitlines() if line.strip()))
if not lines:
    raise SystemExit("Chat corpus is empty.")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{out}: {len(lines)} dialogue examples; no news dilution")
