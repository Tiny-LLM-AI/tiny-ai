export const PAD = "<PAD>";
export const BOS = "<BOS>";
export const EOS = "<EOS>";
export const UNK = "<UNK>";
export const SPECIAL_TOKENS = [PAD, BOS, EOS, UNK] as const;

export const PAD_ID = 0;
export const BOS_ID = 1;
export const EOS_ID = 2;
export const UNK_ID = 3;

/** Character-level tokenizer; the vocabulary is whatever characters appear in the corpus. */
export interface Tokenizer {
  vocab: string[];
  index: Map<string, number>;
}

export function normalizeText(text: string): string {
  return text.normalize("NFC");
}

/** Splits by Unicode code point so Vietnamese letters with diacritics stay one token. */
export function splitChars(text: string): string[] {
  return [...normalizeText(text)];
}

export function tokenizerFromVocab(vocab: string[]): Tokenizer {
  return { vocab: [...vocab], index: new Map(vocab.map((t, i) => [t, i])) };
}

export function buildTokenizer(text: string): Tokenizer {
  const chars = new Set<string>();
  for (const c of splitChars(text)) {
    if (c !== "\n" && c !== "\r") chars.add(c);
  }
  const sorted = [...chars].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  return tokenizerFromVocab([...SPECIAL_TOKENS, ...sorted]);
}

export function encode(tokenizer: Tokenizer, text: string): number[] {
  return splitChars(text).map((c) => tokenizer.index.get(c) ?? UNK_ID);
}

export function isSpecialId(id: number): boolean {
  return id < SPECIAL_TOKENS.length;
}

export function tokenText(tokenizer: Tokenizer, id: number): string {
  return tokenizer.vocab[id] ?? UNK;
}

/** Decodes ids, dropping special tokens. */
export function decode(tokenizer: Tokenizer, ids: number[]): string {
  return ids.filter((id) => !isSpecialId(id)).map((id) => tokenizer.vocab[id] ?? "").join("");
}
