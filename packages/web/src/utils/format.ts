import { END_TOKEN, PAD_TOKEN } from "@math-llm/core";

export function formatNumber(value: number, digits = 3): string {
  if (value === -Infinity) return "−∞";
  if (value === Infinity) return "∞";
  if (Number.isNaN(value)) return "NaN";
  const text = value.toFixed(digits);
  return text.startsWith("-") ? `−${text.slice(1)}` : text;
}

export function formatPercent(probability: number, digits = 1): string {
  return `${(probability * 100).toFixed(digits)}%`;
}

/** Short, readable names for the special tokens. */
export function displayToken(token: string): string {
  if (token === PAD_TOKEN) return "PAD";
  if (token === END_TOKEN) return "END";
  return token;
}

export function indexLabels(count: number): string[] {
  return Array.from({ length: count }, (_, index) => String(index));
}
