import { END_TOKEN, PAD_TOKEN } from "../config.js";
import type { ArithmeticExample } from "./dataset.js";

/**
 * Character-level tokenizer: every character is one token.
 * Token 0 is padding, token 1 marks the end of an answer.
 */
export interface Tokenizer {
  vocabulary: string[];
  tokenIdByCharacter: Map<string, number>;
}

export function createTokenizer(vocabulary: string[]): Tokenizer {
  return {
    vocabulary,
    tokenIdByCharacter: new Map(vocabulary.map((character, id) => [character, id])),
  };
}

export function buildTokenizerFromExamples(examples: ArithmeticExample[]): Tokenizer {
  const characters = new Set(examples.flatMap(({ question, answer }) => [...question, ...answer]));
  return createTokenizer([PAD_TOKEN, END_TOKEN, ...[...characters].sort()]);
}

export function encodeText(tokenizer: Tokenizer, text: string): number[] {
  return [...text].map((character) => {
    const id = tokenizer.tokenIdByCharacter.get(character);
    if (id === undefined) throw new Error(`Character "${character}" is not in the vocabulary.`);
    return id;
  });
}

export function tokenIdOf(tokenizer: Tokenizer, token: string): number {
  const id = tokenizer.tokenIdByCharacter.get(token);
  if (id === undefined) throw new Error(`Token "${token}" is not in the vocabulary.`);
  return id;
}

/** Keeps the last `contextLength` tokens and pads on the left with token 0 (<PAD>). */
export function toContextWindow(tokenIds: number[], contextLength: number): number[] {
  const window = tokenIds.slice(-contextLength);
  while (window.length < contextLength) window.unshift(0);
  return window;
}
