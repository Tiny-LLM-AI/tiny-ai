import { END_TOKEN } from "../config.js";
import type { ArithmeticExample } from "./dataset.js";
import { encodeText, toContextWindow, tokenIdOf, type Tokenizer } from "./tokenizer.js";

/**
 * One training sample = "given these characters, the next character should be X".
 * The example "2-3=" → "-1" becomes three samples:
 *   "2-3="   → "-"
 *   "2-3=-"  → "1"
 *   "2-3=-1" → <END>
 */
export interface TrainingSample {
  contextText: string;
  contextTokenIds: number[];
  targetTokenId: number;
  targetToken: string;
}

export function buildTrainingSamples(
  examples: ArithmeticExample[],
  tokenizer: Tokenizer,
  contextLength: number,
): TrainingSample[] {
  const samples: TrainingSample[] = [];

  for (const { question, answer } of examples) {
    const targets = [...answer, END_TOKEN];
    let contextText = question;

    for (const target of targets) {
      samples.push({
        contextText,
        contextTokenIds: toContextWindow(encodeText(tokenizer, contextText), contextLength),
        targetTokenId: tokenIdOf(tokenizer, target),
        targetToken: target,
      });
      contextText += target;
    }
  }

  return samples;
}
