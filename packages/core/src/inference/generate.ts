import { END_TOKEN, PAD_TOKEN } from "../config.js";
import { encodeText, toContextWindow } from "../data/tokenizer.js";
import { argMax } from "../math/matrix.js";
import { forward, type ForwardTrace } from "../model/forward.js";
import type { Model } from "../model/model.js";

export interface TokenProbability {
  tokenId: number;
  token: string;
  logit: number;
  probability: number;
}

export interface PredictionStep {
  contextText: string;
  contextTokenIds: number[];
  trace: ForwardTrace;
  /** Every vocabulary token, sorted from most to least likely. */
  ranking: TokenProbability[];
  /** The model always picks the most likely token (greedy decoding). */
  chosenToken: string;
}

export interface GenerationResult {
  question: string;
  answer: string;
  steps: PredictionStep[];
}

export function predictNextToken(model: Model, contextText: string): PredictionStep {
  const contextTokenIds = toContextWindow(encodeText(model.tokenizer, contextText), model.config.contextLength);
  const trace = forward(model.parameters, model.config, contextTokenIds);

  const ranking = model.tokenizer.vocabulary
    .map((token, tokenId) => ({
      tokenId,
      token,
      logit: trace.logits.data[tokenId],
      probability: trace.probabilities.data[tokenId],
    }))
    .sort((a, b) => b.probability - a.probability);

  const chosenToken = model.tokenizer.vocabulary[argMax(trace.probabilities.data)];
  return { contextText, contextTokenIds, trace, ranking, chosenToken };
}

/** Repeats "predict next character, append it" until the model outputs <END>. */
export function generateAnswer(model: Model, question: string, maxCharacters = 6): GenerationResult {
  const steps: PredictionStep[] = [];
  let text = question;

  for (let i = 0; i < maxCharacters; i += 1) {
    const step = predictNextToken(model, text);
    steps.push(step);
    if (step.chosenToken === END_TOKEN || step.chosenToken === PAD_TOKEN) break;
    text += step.chosenToken;
  }

  return { question, answer: text.slice(question.length), steps };
}
