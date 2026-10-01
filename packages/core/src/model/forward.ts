import type { ModelConfig } from "../config.js";
import {
  add,
  addRowVector,
  createMatrix,
  getRow,
  matMul,
  matMulTransposeB,
  relu,
  scale,
  sliceColumns,
  softmaxRows,
  writeColumns,
  type Matrix,
} from "../math/matrix.js";
import { layerNormForward, type LayerNormResult } from "./layer-norm.js";
import type { BlockParameters, ModelParameters } from "./parameters.js";

export interface AttentionHeadTrace {
  /** Slices of Q, K, V that belong to this head. Shape: [contextLength × headSize] */
  query: Matrix;
  key: Matrix;
  value: Matrix;
  /** Q · Kᵀ / √headSize. Row i = how much position i "likes" every position. */
  scores: Matrix;
  /** Same as scores, but future positions (column > row) are set to -Infinity. */
  maskedScores: Matrix;
  /** softmax(maskedScores) per row. Each row sums to 1. */
  attentionWeights: Matrix;
  /** attentionWeights · V: a weighted average of the values. */
  output: Matrix;
}

export interface BlockTrace {
  input: Matrix;
  query: Matrix;
  key: Matrix;
  value: Matrix;
  heads: AttentionHeadTrace[];
  concatenatedHeads: Matrix;
  attentionOutput: Matrix;
  attentionResidual: Matrix;
  attentionNorm: LayerNormResult;
  feedForwardHidden: Matrix;
  feedForwardActivated: Matrix;
  feedForwardOutput: Matrix;
  feedForwardResidual: Matrix;
  feedForwardNorm: LayerNormResult;
  output: Matrix;
}

/** Every intermediate matrix computed while turning token ids into probabilities. */
export interface ForwardTrace {
  tokenIds: number[];
  /** Rows of tokenEmbedding picked by the token ids. */
  tokenVectors: Matrix;
  positionVectors: Matrix;
  /** tokenVectors + positionVectors */
  embeddingSum: Matrix;
  blocks: BlockTrace[];
  /** Last row of the final block output: the model only predicts from the last position. */
  finalVector: Matrix;
  /** finalVector · outputWeight + outputBias. One raw score per vocabulary character. */
  logits: Matrix;
  /** softmax(logits). One probability per vocabulary character. */
  probabilities: Matrix;
}

/** -Infinity above the diagonal: position i may only look at positions 0..i. */
export function applyCausalMask(scores: Matrix): Matrix {
  const masked = createMatrix(scores.rows, scores.cols);
  for (let row = 0; row < scores.rows; row += 1) {
    for (let col = 0; col < scores.cols; col += 1) {
      masked.data[row * scores.cols + col] = col <= row ? scores.data[row * scores.cols + col] : -Infinity;
    }
  }
  return masked;
}

function blockForward(input: Matrix, block: BlockParameters, config: ModelConfig): BlockTrace {
  const headSize = config.embeddingSize / config.headCount;

  const query = addRowVector(matMul(input, block.queryWeight), block.queryBias);
  const key = addRowVector(matMul(input, block.keyWeight), block.keyBias);
  const value = addRowVector(matMul(input, block.valueWeight), block.valueBias);

  const concatenatedHeads = createMatrix(input.rows, config.embeddingSize);
  const heads: AttentionHeadTrace[] = [];

  for (let head = 0; head < config.headCount; head += 1) {
    const start = head * headSize;
    const headQuery = sliceColumns(query, start, start + headSize);
    const headKey = sliceColumns(key, start, start + headSize);
    const headValue = sliceColumns(value, start, start + headSize);

    const scores = scale(matMulTransposeB(headQuery, headKey), 1 / Math.sqrt(headSize));
    const maskedScores = applyCausalMask(scores);
    const attentionWeights = softmaxRows(maskedScores);
    const output = matMul(attentionWeights, headValue);

    writeColumns(concatenatedHeads, output, start);
    heads.push({ query: headQuery, key: headKey, value: headValue, scores, maskedScores, attentionWeights, output });
  }

  const attentionOutput = addRowVector(matMul(concatenatedHeads, block.attentionOutputWeight), block.attentionOutputBias);
  const attentionResidual = add(input, attentionOutput);
  const attentionNorm = layerNormForward(attentionResidual, block.attentionNormScale, block.attentionNormShift);

  const feedForwardHidden = addRowVector(
    matMul(attentionNorm.output, block.feedForwardInputWeight),
    block.feedForwardInputBias,
  );
  const feedForwardActivated = relu(feedForwardHidden);
  const feedForwardOutput = addRowVector(
    matMul(feedForwardActivated, block.feedForwardOutputWeight),
    block.feedForwardOutputBias,
  );
  const feedForwardResidual = add(attentionNorm.output, feedForwardOutput);
  const feedForwardNorm = layerNormForward(feedForwardResidual, block.feedForwardNormScale, block.feedForwardNormShift);

  return {
    input,
    query,
    key,
    value,
    heads,
    concatenatedHeads,
    attentionOutput,
    attentionResidual,
    attentionNorm,
    feedForwardHidden,
    feedForwardActivated,
    feedForwardOutput,
    feedForwardResidual,
    feedForwardNorm,
    output: feedForwardNorm.output,
  };
}

export function forward(parameters: ModelParameters, config: ModelConfig, tokenIds: number[]): ForwardTrace {
  if (tokenIds.length !== config.contextLength) {
    throw new Error(`Expected ${config.contextLength} token ids, got ${tokenIds.length}.`);
  }

  const tokenVectors = createMatrix(config.contextLength, config.embeddingSize);
  tokenIds.forEach((tokenId, position) => {
    const row = getRow(parameters.tokenEmbedding, tokenId);
    tokenVectors.data.set(row, position * config.embeddingSize);
  });
  const positionVectors = parameters.positionEmbedding;
  const embeddingSum = add(tokenVectors, positionVectors);

  const blocks: BlockTrace[] = [];
  let hidden = embeddingSum;
  for (const block of parameters.blocks) {
    const trace = blockForward(hidden, block, config);
    blocks.push(trace);
    hidden = trace.output;
  }

  const finalVector = createMatrix(1, config.embeddingSize);
  finalVector.data.set(getRow(hidden, config.contextLength - 1));
  const logits = addRowVector(matMul(finalVector, parameters.outputWeight), parameters.outputBias);
  const probabilities = softmaxRows(logits);

  return { tokenIds, tokenVectors, positionVectors, embeddingSum, blocks, finalVector, logits, probabilities };
}

/** Cross-entropy loss for one sample: -ln(probability of the correct character). */
export function crossEntropyLoss(trace: ForwardTrace, targetTokenId: number): number {
  return -Math.log(Math.max(trace.probabilities.data[targetTokenId], 1e-12));
}
