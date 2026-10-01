import type { ModelConfig } from "../config.js";
import {
  addInPlace,
  cloneMatrix,
  createMatrix,
  matMul,
  matMulTransposeA,
  matMulTransposeB,
  sliceColumns,
  sumRows,
  writeColumns,
  type Matrix,
} from "../math/matrix.js";
import type { BlockTrace, ForwardTrace } from "./forward.js";
import { layerNormBackward } from "./layer-norm.js";
import type { BlockParameters, ModelParameters } from "./parameters.js";

/**
 * Gradient of the loss with respect to the logits.
 * For softmax + cross-entropy this is simply: probabilities − oneHot(target).
 * Negative entry → that score should go up. Positive entry → it should go down.
 */
export function logitGradient(trace: ForwardTrace, targetTokenId: number): Matrix {
  const gradient = cloneMatrix(trace.probabilities);
  gradient.data[targetTokenId] -= 1;
  return gradient;
}

/** Backward pass through softmax for every row: dScores = A ⊙ (dA − rowSum(dA ⊙ A)). */
function softmaxRowsBackward(weights: Matrix, weightsGradient: Matrix): Matrix {
  const out = createMatrix(weights.rows, weights.cols);
  for (let r = 0; r < weights.rows; r += 1) {
    const offset = r * weights.cols;
    let dot = 0;
    for (let c = 0; c < weights.cols; c += 1) dot += weightsGradient.data[offset + c] * weights.data[offset + c];
    for (let c = 0; c < weights.cols; c += 1) {
      out.data[offset + c] = weights.data[offset + c] * (weightsGradient.data[offset + c] - dot);
    }
  }
  return out;
}

/** Returns the gradient with respect to the block input and accumulates parameter gradients. */
function blockBackward(
  outputGradient: Matrix,
  trace: BlockTrace,
  block: BlockParameters,
  gradients: BlockParameters,
  config: ModelConfig,
): Matrix {
  const headSize = config.embeddingSize / config.headCount;

  // Layer norm after the MLP
  const feedForwardNorm = layerNormBackward(outputGradient, trace.feedForwardNorm, block.feedForwardNormScale);
  addInPlace(gradients.feedForwardNormScale, feedForwardNorm.scaleGradient);
  addInPlace(gradients.feedForwardNormShift, feedForwardNorm.shiftGradient);
  const residualGradient = feedForwardNorm.inputGradient;

  // MLP second layer
  addInPlace(gradients.feedForwardOutputWeight, matMulTransposeA(trace.feedForwardActivated, residualGradient));
  addInPlace(gradients.feedForwardOutputBias, sumRows(residualGradient));
  const activatedGradient = matMulTransposeB(residualGradient, block.feedForwardOutputWeight);

  // ReLU lets the gradient through only where the hidden value was positive
  const hiddenGradient = cloneMatrix(activatedGradient);
  for (let i = 0; i < hiddenGradient.data.length; i += 1) {
    if (trace.feedForwardHidden.data[i] <= 0) hiddenGradient.data[i] = 0;
  }

  // MLP first layer
  addInPlace(gradients.feedForwardInputWeight, matMulTransposeA(trace.attentionNorm.output, hiddenGradient));
  addInPlace(gradients.feedForwardInputBias, sumRows(hiddenGradient));
  const attentionNormOutputGradient = matMulTransposeB(hiddenGradient, block.feedForwardInputWeight);
  addInPlace(attentionNormOutputGradient, residualGradient);

  // Layer norm after attention
  const attentionNorm = layerNormBackward(attentionNormOutputGradient, trace.attentionNorm, block.attentionNormScale);
  addInPlace(gradients.attentionNormScale, attentionNorm.scaleGradient);
  addInPlace(gradients.attentionNormShift, attentionNorm.shiftGradient);
  const attentionResidualGradient = attentionNorm.inputGradient;

  // Attention output projection
  addInPlace(gradients.attentionOutputWeight, matMulTransposeA(trace.concatenatedHeads, attentionResidualGradient));
  addInPlace(gradients.attentionOutputBias, sumRows(attentionResidualGradient));
  const concatenatedHeadsGradient = matMulTransposeB(attentionResidualGradient, block.attentionOutputWeight);

  // Each attention head
  const queryGradient = createMatrix(trace.query.rows, trace.query.cols);
  const keyGradient = createMatrix(trace.key.rows, trace.key.cols);
  const valueGradient = createMatrix(trace.value.rows, trace.value.cols);

  trace.heads.forEach((head, index) => {
    const start = index * headSize;
    const headOutputGradient = sliceColumns(concatenatedHeadsGradient, start, start + headSize);

    const weightsGradient = matMulTransposeB(headOutputGradient, head.value);
    const headValueGradient = matMulTransposeA(head.attentionWeights, headOutputGradient);

    const scoresGradient = softmaxRowsBackward(head.attentionWeights, weightsGradient);
    const scaleFactor = 1 / Math.sqrt(headSize);
    for (let i = 0; i < scoresGradient.data.length; i += 1) scoresGradient.data[i] *= scaleFactor;

    writeColumns(queryGradient, matMul(scoresGradient, head.key), start);
    writeColumns(keyGradient, matMulTransposeA(scoresGradient, head.query), start);
    writeColumns(valueGradient, headValueGradient, start);
  });

  // Q, K, V projections
  addInPlace(gradients.queryWeight, matMulTransposeA(trace.input, queryGradient));
  addInPlace(gradients.queryBias, sumRows(queryGradient));
  addInPlace(gradients.keyWeight, matMulTransposeA(trace.input, keyGradient));
  addInPlace(gradients.keyBias, sumRows(keyGradient));
  addInPlace(gradients.valueWeight, matMulTransposeA(trace.input, valueGradient));
  addInPlace(gradients.valueBias, sumRows(valueGradient));

  // The input feeds the residual connection and the three projections
  const inputGradient = cloneMatrix(attentionResidualGradient);
  addInPlace(inputGradient, matMulTransposeB(queryGradient, block.queryWeight));
  addInPlace(inputGradient, matMulTransposeB(keyGradient, block.keyWeight));
  addInPlace(inputGradient, matMulTransposeB(valueGradient, block.valueWeight));
  return inputGradient;
}

/**
 * Backpropagation: walks the forward trace in reverse and adds
 * ∂loss/∂parameter for one sample into `gradients`.
 */
export function backward(
  trace: ForwardTrace,
  targetTokenId: number,
  parameters: ModelParameters,
  gradients: ModelParameters,
  config: ModelConfig,
): void {
  const logitsGradient = logitGradient(trace, targetTokenId);
  addInPlace(gradients.outputWeight, matMulTransposeA(trace.finalVector, logitsGradient));
  addInPlace(gradients.outputBias, logitsGradient);
  const finalVectorGradient = matMulTransposeB(logitsGradient, parameters.outputWeight);

  // Only the last position was used for the prediction
  let hiddenGradient = createMatrix(config.contextLength, config.embeddingSize);
  hiddenGradient.data.set(finalVectorGradient.data, (config.contextLength - 1) * config.embeddingSize);

  for (let index = parameters.blocks.length - 1; index >= 0; index -= 1) {
    hiddenGradient = blockBackward(
      hiddenGradient,
      trace.blocks[index],
      parameters.blocks[index],
      gradients.blocks[index],
      config,
    );
  }

  addInPlace(gradients.positionEmbedding, hiddenGradient);
  trace.tokenIds.forEach((tokenId, position) => {
    const offset = tokenId * config.embeddingSize;
    for (let c = 0; c < config.embeddingSize; c += 1) {
      gradients.tokenEmbedding.data[offset + c] += hiddenGradient.data[position * config.embeddingSize + c];
    }
  });
}
