import { createMatrix, type Matrix } from "../math/matrix.js";

const EPSILON = 1e-5;

export interface LayerNormResult {
  output: Matrix;
  /** (x - mean) / standardDeviation, before scale and shift. Needed for the backward pass. */
  normalized: Matrix;
  standardDeviations: number[];
}

/**
 * Normalizes each row to mean 0 and variance 1, then applies a learned
 * scale and shift: output = normalized × scale + shift.
 */
export function layerNormForward(input: Matrix, scale: Matrix, shift: Matrix): LayerNormResult {
  const { rows, cols } = input;
  const output = createMatrix(rows, cols);
  const normalized = createMatrix(rows, cols);
  const standardDeviations: number[] = [];

  for (let r = 0; r < rows; r += 1) {
    const offset = r * cols;
    let mean = 0;
    for (let c = 0; c < cols; c += 1) mean += input.data[offset + c];
    mean /= cols;

    let variance = 0;
    for (let c = 0; c < cols; c += 1) variance += (input.data[offset + c] - mean) ** 2;
    variance /= cols;

    const standardDeviation = Math.sqrt(variance + EPSILON);
    standardDeviations.push(standardDeviation);

    for (let c = 0; c < cols; c += 1) {
      const value = (input.data[offset + c] - mean) / standardDeviation;
      normalized.data[offset + c] = value;
      output.data[offset + c] = value * scale.data[c] + shift.data[c];
    }
  }

  return { output, normalized, standardDeviations };
}

export interface LayerNormGradients {
  inputGradient: Matrix;
  scaleGradient: Matrix;
  shiftGradient: Matrix;
}

export function layerNormBackward(outputGradient: Matrix, forward: LayerNormResult, scale: Matrix): LayerNormGradients {
  const { rows, cols } = outputGradient;
  const inputGradient = createMatrix(rows, cols);
  const scaleGradient = createMatrix(1, cols);
  const shiftGradient = createMatrix(1, cols);

  for (let r = 0; r < rows; r += 1) {
    const offset = r * cols;
    let meanOfNormalizedGradient = 0;
    let meanOfGradientTimesNormalized = 0;

    for (let c = 0; c < cols; c += 1) {
      const gradient = outputGradient.data[offset + c];
      const normalized = forward.normalized.data[offset + c];
      scaleGradient.data[c] += gradient * normalized;
      shiftGradient.data[c] += gradient;

      const normalizedGradient = gradient * scale.data[c];
      meanOfNormalizedGradient += normalizedGradient;
      meanOfGradientTimesNormalized += normalizedGradient * normalized;
    }
    meanOfNormalizedGradient /= cols;
    meanOfGradientTimesNormalized /= cols;

    for (let c = 0; c < cols; c += 1) {
      const normalizedGradient = outputGradient.data[offset + c] * scale.data[c];
      const normalized = forward.normalized.data[offset + c];
      inputGradient.data[offset + c] =
        (normalizedGradient - meanOfNormalizedGradient - normalized * meanOfGradientTimesNormalized) /
        forward.standardDeviations[r];
    }
  }

  return { inputGradient, scaleGradient, shiftGradient };
}
