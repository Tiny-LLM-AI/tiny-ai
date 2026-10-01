import type { ModelConfig } from "../config.js";
import { createMatrix, type Matrix } from "../math/matrix.js";
import { createRandom, randomNormal } from "../math/random.js";

export interface BlockParameters {
  queryWeight: Matrix;
  queryBias: Matrix;
  keyWeight: Matrix;
  keyBias: Matrix;
  valueWeight: Matrix;
  valueBias: Matrix;
  attentionOutputWeight: Matrix;
  attentionOutputBias: Matrix;
  attentionNormScale: Matrix;
  attentionNormShift: Matrix;
  feedForwardInputWeight: Matrix;
  feedForwardInputBias: Matrix;
  feedForwardOutputWeight: Matrix;
  feedForwardOutputBias: Matrix;
  feedForwardNormScale: Matrix;
  feedForwardNormShift: Matrix;
}

/** Every number the model can learn. Gradients use the same structure. */
export interface ModelParameters {
  tokenEmbedding: Matrix;
  positionEmbedding: Matrix;
  blocks: BlockParameters[];
  outputWeight: Matrix;
  outputBias: Matrix;
}

export interface NamedParameter {
  name: string;
  description: string;
  matrix: Matrix;
}

const BLOCK_PARAMETER_DESCRIPTIONS: Record<keyof BlockParameters, string> = {
  queryWeight: "Turns each character vector into a query: \"what am I looking for?\"",
  queryBias: "Constant added to every query.",
  keyWeight: "Turns each character vector into a key: \"what do I contain?\"",
  keyBias: "Constant added to every key.",
  valueWeight: "Turns each character vector into a value: \"what do I pass on if selected?\"",
  valueBias: "Constant added to every value.",
  attentionOutputWeight: "Mixes the outputs of all attention heads back together.",
  attentionOutputBias: "Constant added after mixing the heads.",
  attentionNormScale: "Layer norm scale after attention (multiplies each normalized feature).",
  attentionNormShift: "Layer norm shift after attention (added to each normalized feature).",
  feedForwardInputWeight: "First MLP layer: expands each vector to a wider hidden layer.",
  feedForwardInputBias: "Constant added in the first MLP layer.",
  feedForwardOutputWeight: "Second MLP layer: projects the hidden layer back to embedding size.",
  feedForwardOutputBias: "Constant added in the second MLP layer.",
  feedForwardNormScale: "Layer norm scale after the MLP.",
  feedForwardNormShift: "Layer norm shift after the MLP.",
};

/** Flattens the parameter tree into a named list, in a stable order. */
export function listParameters(parameters: ModelParameters): NamedParameter[] {
  const list: NamedParameter[] = [
    {
      name: "tokenEmbedding",
      description: "One row per vocabulary character: the vector that represents that character.",
      matrix: parameters.tokenEmbedding,
    },
    {
      name: "positionEmbedding",
      description: "One row per position in the context window: tells the model where a character is.",
      matrix: parameters.positionEmbedding,
    },
  ];

  parameters.blocks.forEach((block, index) => {
    for (const key of Object.keys(BLOCK_PARAMETER_DESCRIPTIONS) as Array<keyof BlockParameters>) {
      list.push({
        name: `block${index + 1}.${key}`,
        description: BLOCK_PARAMETER_DESCRIPTIONS[key],
        matrix: block[key],
      });
    }
  });

  list.push(
    {
      name: "outputWeight",
      description: "Turns the final vector into one score (logit) per vocabulary character.",
      matrix: parameters.outputWeight,
    },
    { name: "outputBias", description: "Constant added to every output score.", matrix: parameters.outputBias },
  );

  return list;
}

export function countParameters(parameters: ModelParameters): number {
  return listParameters(parameters).reduce((total, { matrix }) => total + matrix.data.length, 0);
}

function uniformMatrix(rows: number, cols: number, limit: number, random: () => number): Matrix {
  const matrix = createMatrix(rows, cols);
  for (let i = 0; i < matrix.data.length; i += 1) matrix.data[i] = (random() * 2 - 1) * limit;
  return matrix;
}

/** Glorot/Xavier uniform: keeps signal size roughly stable from layer to layer. */
function denseWeight(inputSize: number, outputSize: number, random: () => number): Matrix {
  return uniformMatrix(inputSize, outputSize, Math.sqrt(6 / (inputSize + outputSize)), random);
}

export function initializeParameters(config: ModelConfig, vocabularySize: number, seed: number): ModelParameters {
  const random = createRandom(seed);
  const { embeddingSize: d, feedForwardSize: f, contextLength } = config;

  const positionEmbedding = createMatrix(contextLength, d);
  for (let i = 0; i < positionEmbedding.data.length; i += 1) positionEmbedding.data[i] = randomNormal(random, 0.02);

  const blocks = Array.from({ length: config.blockCount }, () => ({
    queryWeight: denseWeight(d, d, random),
    queryBias: createMatrix(1, d),
    keyWeight: denseWeight(d, d, random),
    keyBias: createMatrix(1, d),
    valueWeight: denseWeight(d, d, random),
    valueBias: createMatrix(1, d),
    attentionOutputWeight: denseWeight(d, d, random),
    attentionOutputBias: createMatrix(1, d),
    attentionNormScale: createMatrix(1, d, 1),
    attentionNormShift: createMatrix(1, d),
    feedForwardInputWeight: denseWeight(d, f, random),
    feedForwardInputBias: createMatrix(1, f),
    feedForwardOutputWeight: denseWeight(f, d, random),
    feedForwardOutputBias: createMatrix(1, d),
    feedForwardNormScale: createMatrix(1, d, 1),
    feedForwardNormShift: createMatrix(1, d),
  }));

  return {
    tokenEmbedding: uniformMatrix(vocabularySize, d, 0.05, random),
    positionEmbedding,
    blocks,
    outputWeight: denseWeight(d, vocabularySize, random),
    outputBias: createMatrix(1, vocabularySize),
  };
}

/** Same structure as `parameters`, every value set to zero. Used to accumulate gradients. */
export function zerosLike(parameters: ModelParameters): ModelParameters {
  const zero = (matrix: Matrix) => createMatrix(matrix.rows, matrix.cols);
  return {
    tokenEmbedding: zero(parameters.tokenEmbedding),
    positionEmbedding: zero(parameters.positionEmbedding),
    blocks: parameters.blocks.map((block) => {
      const copy = {} as BlockParameters;
      for (const key of Object.keys(block) as Array<keyof BlockParameters>) copy[key] = zero(block[key]);
      return copy;
    }),
    outputWeight: zero(parameters.outputWeight),
    outputBias: zero(parameters.outputBias),
  };
}

export function cloneParameters(parameters: ModelParameters): ModelParameters {
  const copy = zerosLike(parameters);
  const source = listParameters(parameters);
  listParameters(copy).forEach(({ matrix }, index) => matrix.data.set(source[index].matrix.data));
  return copy;
}
