import type { ModelConfig, TrainingConfig } from "../config.js";
import type { ArithmeticExample } from "../data/dataset.js";
import { buildTokenizerFromExamples } from "../data/tokenizer.js";
import { buildTrainingSamples, type TrainingSample } from "../data/training-samples.js";
import { scale, type Matrix } from "../math/matrix.js";
import { createRandom, shuffleInPlace } from "../math/random.js";
import { backward, logitGradient } from "../model/backward.js";
import { crossEntropyLoss, forward, type ForwardTrace } from "../model/forward.js";
import type { Model } from "../model/model.js";
import {
  cloneParameters,
  initializeParameters,
  listParameters,
  zerosLike,
  type ModelParameters,
} from "../model/parameters.js";
import { generateAnswer } from "../inference/generate.js";
import { applyOptimizerStep, cloneOptimizerState, createOptimizer, type OptimizerState } from "./optimizer.js";

export interface EpochRecord {
  epoch: number;
  averageLoss: number;
  /** Share of dataset questions the model answers exactly right. */
  accuracy: number;
}

export interface TrainingSession {
  model: Model;
  trainingConfig: TrainingConfig;
  examples: ArithmeticExample[];
  samples: TrainingSample[];
  optimizer: OptimizerState;
  random: () => number;
  epoch: number;
  history: EpochRecord[];
}

/** Full record of one single-sample training step, for explaining what changed and why. */
export interface TrainingStepReport {
  sample: TrainingSample;
  traceBefore: ForwardTrace;
  lossBefore: number;
  /** probabilities − oneHot(target): the starting point of backpropagation. */
  logitGradient: Matrix;
  gradients: ModelParameters;
  parametersBefore: ModelParameters;
  parametersAfter: ModelParameters;
  /** Optimizer state right after this update (Adam moments are needed to explain each change). */
  optimizerAfter: OptimizerState;
  traceAfter: ForwardTrace;
  lossAfter: number;
}

export function createTrainingSession(
  examples: ArithmeticExample[],
  modelConfig: ModelConfig,
  trainingConfig: TrainingConfig,
): TrainingSession {
  if (modelConfig.embeddingSize % modelConfig.headCount !== 0) {
    throw new Error("embeddingSize must be divisible by headCount.");
  }
  const tokenizer = buildTokenizerFromExamples(examples);
  const parameters = initializeParameters(modelConfig, tokenizer.vocabulary.length, trainingConfig.seed);
  const model: Model = { config: modelConfig, tokenizer, parameters };

  return {
    model,
    trainingConfig,
    examples,
    samples: buildTrainingSamples(examples, tokenizer, modelConfig.contextLength),
    optimizer: createOptimizer(trainingConfig.optimizer, trainingConfig.learningRate, parameters),
    random: createRandom(trainingConfig.seed + 1),
    epoch: 0,
    history: [],
  };
}

/** Forward + backward on every sample in the batch, then one optimizer update with the averaged gradient. */
export function trainOnBatch(session: TrainingSession, batch: TrainingSample[]): { loss: number; gradients: ModelParameters } {
  const { model } = session;
  const gradients = zerosLike(model.parameters);
  let totalLoss = 0;

  for (const sample of batch) {
    const trace = forward(model.parameters, model.config, sample.contextTokenIds);
    totalLoss += crossEntropyLoss(trace, sample.targetTokenId);
    backward(trace, sample.targetTokenId, model.parameters, gradients, model.config);
  }

  for (const { matrix } of listParameters(gradients)) {
    matrix.data.set(scale(matrix, 1 / batch.length).data);
  }
  applyOptimizerStep(session.optimizer, model.parameters, gradients);
  return { loss: totalLoss / batch.length, gradients };
}

export function measureAccuracy(model: Model, examples: ArithmeticExample[]): number {
  const correct = examples.filter(({ question, answer }) => generateAnswer(model, question).answer === answer).length;
  return correct / examples.length;
}

/** One pass over all samples in a random order. */
export function trainOneEpoch(session: TrainingSession): EpochRecord {
  const order = [...session.samples];
  shuffleInPlace(order, session.random);

  let totalLoss = 0;
  for (let start = 0; start < order.length; start += session.trainingConfig.batchSize) {
    const batch = order.slice(start, start + session.trainingConfig.batchSize);
    totalLoss += trainOnBatch(session, batch).loss * batch.length;
  }

  session.epoch += 1;
  const record: EpochRecord = {
    epoch: session.epoch,
    averageLoss: totalLoss / order.length,
    accuracy: measureAccuracy(session.model, session.examples),
  };
  session.history.push(record);
  return record;
}

/** Trains on exactly one sample and keeps every intermediate value so the update can be inspected. */
export function trainOneSampleWithReport(session: TrainingSession, sample: TrainingSample): TrainingStepReport {
  const { model } = session;
  const parametersBefore = cloneParameters(model.parameters);
  const traceBefore = forward(model.parameters, model.config, sample.contextTokenIds);
  const lossBefore = crossEntropyLoss(traceBefore, sample.targetTokenId);

  const { gradients } = trainOnBatch(session, [sample]);

  const traceAfter = forward(model.parameters, model.config, sample.contextTokenIds);
  return {
    sample,
    traceBefore,
    lossBefore,
    logitGradient: logitGradient(traceBefore, sample.targetTokenId),
    gradients,
    parametersBefore,
    parametersAfter: cloneParameters(model.parameters),
    optimizerAfter: cloneOptimizerState(session.optimizer),
    traceAfter,
    lossAfter: crossEntropyLoss(traceAfter, sample.targetTokenId),
  };
}
