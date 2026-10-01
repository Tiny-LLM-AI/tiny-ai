import type { OptimizerName } from "../config.js";
import { cloneParameters, listParameters, zerosLike, type ModelParameters } from "../model/parameters.js";

export const ADAM_BETA1 = 0.9;
export const ADAM_BETA2 = 0.999;
export const ADAM_EPSILON = 1e-8;

export interface OptimizerState {
  name: OptimizerName;
  learningRate: number;
  stepCount: number;
  /** Adam only: running average of gradients ("momentum"). */
  firstMoment?: ModelParameters;
  /** Adam only: running average of squared gradients (how noisy each weight's gradient is). */
  secondMoment?: ModelParameters;
}

export function cloneOptimizerState(state: OptimizerState): OptimizerState {
  return {
    ...state,
    firstMoment: state.firstMoment && cloneParameters(state.firstMoment),
    secondMoment: state.secondMoment && cloneParameters(state.secondMoment),
  };
}

export function createOptimizer(name: OptimizerName, learningRate: number, parameters: ModelParameters): OptimizerState {
  return name === "adam"
    ? { name, learningRate, stepCount: 0, firstMoment: zerosLike(parameters), secondMoment: zerosLike(parameters) }
    : { name, learningRate, stepCount: 0 };
}

/**
 * Moves every weight against its gradient.
 *   SGD:  weight ← weight − learningRate × gradient
 *   Adam: weight ← weight − learningRate × m̂ / (√v̂ + ε)
 *         where m̂ is the averaged gradient and v̂ the averaged squared gradient.
 */
export function applyOptimizerStep(state: OptimizerState, parameters: ModelParameters, gradients: ModelParameters): void {
  state.stepCount += 1;
  const weights = listParameters(parameters);
  const grads = listParameters(gradients);

  if (state.name === "sgd") {
    weights.forEach(({ matrix }, index) => {
      const gradient = grads[index].matrix.data;
      for (let i = 0; i < matrix.data.length; i += 1) matrix.data[i] -= state.learningRate * gradient[i];
    });
    return;
  }

  const firstMoments = listParameters(state.firstMoment!);
  const secondMoments = listParameters(state.secondMoment!);
  const firstCorrection = 1 - ADAM_BETA1 ** state.stepCount;
  const secondCorrection = 1 - ADAM_BETA2 ** state.stepCount;

  weights.forEach(({ matrix }, index) => {
    const gradient = grads[index].matrix.data;
    const m = firstMoments[index].matrix.data;
    const v = secondMoments[index].matrix.data;
    for (let i = 0; i < matrix.data.length; i += 1) {
      m[i] = ADAM_BETA1 * m[i] + (1 - ADAM_BETA1) * gradient[i];
      v[i] = ADAM_BETA2 * v[i] + (1 - ADAM_BETA2) * gradient[i] ** 2;
      const mHat = m[i] / firstCorrection;
      const vHat = v[i] / secondCorrection;
      matrix.data[i] -= (state.learningRate * mHat) / (Math.sqrt(vHat) + ADAM_EPSILON);
    }
  });
}
