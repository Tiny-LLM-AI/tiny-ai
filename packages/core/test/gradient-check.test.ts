import assert from "node:assert/strict";
import { test } from "node:test";
import { backward } from "../src/model/backward.js";
import { crossEntropyLoss, forward } from "../src/model/forward.js";
import { initializeParameters, listParameters, zerosLike } from "../src/model/parameters.js";
import type { ModelConfig } from "../src/config.js";

/**
 * Compares the hand-written backward pass with a numerical estimate:
 *   ∂loss/∂w ≈ (loss(w + h) − loss(w − h)) / 2h
 * If backpropagation has a bug, the two numbers disagree.
 */
test("backward pass matches numerical gradients", () => {
  const config: ModelConfig = { contextLength: 5, embeddingSize: 8, headCount: 2, feedForwardSize: 12, blockCount: 2 };
  const vocabularySize = 7;
  const parameters = initializeParameters(config, vocabularySize, 123);
  for (const { matrix } of listParameters(parameters)) {
    for (let i = 0; i < matrix.data.length; i += 1) matrix.data[i] += (Math.sin(i * 7.1 + matrix.cols) * 0.3);
  }
  const tokenIds = [0, 3, 2, 5, 4];
  const target = 6;

  const gradients = zerosLike(parameters);
  backward(forward(parameters, config, tokenIds), target, parameters, gradients, config);

  const step = 1e-5;
  const analytic = listParameters(gradients);
  let checked = 0;

  listParameters(parameters).forEach(({ name, matrix }, index) => {
    const indices = [0, Math.floor(matrix.data.length / 2), matrix.data.length - 1];
    for (const i of indices) {
      const original = matrix.data[i];
      matrix.data[i] = original + step;
      const lossPlus = crossEntropyLoss(forward(parameters, config, tokenIds), target);
      matrix.data[i] = original - step;
      const lossMinus = crossEntropyLoss(forward(parameters, config, tokenIds), target);
      matrix.data[i] = original;

      const numerical = (lossPlus - lossMinus) / (2 * step);
      const computed = analytic[index].matrix.data[i];
      const tolerance = 1e-6 + 1e-4 * Math.max(Math.abs(numerical), Math.abs(computed));
      assert.ok(
        Math.abs(numerical - computed) <= tolerance,
        `${name}[${i}]: backward=${computed}, numerical=${numerical}`,
      );
      checked += 1;
    }
  });

  assert.ok(checked > 50);
});
