import * as tf from "@tensorflow/tfjs";
import { beforeAll, expect, it } from "vitest";
import { buildTokenizer, createModel, DEFAULT_CONFIG, disposeModel, forward, lossFromLogits } from "../src/index.js";

beforeAll(async () => { await tf.setBackend("cpu"); await tf.ready(); });
const make = () => createModel({ ...DEFAULT_CONFIG, contextLength: 8, dModel: 8, heads: 2, layers: 2, ffnSize: 16, dropout: 0 }, buildTokenizer("abcdef"));

it("cross entropy matches an independent probability calculation", () => {
  const actual = tf.tidy(() => lossFromLogits(tf.tensor3d([[[0, Math.log(3)], [Math.log(4), 0]]]), tf.tensor2d([[1, 0]], [1, 2], "int32")).dataSync()[0]);
  expect(actual).toBeCloseTo(-(Math.log(0.75) + Math.log(0.8)) / 2, 6);
});

it("every attention head normalizes over past keys and gives future tokens zero weight", () => {
  const model = make();
  const attention: tf.Tensor[] = [];
  try {
    tf.tidy(() => forward(model, tf.tensor2d([[1, 4, 5, 6]], [1, 4], "int32"), { attention }));
    for (const tensor of attention) {
      const heads = tensor.arraySync() as number[][][][];
      for (const head of heads[0]) for (let q = 0; q < 4; q++) {
        expect(head[q].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
        for (let k = q + 1; k < 4; k++) expect(head[q][k]).toBe(0);
      }
    }
  } finally { attention.forEach(t => t.dispose()); disposeModel(model); }
});

it("batch rows cannot change one another's logits", () => {
  const model = make();
  try {
    const alone = tf.tidy(() => forward(model, tf.tensor2d([[1,4,5]], [1,3], "int32")).arraySync());
    const batch = tf.tidy(() => forward(model, tf.tensor2d([[1,4,5], [6,7,8]], [2,3], "int32")).arraySync());
    expect(batch[0]).toEqual(alone[0]);
  } finally { disposeModel(model); }
});

it("autodiff agrees with numerical derivatives throughout embedding, attention, MLP and norms", () => {
  const model = make();
  const x = tf.tensor2d([[1,4,5,6]], [1,4], "int32");
  const y = tf.tensor2d([[4,5,6,2]], [1,4], "int32");
  const objective = () => lossFromLogits(forward(model, x), y);
  const { value, grads } = tf.variableGrads(objective, [...model.params.values()]);
  try {
    for (const [name, variable] of model.params) {
      const g = grads[variable.name].dataSync();
      expect([...g].every(Number.isFinite), name).toBe(true);
      let index = 0;
      for (let i = 1; i < g.length; i++) if (Math.abs(g[i]) > Math.abs(g[index])) index = i;
      const original = Float32Array.from(variable.dataSync());
      const delta = 0.001;
      const at = (offset: number) => {
        const values = original.slice(); values[index] += offset;
        tf.tidy(() => variable.assign(tf.tensor(values, variable.shape)));
        return tf.tidy(() => objective().dataSync()[0]);
      };
      const numerical = (at(delta) - at(-delta)) / (2 * delta);
      tf.tidy(() => variable.assign(tf.tensor(original, variable.shape)));
      expect(Math.abs(numerical - g[index]), name).toBeLessThan(0.001 + 0.03 * Math.abs(g[index]));
    }
  } finally { value.dispose(); Object.values(grads).forEach(t => t.dispose()); x.dispose(); y.dispose(); disposeModel(model); }
});
