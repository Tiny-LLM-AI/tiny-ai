import { readFileSync, writeFileSync } from "node:fs";
import { DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG } from "../config.js";
import { validateDataset, type ArithmeticDatasetFile } from "../data/dataset.js";
import { countParameters } from "../model/parameters.js";
import { serializeModel } from "../persistence/model-file.js";
import { createTrainingSession, trainOneEpoch } from "../training/trainer.js";
import { DATASET_PATH, MODEL_PATH } from "./paths.js";

const epochsArgument = process.argv.indexOf("--epochs");
const epochs = epochsArgument === -1 ? DEFAULT_TRAINING_CONFIG.epochs : Number(process.argv[epochsArgument + 1]);

const examples = validateDataset(JSON.parse(readFileSync(DATASET_PATH, "utf8")) as ArithmeticDatasetFile);
const session = createTrainingSession(examples, DEFAULT_MODEL_CONFIG, { ...DEFAULT_TRAINING_CONFIG, epochs });

console.log(`Examples: ${examples.length}, training samples: ${session.samples.length}`);
console.log(`Vocabulary: ${session.model.tokenizer.vocabulary.join(" ")}`);
console.log(`Parameters: ${countParameters(session.model.parameters)}`);

const startedAt = performance.now();
for (let epoch = 1; epoch <= epochs; epoch += 1) {
  const record = trainOneEpoch(session);
  if (epoch === 1 || epoch % 10 === 0 || epoch === epochs) {
    console.log(
      `Epoch ${String(record.epoch).padStart(3)}  loss ${record.averageLoss.toFixed(4)}  accuracy ${(record.accuracy * 100).toFixed(1)}%`,
    );
  }
}
console.log(`Training took ${((performance.now() - startedAt) / 1000).toFixed(2)}s`);

writeFileSync(MODEL_PATH, JSON.stringify(serializeModel(session.model)));
console.log(`Saved ${MODEL_PATH}`);
