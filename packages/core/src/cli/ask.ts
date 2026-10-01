import { readFileSync } from "node:fs";
import { generateAnswer } from "../inference/generate.js";
import { deserializeModel, type ModelFile } from "../persistence/model-file.js";
import { MODEL_PATH } from "./paths.js";

const question = process.argv[2];
if (!question) {
  console.error('Usage: npm run ask -- "2+3="');
  process.exit(1);
}

const model = deserializeModel(JSON.parse(readFileSync(MODEL_PATH, "utf8")) as ModelFile);
const result = generateAnswer(model, question);

for (const step of result.steps) {
  const top = step.ranking
    .slice(0, 3)
    .map(({ token, probability }) => `${token} ${(probability * 100).toFixed(1)}%`)
    .join(", ");
  console.log(`"${step.contextText}" → ${step.chosenToken}   (top 3: ${top})`);
}
console.log(`Answer: ${question}${result.answer}`);
