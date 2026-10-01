import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");

export const DATASET_PATH = resolve(repositoryRoot, "train-data/arithmetic.json");
export const MODEL_PATH = resolve(repositoryRoot, "tiny-model.json");
