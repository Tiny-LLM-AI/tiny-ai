import type { ModelConfig } from "../config.js";
import { createTokenizer } from "../data/tokenizer.js";
import type { Model } from "../model/model.js";
import { countParameters, initializeParameters, listParameters } from "../model/parameters.js";

export const MODEL_FILE_FORMAT = "tiny-arithmetic-transformer/v3";

export interface ModelFile {
  format: typeof MODEL_FILE_FORMAT;
  config: ModelConfig;
  vocabulary: string[];
  parameterCount: number;
  parameters: Array<{ name: string; rows: number; cols: number; values: number[][] }>;
}

export function serializeModel(model: Model): ModelFile {
  return {
    format: MODEL_FILE_FORMAT,
    config: model.config,
    vocabulary: model.tokenizer.vocabulary,
    parameterCount: countParameters(model.parameters),
    parameters: listParameters(model.parameters).map(({ name, matrix }) => ({
      name,
      rows: matrix.rows,
      cols: matrix.cols,
      values: Array.from({ length: matrix.rows }, (_, row) =>
        Array.from(matrix.data.subarray(row * matrix.cols, (row + 1) * matrix.cols)),
      ),
    })),
  };
}

export function deserializeModel(file: ModelFile): Model {
  if (file.format !== MODEL_FILE_FORMAT) throw new Error(`Unsupported model file format: ${file.format}`);

  const tokenizer = createTokenizer(file.vocabulary);
  const parameters = initializeParameters(file.config, file.vocabulary.length, 0);
  const savedByName = new Map(file.parameters.map((saved) => [saved.name, saved]));

  for (const { name, matrix } of listParameters(parameters)) {
    const saved = savedByName.get(name);
    if (!saved || saved.rows !== matrix.rows || saved.cols !== matrix.cols) {
      throw new Error(`Model file is missing parameter "${name}" or its shape does not match.`);
    }
    matrix.data.set(saved.values.flat());
  }

  return { config: file.config, tokenizer, parameters };
}
