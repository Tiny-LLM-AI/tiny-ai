import type { ModelConfig } from "../config.js";
import type { Tokenizer } from "../data/tokenizer.js";
import type { ModelParameters } from "./parameters.js";

/** Everything needed to run the model: its shape, its vocabulary and its learned numbers. */
export interface Model {
  config: ModelConfig;
  tokenizer: Tokenizer;
  parameters: ModelParameters;
}
