export interface ModelConfig {
  /** How many characters the model can look at (the context window). */
  contextLength: number;
  /** Length of the vector that represents one character. */
  embeddingSize: number;
  /** Attention is split into this many independent heads. */
  headCount: number;
  /** Hidden layer width inside the feed-forward network (MLP). */
  feedForwardSize: number;
  /** How many Transformer blocks are stacked. */
  blockCount: number;
}

export type OptimizerName = "sgd" | "adam";

export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: OptimizerName;
  seed: number;
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  contextLength: 8,
  embeddingSize: 16,
  headCount: 2,
  feedForwardSize: 32,
  blockCount: 1,
};

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  learningRate: 0.01,
  batchSize: 16,
  epochs: 100,
  optimizer: "adam",
  seed: 42,
};

export const PAD_TOKEN = "<PAD>";
export const END_TOKEN = "<END>";
