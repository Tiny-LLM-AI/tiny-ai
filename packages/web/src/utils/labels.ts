import type { Matrix, Model } from "@math-llm/core";
import { displayToken, indexLabels } from "./format";

/** Row labels for activations: "position:character", e.g. "3:=". */
export function contextRowLabels(model: Model, tokenIds: number[]): string[] {
  return tokenIds.map((id, position) => `${position}:${displayToken(model.tokenizer.vocabulary[id])}`);
}

export function vocabularyLabels(model: Model): string[] {
  return model.tokenizer.vocabulary.map(displayToken);
}

/** Axis labels for a learnable parameter, based on what its rows and columns mean. */
export function parameterAxisLabels(
  name: string,
  matrix: Matrix,
  model: Model,
): { rowLabels: string[]; columnLabels: string[] } {
  if (name === "tokenEmbedding") return { rowLabels: vocabularyLabels(model), columnLabels: indexLabels(matrix.cols) };
  if (name === "positionEmbedding") {
    return { rowLabels: indexLabels(matrix.rows).map((i) => `pos ${i}`), columnLabels: indexLabels(matrix.cols) };
  }
  if (name === "outputWeight") return { rowLabels: indexLabels(matrix.rows), columnLabels: vocabularyLabels(model) };
  if (name === "outputBias") return { rowLabels: ["bias"], columnLabels: vocabularyLabels(model) };
  return {
    rowLabels: matrix.rows === 1 ? [""] : indexLabels(matrix.rows),
    columnLabels: indexLabels(matrix.cols),
  };
}
