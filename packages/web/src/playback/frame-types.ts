import type { Matrix } from "@math-llm/core";
import type { CellPosition, ColorMode } from "../components/MatrixHeatmap";

export interface MatrixPanel {
  key: string;
  title: string;
  matrix: Matrix;
  rowLabels: string[];
  columnLabels: string[];
  colorMode?: ColorMode;
  digits?: number;
  highlightedRows?: number[];
  highlightedColumns?: number[];
  highlightedCells?: CellPosition[];
  activeCells?: CellPosition[];
  isCellRevealed?: (row: number, col: number) => boolean;
}

/** One still picture of the animation: what is shown, what is being calculated, and why. */
export interface Frame {
  narration: string;
  calculation: string[];
  panels: MatrixPanel[];
}

export interface PlaybackStage {
  title: string;
  shortTitle: string;
  formula: string;
  frames: Frame[];
}
