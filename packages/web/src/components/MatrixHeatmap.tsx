import type { ReactNode } from "react";
import type { Matrix } from "@math-llm/core";
import { formatNumber, indexLabels } from "../utils/format";
import { useDisplaySettings } from "./DisplaySettings";

export interface CellPosition {
  row: number;
  col: number;
}

/** "signed": blue = negative, red = positive. "probability": 0 → white, 1 → green. "plain": no color (ids). */
export type ColorMode = "signed" | "probability" | "plain";

interface MatrixHeatmapProps {
  matrix: Matrix;
  title: string;
  caption?: ReactNode;
  rowLabels?: string[];
  columnLabels?: string[];
  colorMode?: ColorMode;
  /** Color scale limit. Defaults to the largest absolute finite value in the matrix. */
  scaleLimit?: number;
  selectedCell?: CellPosition | null;
  highlightedRows?: number[];
  highlightedColumns?: number[];
  /** Cells currently being read (thick amber outline). */
  highlightedCells?: CellPosition[];
  /** Cells currently being written (thick orange outline). */
  activeCells?: CellPosition[];
  /** Cells for which this returns false are drawn empty: not computed yet. */
  isCellRevealed?: (row: number, col: number) => boolean;
  onCellClick?: (cell: CellPosition) => void;
  digits?: number;
}

function largestAbsoluteValue(matrix: Matrix): number {
  let max = 0;
  for (const value of matrix.data) if (Number.isFinite(value)) max = Math.max(max, Math.abs(value));
  return max || 1;
}

function cellColor(value: number, limit: number, mode: ColorMode): string {
  if (!Number.isFinite(value)) return "var(--masked)";
  if (mode === "plain") return "#f8fafc";
  if (mode === "probability") return `rgba(22, 163, 74, ${Math.min(1, Math.max(0, value)).toFixed(3)})`;
  const strength = Math.min(1, Math.abs(value) / limit);
  return value >= 0 ? `rgba(220, 38, 38, ${strength.toFixed(3)})` : `rgba(37, 99, 235, ${strength.toFixed(3)})`;
}

function isStrong(value: number, limit: number, mode: ColorMode): boolean {
  if (!Number.isFinite(value) || mode === "plain") return false;
  return mode === "probability" ? value > 0.55 : Math.abs(value) / limit > 0.55;
}

const containsCell = (cells: CellPosition[], row: number, col: number) =>
  cells.some((cell) => cell.row === row && cell.col === col);

export function MatrixHeatmap({
  matrix,
  title,
  caption,
  rowLabels,
  columnLabels,
  colorMode = "signed",
  scaleLimit,
  selectedCell,
  highlightedRows = [],
  highlightedColumns = [],
  highlightedCells = [],
  activeCells = [],
  isCellRevealed,
  onCellClick,
  digits = 2,
}: MatrixHeatmapProps) {
  const { showCellValues } = useDisplaySettings();
  const limit = scaleLimit ?? largestAbsoluteValue(matrix);
  const rows = rowLabels ?? indexLabels(matrix.rows);
  const columns = columnLabels ?? indexLabels(matrix.cols);

  return (
    <figure className="heatmap">
      <figcaption>
        <span className="heatmap-title">{title}</span>
        <span className="shape">
          [{matrix.rows} × {matrix.cols}]
        </span>
      </figcaption>
      <div className="heatmap-scroll">
        <div
          className={`heatmap-grid ${showCellValues ? "with-values" : ""} ${onCellClick ? "clickable" : ""}`}
          style={{ gridTemplateColumns: `auto repeat(${matrix.cols}, var(--cell-width))` }}
        >
          <div />
          {columns.map((label, col) => (
            <div key={col} className={`axis-label column ${highlightedColumns.includes(col) ? "highlight" : ""}`}>
              {label}
            </div>
          ))}
          {Array.from({ length: matrix.rows }, (_, row) => (
            <div key={row} className="heatmap-row">
              <div className={`axis-label row ${highlightedRows.includes(row) ? "highlight" : ""}`}>{rows[row]}</div>
              {Array.from({ length: matrix.cols }, (_, col) => {
                const value = matrix.data[row * matrix.cols + col];
                const revealed = isCellRevealed ? isCellRevealed(row, col) : true;
                const classes = [
                  "cell",
                  revealed ? "" : "pending",
                  revealed && isStrong(value, limit, colorMode) ? "strong" : "",
                  highlightedRows.includes(row) || highlightedColumns.includes(col) ? "highlight" : "",
                  containsCell(highlightedCells, row, col) ? "reading" : "",
                  containsCell(activeCells, row, col) ? "writing" : "",
                  selectedCell?.row === row && selectedCell?.col === col ? "selected" : "",
                ].join(" ");
                return (
                  <div
                    key={col}
                    className={classes}
                    style={revealed ? { background: cellColor(value, limit, colorMode) } : undefined}
                    title={revealed ? `row ${rows[row]}, column ${columns[col]}: ${formatNumber(value, 5)}` : "not computed yet"}
                    onClick={onCellClick ? () => onCellClick({ row, col }) : undefined}
                  >
                    {revealed && showCellValues ? formatNumber(value, digits) : ""}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      {caption && <div className="heatmap-caption">{caption}</div>}
    </figure>
  );
}
