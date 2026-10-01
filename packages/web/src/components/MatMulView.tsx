import { useState, type ReactNode } from "react";
import { getColumn, getRow, getValue, type Matrix } from "@math-llm/core";
import { DotProductExplainer } from "./DotProductExplainer";
import { MatrixHeatmap, type CellPosition } from "./MatrixHeatmap";

interface MatMulViewProps {
  /** Name of the output, e.g. "Q". Cells are referred to as Q[row, col]. */
  resultName: string;
  title: string;
  caption?: ReactNode;
  result: Matrix;
  rowLabels?: string[];
  columnLabels?: string[];
  left: Matrix;
  leftName: string;
  right: Matrix;
  rightName: string;
  /** True when the formula uses rightᵀ, so cell [i, j] uses row j of `right` instead of column j. */
  rightIsTransposed?: boolean;
  bias?: Matrix;
  biasName?: string;
  scaleFactor?: number;
  scaleDescription?: string;
  termLabels?: string[];
  isMasked?: (cell: CellPosition) => boolean;
  colorMode?: "signed" | "probability";
}

/**
 * Heatmap of `result = left · right (+ bias)`. Click a cell to see the
 * row of `left` and the column of `right` that were multiplied to produce it.
 */
export function MatMulView(props: MatMulViewProps) {
  const { result, left, right, rightIsTransposed, bias, isMasked } = props;
  const [selected, setSelected] = useState<CellPosition>({ row: result.rows - 1, col: 0 });
  const cell = selected.row < result.rows && selected.col < result.cols ? selected : { row: 0, col: 0 };

  const rowLabel = props.rowLabels?.[cell.row] ?? cell.row;
  const columnLabel = props.columnLabels?.[cell.col] ?? cell.col;

  return (
    <div className="matmul-view">
      <MatrixHeatmap
        matrix={result}
        title={props.title}
        caption={props.caption ?? "Click a cell to see how it was calculated."}
        rowLabels={props.rowLabels}
        columnLabels={props.columnLabels}
        colorMode={props.colorMode}
        selectedCell={cell}
        onCellClick={setSelected}
      />
      {isMasked?.(cell) ? (
        <div className="explainer">
          <div className="explainer-title">
            {props.resultName}[{rowLabel}, {columnLabel}] is masked
          </div>
          <p className="explainer-text">
            Position {cell.col} comes after position {cell.row}. A language model must not peek at characters it has
            not generated yet, so this score is replaced with −∞ and its attention weight becomes 0.
          </p>
        </div>
      ) : (
        <DotProductExplainer
          resultName={`${props.resultName}[${rowLabel}, ${columnLabel}]`}
          leftName={`${props.leftName} row ${rowLabel}`}
          rightName={`${props.rightName} ${rightIsTransposed ? "row" : "column"} ${
            rightIsTransposed ? (props.rowLabels?.[cell.col] ?? cell.col) : columnLabel
          }`}
          leftValues={getRow(left, cell.row)}
          rightValues={rightIsTransposed ? getRow(right, cell.col) : getColumn(right, cell.col)}
          termLabels={props.termLabels}
          scaleFactor={props.scaleFactor}
          scaleDescription={props.scaleDescription}
          bias={bias ? getValue(bias, 0, cell.col) : undefined}
          biasName={props.biasName}
          result={getValue(result, cell.row, cell.col)}
        />
      )}
    </div>
  );
}
