import { getValue, matrixFromRows, type Matrix } from "@math-llm/core";
import type { CellPosition, ColorMode } from "../components/MatrixHeatmap";
import { formatNumber, formatPercent } from "../utils/format";
import type { Frame, MatrixPanel } from "./frame-types";

export const num = (value: number) => formatNumber(value, 4);

/** Wraps negative numbers in parentheses so "a × (−b)" reads naturally. */
export const term = (value: number) => (value < 0 ? `(${num(value)})` : num(value));

export const rowMatrix = (values: number[]) => matrixFromRows([values]);

export const rowCells = (row: number, cols: number): CellPosition[] =>
  Array.from({ length: cols }, (_, col) => ({ row, col }));

/** A matrix as it should be displayed: its data, title and axis labels. */
export interface Operand {
  key: string;
  title: string;
  matrix: Matrix;
  rowLabels: string[];
  columnLabels: string[];
  colorMode?: ColorMode;
  digits?: number;
}

export function panel(operand: Operand, extra: Partial<MatrixPanel> = {}): MatrixPanel {
  return { ...operand, ...extra };
}

/** Remembers which cells of a result matrix have been computed so far. */
export class RevealTracker {
  private readonly cells = new Set<number>();

  constructor(private readonly cols: number) {}

  add(row: number, col: number): void {
    this.cells.add(row * this.cols + col);
  }

  addRow(row: number): void {
    for (let col = 0; col < this.cols; col += 1) this.add(row, col);
  }

  addAll(rows: number): void {
    for (let row = 0; row < rows; row += 1) this.addRow(row);
  }

  snapshot(): (row: number, col: number) => boolean {
    const copy = new Set(this.cells);
    return (row, col) => copy.has(row * this.cols + col);
  }
}

export interface MatMulSpec {
  intro: string;
  left: Operand;
  leftName: string;
  right: Operand;
  rightName: string;
  /** The formula uses rightᵀ: cell [i, j] combines row i of left with ROW j of right. */
  rightTransposed?: boolean;
  bias?: Operand;
  biasName?: string;
  scale?: { factor: number; description: string };
  result: Operand;
  resultName: string;
  /** The row that is calculated cell by cell. */
  focusRow: number;
  /** "full": every product of one cell, then every other cell of the focus row. "summary": the result appears at once. */
  detail: "full" | "summary";
  /** When true, the other rows are filled one by one before the focus row; otherwise all at once afterwards. */
  otherRowsFirst: boolean;
  otherRowsNote: string;
  extraPanels?: MatrixPanel[];
}

interface MatMulFocus {
  row?: number;
  col?: number;
  /** Index of the product currently being calculated. */
  k?: number;
  biasStep?: boolean;
  active?: CellPosition[];
}

/** Frames for `result = left · right (× scale) (+ bias)`, calculated the way you would by hand. */
export function matMulFrames(spec: MatMulSpec): Frame[] {
  const { left, right, bias, result, focusRow } = spec;
  const rows = result.matrix.rows;
  const cols = result.matrix.cols;
  const inner = left.matrix.cols;
  const revealed = new RevealTracker(cols);
  const frames: Frame[] = [];
  const rowLabel = (row: number) => result.rowLabels[row];
  const columnLabel = (col: number) => result.columnLabels[col];
  const rightSliceName = spec.rightTransposed ? "row" : "column";

  const rightValue = (k: number, col: number) =>
    spec.rightTransposed ? getValue(right.matrix, col, k) : getValue(right.matrix, k, col);
  const rightCellName = (k: number, col: number) =>
    spec.rightTransposed
      ? `${spec.rightName}[${right.rowLabels[col]}, ${right.columnLabels[k]}]`
      : `${spec.rightName}[${right.rowLabels[k]}, ${right.columnLabels[col]}]`;

  const panels = ({ row, col, k, biasStep, active = [] }: MatMulFocus): MatrixPanel[] => {
    const reading = k !== undefined;
    const leftPanel = panel(left, {
      highlightedRows: row !== undefined && !reading ? [row] : [],
      highlightedCells: row !== undefined && reading ? [{ row, col: k }] : [],
    });

    let rightExtra: Partial<MatrixPanel> = {};
    if (col !== undefined && reading) {
      rightExtra = { highlightedCells: [spec.rightTransposed ? { row: col, col: k } : { row: k, col }] };
    } else if (col !== undefined) {
      rightExtra = spec.rightTransposed ? { highlightedRows: [col] } : { highlightedColumns: [col] };
    }

    const list = [...(spec.extraPanels ?? []), leftPanel, panel(right, rightExtra)];
    if (bias) {
      list.push(panel(bias, { highlightedCells: col !== undefined && (biasStep || !reading) ? [{ row: 0, col }] : [] }));
    }
    list.push(panel(result, { isCellRevealed: revealed.snapshot(), activeCells: active }));
    return list;
  };

  const push = (narration: string, calculation: string[], focus: MatMulFocus) =>
    frames.push({ narration, calculation, panels: panels(focus) });

  const productsExpression = (row: number, col: number) =>
    Array.from({ length: inner }, (_, k) => `${term(getValue(left.matrix, row, k))}×${term(rightValue(k, col))}`).join(
      " + ",
    );

  const finishLine = (row: number, col: number) => {
    let value = 0;
    for (let k = 0; k < inner; k += 1) value += getValue(left.matrix, row, k) * rightValue(k, col);
    let line = `= ${num(value)}`;
    if (spec.scale) {
      value *= spec.scale.factor;
      line += `   × ${num(spec.scale.factor)} (${spec.scale.description}) = ${num(value)}`;
    }
    if (bias) {
      const biasValue = getValue(bias.matrix, 0, col);
      value += biasValue;
      line += `   + ${spec.biasName ?? "bias"} ${term(biasValue)} = ${num(value)}`;
    }
    return line;
  };

  push(spec.intro, [], {});

  if (spec.detail === "summary") {
    revealed.addAll(rows);
    push(
      spec.otherRowsNote,
      [
        `Example, ${spec.resultName}[${rowLabel(focusRow)}, ${columnLabel(0)}] = ${productsExpression(focusRow, 0)}`,
        finishLine(focusRow, 0),
      ],
      { row: focusRow, col: 0, active: [{ row: focusRow, col: 0 }] },
    );
    return frames;
  }

  if (spec.otherRowsFirst) {
    for (let row = 0; row < rows; row += 1) {
      if (row === focusRow) continue;
      revealed.addRow(row);
      push(
        `Row ${rowLabel(row)}: every cell combines row ${rowLabel(row)} of ${spec.leftName} with one ${rightSliceName} of ${spec.rightName}. ${spec.otherRowsNote}`,
        [],
        { row, active: rowCells(row, cols) },
      );
    }
  }

  const firstColumn = 0;
  const cellName = `${spec.resultName}[${rowLabel(focusRow)}, ${columnLabel(firstColumn)}]`;
  let runningSum = 0;
  for (let k = 0; k < inner; k += 1) {
    const a = getValue(left.matrix, focusRow, k);
    const b = rightValue(k, firstColumn);
    runningSum += a * b;
    push(
      k === 0
        ? `Now one cell in full detail: ${cellName}. Take row ${rowLabel(focusRow)} of ${spec.leftName} and ${rightSliceName} ${columnLabel(firstColumn)} of ${spec.rightName}. Multiply the two lists pair by pair (${inner} pairs) and keep a running sum.`
        : `Pair ${k + 1} of ${inner}: multiply, then add to the running sum.`,
      [
        `${spec.leftName}[${rowLabel(focusRow)}, ${left.columnLabels[k]}] × ${rightCellName(k, firstColumn)} = ${term(a)} × ${term(b)} = ${num(a * b)}`,
        `running sum = ${num(runningSum)}`,
      ],
      { row: focusRow, col: firstColumn, k, active: [{ row: focusRow, col: firstColumn }] },
    );
  }

  revealed.add(focusRow, firstColumn);
  const closing = [`sum of all ${inner} products = ${num(runningSum)}`];
  let value = runningSum;
  if (spec.scale) {
    closing.push(`${num(value)} × ${num(spec.scale.factor)} (${spec.scale.description}) = ${num(value * spec.scale.factor)}`);
    value *= spec.scale.factor;
  }
  if (bias) {
    const biasValue = getValue(bias.matrix, 0, firstColumn);
    closing.push(`${num(value)} + ${spec.biasName ?? "bias"}[${columnLabel(firstColumn)}] ${term(biasValue)} = ${num(value + biasValue)}`);
    value += biasValue;
  }
  push(
    `Cell finished: ${cellName} = ${num(getValue(result.matrix, focusRow, firstColumn))}.${
      bias ? ` The bias is a learned constant added to every cell of column ${columnLabel(firstColumn)}.` : ""
    }`,
    closing,
    { row: focusRow, col: firstColumn, biasStep: true, active: [{ row: focusRow, col: firstColumn }] },
  );

  for (let col = 0; col < cols; col += 1) {
    if (col === firstColumn) continue;
    revealed.add(focusRow, col);
    push(
      `${spec.resultName}[${rowLabel(focusRow)}, ${columnLabel(col)}]: same row of ${spec.leftName}, next ${rightSliceName} of ${spec.rightName}. Same recipe: multiply pair by pair, add everything up.`,
      [productsExpression(focusRow, col), finishLine(focusRow, col)],
      { row: focusRow, col, active: [{ row: focusRow, col }] },
    );
  }

  if (!spec.otherRowsFirst && rows > 1) {
    revealed.addAll(rows);
    push(spec.otherRowsNote, [], {});
  } else {
    push(`${spec.resultName} is complete: ${rows} × ${cols} numbers.`, [], {});
  }
  return frames;
}

export interface SoftmaxSpec {
  intro: string;
  labels: string[];
  scores: number[];
  scoresTitle: string;
  resultTitle: string;
  extraPanels?: MatrixPanel[];
  closing: string;
}

export function softmaxFrames(spec: SoftmaxSpec): Frame[] {
  const finiteScores = spec.scores.filter(Number.isFinite);
  const max = Math.max(...finiteScores);
  const exponentials = spec.scores.map((score) => (Number.isFinite(score) ? Math.exp(score - max) : 0));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  const probabilities = exponentials.map((value) => value / total);
  const size = spec.scores.length;

  const scores: Operand = { key: "scores", title: spec.scoresTitle, matrix: rowMatrix(spec.scores), rowLabels: [""], columnLabels: spec.labels };
  const exps: Operand = { key: "exps", title: "e^(score − max)", matrix: rowMatrix(exponentials), rowLabels: [""], columnLabels: spec.labels, digits: 3 };
  const probs: Operand = { key: "probs", title: spec.resultTitle, matrix: rowMatrix(probabilities), rowLabels: [""], columnLabels: spec.labels, colorMode: "probability", digits: 3 };

  const expRevealed = new RevealTracker(size);
  const probRevealed = new RevealTracker(size);
  const frames: Frame[] = [];
  const push = (narration: string, calculation: string[], focus: { score?: number[]; exp?: number[]; writeExp?: number; writeProb?: number }) =>
    frames.push({
      narration,
      calculation,
      panels: [
        ...(spec.extraPanels ?? []),
        panel(scores, { highlightedCells: (focus.score ?? []).map((col) => ({ row: 0, col })) }),
        panel(exps, {
          isCellRevealed: expRevealed.snapshot(),
          highlightedCells: (focus.exp ?? []).map((col) => ({ row: 0, col })),
          activeCells: focus.writeExp === undefined ? [] : [{ row: 0, col: focus.writeExp }],
        }),
        panel(probs, {
          isCellRevealed: probRevealed.snapshot(),
          activeCells: focus.writeProb === undefined ? [] : [{ row: 0, col: focus.writeProb }],
        }),
      ],
    });

  const all = Array.from({ length: size }, (_, index) => index);
  push(spec.intro, [], {});
  push(
    "First find the largest score. Subtracting it from every score does not change the final percentages, it only keeps e^x from becoming astronomically large.",
    [`max = ${num(max)}`],
    { score: all.filter((index) => spec.scores[index] === max) },
  );

  all.forEach((index) => {
    expRevealed.add(0, index);
    const score = spec.scores[index];
    push(
      Number.isFinite(score)
        ? `Raise e (≈ 2.718) to the power (score − max) for "${spec.labels[index]}". Bigger scores give much bigger results.`
        : `"${spec.labels[index]}" is masked (−∞). e^(−∞) = 0, so it will get exactly 0%.`,
      [Number.isFinite(score) ? `e^(${num(score)} − ${num(max)}) = e^(${num(score - max)}) = ${num(exponentials[index])}` : "e^(−∞) = 0"],
      { score: [index], writeExp: index },
    );
  });

  push(
    "Add all of them up. This total is what every value gets divided by.",
    [`sum = ${exponentials.map(num).join(" + ")} = ${num(total)}`],
    { exp: all },
  );

  all.forEach((index) => {
    probRevealed.add(0, index);
    push(
      `Share of "${spec.labels[index]}": its value divided by the total.`,
      [`${num(exponentials[index])} ÷ ${num(total)} = ${num(probabilities[index])} = ${formatPercent(probabilities[index])}`],
      { exp: [index], writeProb: index },
    );
  });

  push(spec.closing, [`${probabilities.map((p) => formatPercent(p)).join(" + ")} = 100%`], {});
  return frames;
}

export interface LayerNormSpec {
  intro: string;
  rowLabel: string;
  input: number[];
  normalized: number[];
  output: number[];
  scale: number[];
  shift: number[];
  standardDeviation: number;
}

export function layerNormFrames(spec: LayerNormSpec): Frame[] {
  const size = spec.input.length;
  const labels = Array.from({ length: size }, (_, index) => String(index));
  const mean = spec.input.reduce((sum, value) => sum + value, 0) / size;
  const variance = spec.input.reduce((sum, value) => sum + (value - mean) ** 2, 0) / size;

  const operand = (key: string, title: string, values: number[]): Operand => ({
    key,
    title,
    matrix: rowMatrix(values),
    rowLabels: [spec.rowLabel],
    columnLabels: labels,
  });
  const input = operand("input", "input row", spec.input);
  const normalized = operand("normalized", "normalized", spec.normalized);
  const scale = operand("scale", "scale (learned)", spec.scale);
  const shift = operand("shift", "shift (learned)", spec.shift);
  const output = operand("output", "output row", spec.output);

  const frame = (narration: string, calculation: string[], step: number): Frame => ({
    narration,
    calculation,
    panels: [
      panel(input, { highlightedRows: step === 1 || step === 2 ? [0] : [] }),
      panel(normalized, { isCellRevealed: () => step >= 3, activeCells: step === 3 ? rowCells(0, size) : [] }),
      panel(scale, { highlightedRows: step === 4 ? [0] : [] }),
      panel(shift, { highlightedRows: step === 4 ? [0] : [] }),
      panel(output, { isCellRevealed: () => step >= 4, activeCells: step === 4 ? rowCells(0, size) : [] }),
    ],
  });

  const firstThree = [0, 1, 2];
  return [
    frame(spec.intro, [], 0),
    frame(
      "Step 1: the mean (average) of the row.",
      [`mean = (${spec.input.map(term).join(" + ")}) ÷ ${size} = ${num(mean)}`],
      1,
    ),
    frame(
      "Step 2: the standard deviation, i.e. how spread out the numbers are around the mean.",
      [
        `variance = average of (value − mean)² = ${num(variance)}`,
        `standard deviation = √(variance + 0.00001) = ${num(spec.standardDeviation)}`,
      ],
      2,
    ),
    frame(
      "Step 3: subtract the mean and divide by the standard deviation. The row now has mean 0 and spread 1.",
      [
        ...firstThree.map(
          (index) => `feature ${index}: (${num(spec.input[index])} − ${num(mean)}) ÷ ${num(spec.standardDeviation)} = ${num(spec.normalized[index])}`,
        ),
        `… the same for all ${size} features`,
      ],
      3,
    ),
    frame(
      "Step 4: multiply by the learned scale and add the learned shift, feature by feature.",
      [
        ...firstThree.map(
          (index) =>
            `feature ${index}: ${num(spec.normalized[index])} × ${term(spec.scale[index])} + ${term(spec.shift[index])} = ${num(spec.output[index])}`,
        ),
        `… the same for all ${size} features`,
      ],
      4,
    ),
  ];
}

export interface AddSpec {
  intro: string;
  a: Operand;
  aName: string;
  b: Operand;
  bName: string;
  result: Operand;
  resultName: string;
  /** Rows calculated one frame each; all other rows appear in a final frame with `otherRowsNote`. */
  rows: number[];
  otherRowsNote?: string;
}

/** Frames for element-by-element addition: result[i, j] = a[i, j] + b[i, j]. */
export function addFrames(spec: AddSpec): Frame[] {
  const { a, b, result } = spec;
  const cols = result.matrix.cols;
  const revealed = new RevealTracker(cols);
  const frames: Frame[] = [];
  const push = (narration: string, calculation: string[], row?: number) =>
    frames.push({
      narration,
      calculation,
      panels: [
        panel(a, { highlightedRows: row === undefined ? [] : [row] }),
        panel(b, { highlightedRows: row === undefined ? [] : [row] }),
        panel(result, { isCellRevealed: revealed.snapshot(), activeCells: row === undefined ? [] : rowCells(row, cols) }),
      ],
    });

  push(spec.intro, [], undefined);
  for (const row of spec.rows) {
    revealed.addRow(row);
    const label = result.rowLabels[row];
    push(
      `Row ${label}: add the two rows number by number (no multiplication).`,
      [
        ...[0, 1, 2].map(
          (col) =>
            `${spec.resultName}[${label}, ${col}] = ${num(getValue(a.matrix, row, col))} + ${term(getValue(b.matrix, row, col))} = ${num(getValue(result.matrix, row, col))}`,
        ),
        `… the same for all ${cols} columns`,
      ],
      row,
    );
  }
  if (spec.otherRowsNote) {
    revealed.addAll(result.matrix.rows);
    push(spec.otherRowsNote, [], undefined);
  }
  return frames;
}
