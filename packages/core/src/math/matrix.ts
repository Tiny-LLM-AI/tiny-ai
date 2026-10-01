/**
 * A dense row-major matrix. Vectors are stored as 1×n matrices so every
 * parameter, activation and gradient in the model has the same shape type.
 */
export interface Matrix {
  rows: number;
  cols: number;
  data: Float64Array;
}

export function createMatrix(rows: number, cols: number, fill = 0): Matrix {
  const data = new Float64Array(rows * cols);
  if (fill !== 0) data.fill(fill);
  return { rows, cols, data };
}

export function matrixFromRows(values: number[][]): Matrix {
  const rows = values.length;
  const cols = rows === 0 ? 0 : values[0].length;
  const matrix = createMatrix(rows, cols);
  values.forEach((row, r) => row.forEach((value, c) => (matrix.data[r * cols + c] = value)));
  return matrix;
}

export function cloneMatrix(matrix: Matrix): Matrix {
  return { rows: matrix.rows, cols: matrix.cols, data: new Float64Array(matrix.data) };
}

export function getValue(matrix: Matrix, row: number, col: number): number {
  return matrix.data[row * matrix.cols + col];
}

export function getRow(matrix: Matrix, row: number): number[] {
  return Array.from(matrix.data.subarray(row * matrix.cols, (row + 1) * matrix.cols));
}

export function getColumn(matrix: Matrix, col: number): number[] {
  return Array.from({ length: matrix.rows }, (_, row) => matrix.data[row * matrix.cols + col]);
}

/** C = A · B */
export function matMul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) throw new Error(`matMul shape mismatch: [${a.rows}×${a.cols}] · [${b.rows}×${b.cols}]`);
  const out = createMatrix(a.rows, b.cols);
  for (let i = 0; i < a.rows; i += 1) {
    for (let k = 0; k < a.cols; k += 1) {
      const aValue = a.data[i * a.cols + k];
      if (aValue === 0) continue;
      const bOffset = k * b.cols;
      const outOffset = i * b.cols;
      for (let j = 0; j < b.cols; j += 1) out.data[outOffset + j] += aValue * b.data[bOffset + j];
    }
  }
  return out;
}

/** C = Aᵀ · B, without building Aᵀ. */
export function matMulTransposeA(a: Matrix, b: Matrix): Matrix {
  if (a.rows !== b.rows) throw new Error("matMulTransposeA shape mismatch");
  const out = createMatrix(a.cols, b.cols);
  for (let k = 0; k < a.rows; k += 1) {
    for (let i = 0; i < a.cols; i += 1) {
      const aValue = a.data[k * a.cols + i];
      if (aValue === 0) continue;
      for (let j = 0; j < b.cols; j += 1) out.data[i * b.cols + j] += aValue * b.data[k * b.cols + j];
    }
  }
  return out;
}

/** C = A · Bᵀ, without building Bᵀ. */
export function matMulTransposeB(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.cols) throw new Error("matMulTransposeB shape mismatch");
  const out = createMatrix(a.rows, b.rows);
  for (let i = 0; i < a.rows; i += 1) {
    for (let j = 0; j < b.rows; j += 1) {
      let sum = 0;
      for (let k = 0; k < a.cols; k += 1) sum += a.data[i * a.cols + k] * b.data[j * b.cols + k];
      out.data[i * b.rows + j] = sum;
    }
  }
  return out;
}

export function add(a: Matrix, b: Matrix): Matrix {
  const out = cloneMatrix(a);
  for (let i = 0; i < out.data.length; i += 1) out.data[i] += b.data[i];
  return out;
}

/** Adds a 1×cols bias row to every row of the matrix. */
export function addRowVector(matrix: Matrix, bias: Matrix): Matrix {
  const out = cloneMatrix(matrix);
  for (let r = 0; r < out.rows; r += 1) {
    for (let c = 0; c < out.cols; c += 1) out.data[r * out.cols + c] += bias.data[c];
  }
  return out;
}

export function scale(matrix: Matrix, factor: number): Matrix {
  const out = cloneMatrix(matrix);
  for (let i = 0; i < out.data.length; i += 1) out.data[i] *= factor;
  return out;
}

/** In-place: target += source */
export function addInPlace(target: Matrix, source: Matrix): void {
  for (let i = 0; i < target.data.length; i += 1) target.data[i] += source.data[i];
}

/** Sums every row into a single 1×cols row (used for bias gradients). */
export function sumRows(matrix: Matrix): Matrix {
  const out = createMatrix(1, matrix.cols);
  for (let r = 0; r < matrix.rows; r += 1) {
    for (let c = 0; c < matrix.cols; c += 1) out.data[c] += matrix.data[r * matrix.cols + c];
  }
  return out;
}

export function sliceColumns(matrix: Matrix, start: number, end: number): Matrix {
  const out = createMatrix(matrix.rows, end - start);
  for (let r = 0; r < matrix.rows; r += 1) {
    for (let c = start; c < end; c += 1) out.data[r * out.cols + (c - start)] = matrix.data[r * matrix.cols + c];
  }
  return out;
}

/** Writes `source` into the columns of `target` starting at `start`. */
export function writeColumns(target: Matrix, source: Matrix, start: number): void {
  for (let r = 0; r < source.rows; r += 1) {
    for (let c = 0; c < source.cols; c += 1) target.data[r * target.cols + start + c] = source.data[r * source.cols + c];
  }
}

export function relu(matrix: Matrix): Matrix {
  const out = cloneMatrix(matrix);
  for (let i = 0; i < out.data.length; i += 1) if (out.data[i] < 0) out.data[i] = 0;
  return out;
}

/** Softmax applied independently to every row. Entries equal to -Infinity become exactly 0. */
export function softmaxRows(matrix: Matrix): Matrix {
  const out = createMatrix(matrix.rows, matrix.cols);
  for (let r = 0; r < matrix.rows; r += 1) {
    const offset = r * matrix.cols;
    let max = -Infinity;
    for (let c = 0; c < matrix.cols; c += 1) max = Math.max(max, matrix.data[offset + c]);
    let sum = 0;
    for (let c = 0; c < matrix.cols; c += 1) {
      const value = Math.exp(matrix.data[offset + c] - max);
      out.data[offset + c] = value;
      sum += value;
    }
    for (let c = 0; c < matrix.cols; c += 1) out.data[offset + c] /= sum;
  }
  return out;
}

export function argMax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
  return best;
}
