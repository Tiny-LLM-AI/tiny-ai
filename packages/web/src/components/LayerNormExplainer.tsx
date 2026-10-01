import { getRow, type LayerNormResult, type Matrix } from "@math-llm/core";
import { formatNumber } from "../utils/format";

interface LayerNormExplainerProps {
  input: Matrix;
  result: LayerNormResult;
  scale: Matrix;
  shift: Matrix;
  row: number;
  rowLabel: string;
}

export function LayerNormExplainer({ input, result, scale, shift, row, rowLabel }: LayerNormExplainerProps) {
  const values = getRow(input, row);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const standardDeviation = result.standardDeviations[row];
  const normalized = getRow(result.normalized, row);
  const output = getRow(result.output, row);

  return (
    <div className="explainer">
      <div className="explainer-title">Layer norm of row {rowLabel}</div>
      <p className="explainer-text">
        Layer norm rescales each row so its numbers have mean 0 and spread 1. This keeps values in a stable range no
        matter how many layers are stacked. Then a learned scale and shift let the model choose the final range per
        feature.
      </p>
      <div className="formula">
        mean = {formatNumber(mean, 4)}, standard deviation = {formatNumber(standardDeviation, 4)}
        <br />
        output = (input − mean) ÷ standard deviation × scale + shift
      </div>
      <div className="table-scroll">
        <table className="number-table">
          <thead>
            <tr>
              <th>feature</th>
              <th>input</th>
              <th>normalized</th>
              <th>× scale</th>
              <th>+ shift</th>
              <th>output</th>
            </tr>
          </thead>
          <tbody>
            {values.map((value, index) => (
              <tr key={index}>
                <td className="muted">{index}</td>
                <td>{formatNumber(value)}</td>
                <td>{formatNumber(normalized[index])}</td>
                <td>{formatNumber(scale.data[index])}</td>
                <td>{formatNumber(shift.data[index])}</td>
                <td>
                  <strong>{formatNumber(output[index])}</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
