import { formatNumber } from "../utils/format";

interface DotProductExplainerProps {
  /** e.g. "Q[3, 5]" */
  resultName: string;
  leftName: string;
  rightName: string;
  leftValues: number[];
  rightValues: number[];
  termLabels?: string[];
  /** Multiplied with the sum before the bias, e.g. 1/√headSize for attention scores. */
  scaleFactor?: number;
  scaleDescription?: string;
  bias?: number;
  biasName?: string;
  result: number;
}

/** Shows one output cell of a matrix multiplication as an explicit sum of products. */
export function DotProductExplainer({
  resultName,
  leftName,
  rightName,
  leftValues,
  rightValues,
  termLabels,
  scaleFactor,
  scaleDescription,
  bias,
  biasName,
  result,
}: DotProductExplainerProps) {
  const products = leftValues.map((value, index) => value * rightValues[index]);
  const sum = products.reduce((total, value) => total + value, 0);
  const scaled = scaleFactor === undefined ? sum : sum * scaleFactor;

  return (
    <div className="explainer">
      <div className="explainer-title">
        How <code>{resultName}</code> is calculated
      </div>
      <p className="explainer-text">
        Multiply each number in <code>{leftName}</code> with the matching number in <code>{rightName}</code>, then add
        everything up{scaleFactor !== undefined ? `, ${scaleDescription ?? "scale"}` : ""}
        {bias !== undefined ? `, then add ${biasName ?? "the bias"}` : ""}.
      </p>
      <div className="table-scroll">
        <table className="number-table">
          <thead>
            <tr>
              <th>k</th>
              <th>{leftName}</th>
              <th />
              <th>{rightName}</th>
              <th />
              <th>product</th>
            </tr>
          </thead>
          <tbody>
            {products.map((product, index) => (
              <tr key={index}>
                <td className="muted">{termLabels?.[index] ?? index}</td>
                <td>{formatNumber(leftValues[index])}</td>
                <td className="muted">×</td>
                <td>{formatNumber(rightValues[index])}</td>
                <td className="muted">=</td>
                <td className={product >= 0 ? "positive" : "negative"}>{formatNumber(product, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="equation">
        <div>
          sum of products = <strong>{formatNumber(sum, 4)}</strong>
        </div>
        {scaleFactor !== undefined && (
          <div>
            × {formatNumber(scaleFactor, 4)} ({scaleDescription}) = <strong>{formatNumber(scaled, 4)}</strong>
          </div>
        )}
        {bias !== undefined && (
          <div>
            + {biasName ?? "bias"} ({formatNumber(bias, 4)}) = <strong>{formatNumber(scaled + bias, 4)}</strong>
          </div>
        )}
        <div className="result">
          {resultName} = <strong>{formatNumber(result, 4)}</strong>
        </div>
      </div>
    </div>
  );
}
