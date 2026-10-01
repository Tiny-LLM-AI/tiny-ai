import { formatNumber, formatPercent } from "../utils/format";

interface SoftmaxExplainerProps {
  title: string;
  labels: string[];
  scores: number[];
  highlightIndex?: number;
}

/**
 * softmax(xᵢ) = e^(xᵢ − max) / Σ e^(xⱼ − max)
 * Subtracting the max does not change the result; it only keeps e^x from overflowing.
 */
export function SoftmaxExplainer({ title, labels, scores, highlightIndex }: SoftmaxExplainerProps) {
  const max = Math.max(...scores.filter(Number.isFinite));
  const exponentials = scores.map((score) => (Number.isFinite(score) ? Math.exp(score - max) : 0));
  const total = exponentials.reduce((sum, value) => sum + value, 0);

  return (
    <div className="explainer">
      <div className="explainer-title">{title}</div>
      <p className="explainer-text">
        Softmax turns any list of scores into probabilities that are all positive and add up to 1. A bigger score gets
        a much bigger share because of the exponential. A score of −∞ (masked) becomes exactly 0.
      </p>
      <div className="formula">softmax(scoreᵢ) = e^(scoreᵢ − max) ÷ Σ e^(scoreⱼ − max), with max = {formatNumber(max)}</div>
      <div className="table-scroll">
        <table className="number-table">
          <thead>
            <tr>
              <th />
              <th>score</th>
              <th>e^(score − max)</th>
              <th>÷ {formatNumber(total, 4)}</th>
            </tr>
          </thead>
          <tbody>
            {scores.map((score, index) => (
              <tr key={index} className={index === highlightIndex ? "row-highlight" : ""}>
                <td className="muted">{labels[index]}</td>
                <td>{formatNumber(score)}</td>
                <td>{formatNumber(exponentials[index], 4)}</td>
                <td>
                  <strong>{formatPercent(exponentials[index] / total)}</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
