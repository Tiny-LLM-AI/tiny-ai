import { displayToken, formatNumber, formatPercent } from "../utils/format";

interface ProbabilityBarsProps {
  tokens: string[];
  probabilities: ArrayLike<number>;
  logits?: ArrayLike<number>;
  /** Shown in green: the correct answer (training) or the chosen token (inference). */
  highlightToken?: string;
  highlightLabel?: string;
}

export function ProbabilityBars({ tokens, probabilities, logits, highlightToken, highlightLabel }: ProbabilityBarsProps) {
  const order = tokens.map((_, index) => index).sort((a, b) => probabilities[b] - probabilities[a]);

  return (
    <div className="probability-bars">
      {order.map((index) => {
        const token = tokens[index];
        const highlighted = token === highlightToken;
        return (
          <div key={token} className={`probability-row ${highlighted ? "highlight" : ""}`}>
            <span className="token">{displayToken(token)}</span>
            {logits && <span className="logit">logit {formatNumber(logits[index], 2)}</span>}
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${probabilities[index] * 100}%` }} />
            </div>
            <span className="value">{formatPercent(probabilities[index])}</span>
            {highlighted && highlightLabel && <span className="badge">{highlightLabel}</span>}
          </div>
        );
      })}
    </div>
  );
}
