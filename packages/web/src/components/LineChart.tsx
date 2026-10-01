interface LineChartProps {
  title: string;
  values: number[];
  /** Fixed y-axis maximum (e.g. 1 for accuracy). Defaults to the largest value. */
  maxValue?: number;
  formatValue: (value: number) => string;
  color: string;
}

const WIDTH = 520;
const HEIGHT = 140;
const PADDING = 28;

export function LineChart({ title, values, maxValue, formatValue, color }: LineChartProps) {
  const max = maxValue ?? Math.max(...values, 1e-9);
  const last = values.at(-1);

  const points = values
    .map((value, index) => {
      const x = PADDING + (values.length === 1 ? 0 : (index / (values.length - 1)) * (WIDTH - PADDING * 2));
      const y = HEIGHT - PADDING - (value / max) * (HEIGHT - PADDING * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="line-chart">
      <div className="chart-header">
        <span>{title}</span>
        <strong>{last === undefined ? "—" : formatValue(last)}</strong>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={title}>
        <line x1={PADDING} y1={HEIGHT - PADDING} x2={WIDTH - PADDING} y2={HEIGHT - PADDING} className="axis" />
        <line x1={PADDING} y1={PADDING} x2={PADDING} y2={HEIGHT - PADDING} className="axis" />
        <text x={4} y={PADDING + 4} className="axis-text">
          {formatValue(max)}
        </text>
        <text x={4} y={HEIGHT - PADDING + 4} className="axis-text">
          0
        </text>
        <text x={WIDTH - PADDING} y={HEIGHT - 8} className="axis-text" textAnchor="end">
          epoch {values.length}
        </text>
        {values.length > 0 && <polyline points={points} fill="none" stroke={color} strokeWidth={2} />}
      </svg>
    </div>
  );
}
