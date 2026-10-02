export interface LineChartOptions {
  title: string;
  values: number[];
  maxValue?: number;
  formatValue: (value: number) => string;
  color: string;
}

const WIDTH = 280;
const HEIGHT = 100;
const PADDING = 24;

export function renderLineChart({
  title,
  values,
  maxValue,
  formatValue,
  color,
}: LineChartOptions): string {
  const dataMax = values.length > 0 ? Math.max(...values) : 1;
  const max = maxValue ?? Math.max(dataMax, 1e-9);
  const last = values.at(-1);

  const points = values
    .map((value, index) => {
      const x =
        PADDING +
        (values.length === 1
          ? 0
          : (index / (values.length - 1)) * (WIDTH - PADDING * 2));
      const y = HEIGHT - PADDING - (value / max) * (HEIGHT - PADDING * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return `
    <div class="line-chart">
      <div class="chart-header">
        <span>${title}</span>
        <strong>${last === undefined ? "-" : formatValue(last)}</strong>
      </div>
      <svg viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="${title}">
        <line x1="${PADDING}" y1="${HEIGHT - PADDING}" x2="${WIDTH - PADDING}" y2="${HEIGHT - PADDING}" class="axis" />
        <line x1="${PADDING}" y1="${PADDING}" x2="${PADDING}" y2="${HEIGHT - PADDING}" class="axis" />
        <text x="4" y="${PADDING + 4}" class="axis-text">${formatValue(max)}</text>
        <text x="4" y="${HEIGHT - PADDING + 4}" class="axis-text">0</text>
        <text x="${WIDTH - PADDING}" y="${HEIGHT - 8}" class="axis-text" text-anchor="end">${values.length} pts</text>
        ${values.length > 0 ? `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2" />` : ""}
      </svg>
    </div>
  `;
}

export function paintChartRow(
  container: HTMLElement,
  lossHistory: number[],
  accuracyHistory: number[],
): void {
  container.innerHTML = `
    <div class="chart-row">
      ${renderLineChart({
        title: "Loss",
        values: lossHistory,
        formatValue: (v) => v.toFixed(3),
        color: "#2563eb",
      })}
      ${renderLineChart({
        title: "Val char acc",
        values: accuracyHistory,
        maxValue: 1,
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
        color: "#16a34a",
      })}
    </div>
  `;
}
