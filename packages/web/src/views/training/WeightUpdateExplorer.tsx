import { useState } from "react";
import {
  ADAM_BETA1,
  ADAM_BETA2,
  ADAM_EPSILON,
  createMatrix,
  getValue,
  listParameters,
  type Model,
  type TrainingStepReport,
} from "@math-llm/core";
import { MatrixHeatmap, type CellPosition } from "../../components/MatrixHeatmap";
import { formatNumber } from "../../utils/format";
import { parameterAxisLabels } from "../../utils/labels";

interface WeightUpdateExplorerProps {
  model: Model;
  report: TrainingStepReport;
  parameterName: string;
  onParameterChange: (name: string) => void;
}

export function WeightUpdateExplorer({ model, report, parameterName, onParameterChange }: WeightUpdateExplorerProps) {
  const [selected, setSelected] = useState<CellPosition>({ row: 0, col: report.sample.targetTokenId });

  const before = listParameters(report.parametersBefore);
  const index = Math.max(0, before.findIndex(({ name }) => name === parameterName));
  const { name, description, matrix: weightBefore } = before[index];
  const gradient = listParameters(report.gradients)[index].matrix;
  const weightAfter = listParameters(report.parametersAfter)[index].matrix;

  const change = createMatrix(weightBefore.rows, weightBefore.cols);
  for (let i = 0; i < change.data.length; i += 1) change.data[i] = weightAfter.data[i] - weightBefore.data[i];

  const { rowLabels, columnLabels } = parameterAxisLabels(name, weightBefore, model);
  const cell = selected.row < weightBefore.rows && selected.col < weightBefore.cols ? selected : { row: 0, col: 0 };
  const cellName = `${name}[${rowLabels[cell.row] || cell.row}, ${columnLabels[cell.col]}]`;

  const sharedProps = { rowLabels, columnLabels, selectedCell: cell, onCellClick: setSelected };
  const weightScale = Math.max(...Array.from(weightBefore.data, Math.abs), ...Array.from(weightAfter.data, Math.abs), 1e-9);

  return (
    <div>
      <label className="inline-label">
        Parameter
        <select value={name} onChange={(event) => onParameterChange(event.target.value)}>
          {before.map((parameter) => (
            <option key={parameter.name} value={parameter.name}>
              {parameter.name} [{parameter.matrix.rows}×{parameter.matrix.cols}]
            </option>
          ))}
        </select>
      </label>
      <p className="muted">{description}</p>

      <div className="matrix-row">
        <MatrixHeatmap matrix={weightBefore} title="weight before" scaleLimit={weightScale} {...sharedProps} />
        <MatrixHeatmap
          matrix={gradient}
          title="gradient (∂loss / ∂weight)"
          digits={3}
          caption="Red = increasing this weight increases the loss. Blue = increasing it decreases the loss. White = no effect on this sample."
          {...sharedProps}
        />
        <MatrixHeatmap
          matrix={change}
          title="change = after − before"
          digits={3}
          caption="Always the opposite sign of the gradient: weights move downhill."
          {...sharedProps}
        />
        <MatrixHeatmap matrix={weightAfter} title="weight after" scaleLimit={weightScale} {...sharedProps} />
      </div>

      <CellUpdateExplanation
        report={report}
        parameterIndex={index}
        cellName={cellName}
        flatIndex={cell.row * weightBefore.cols + cell.col}
        before={getValue(weightBefore, cell.row, cell.col)}
        gradient={getValue(gradient, cell.row, cell.col)}
        after={getValue(weightAfter, cell.row, cell.col)}
      />
    </div>
  );
}

interface CellUpdateExplanationProps {
  report: TrainingStepReport;
  parameterIndex: number;
  flatIndex: number;
  cellName: string;
  before: number;
  gradient: number;
  after: number;
}

function CellUpdateExplanation({ report, parameterIndex, flatIndex, cellName, before, gradient, after }: CellUpdateExplanationProps) {
  const optimizer = report.optimizerAfter;
  const learningRate = optimizer.learningRate;
  const direction =
    gradient === 0
      ? "The gradient is 0: this weight did not affect the prediction for this sample, so it does not move."
      : gradient > 0
        ? "The gradient is positive: making this weight bigger would increase the loss, so it is decreased."
        : "The gradient is negative: making this weight bigger would decrease the loss, so it is increased.";

  return (
    <div className="explainer">
      <div className="explainer-title">
        Update of <code>{cellName}</code>
      </div>
      <p className="explainer-text">{direction}</p>
      {optimizer.name === "sgd" ? (
        <div className="formula">
          after = before − learningRate × gradient
          <br />= {formatNumber(before, 5)} − {learningRate} × {formatNumber(gradient, 5)} ={" "}
          <strong>{formatNumber(after, 5)}</strong>
        </div>
      ) : (
        <AdamExplanation
          report={report}
          parameterIndex={parameterIndex}
          flatIndex={flatIndex}
          before={before}
          gradient={gradient}
          after={after}
        />
      )}
    </div>
  );
}

function AdamExplanation({
  report,
  parameterIndex,
  flatIndex,
  before,
  gradient,
  after,
}: Omit<CellUpdateExplanationProps, "cellName">) {
  const optimizer = report.optimizerAfter;
  const step = optimizer.stepCount;
  const m = listParameters(optimizer.firstMoment!)[parameterIndex].matrix.data[flatIndex];
  const v = listParameters(optimizer.secondMoment!)[parameterIndex].matrix.data[flatIndex];
  const mHat = m / (1 - ADAM_BETA1 ** step);
  const vHat = v / (1 - ADAM_BETA2 ** step);
  const update = (optimizer.learningRate * mHat) / (Math.sqrt(vHat) + ADAM_EPSILON);

  return (
    <>
      <p className="explainer-text">
        Adam does not use the raw gradient directly. It keeps a running average of the gradient (m, like momentum) and
        of the squared gradient (v, how large gradients usually are for this weight). Dividing m by √v gives every
        weight a step of roughly learningRate in size, whether its gradients are tiny or huge. This is update number{" "}
        {step}.
        {step === 1 &&
          " On the very first update m̂ ÷ √v̂ is exactly +1 or −1, so every weight moves by exactly ± learningRate. Switch the optimizer to SGD to see changes that are proportional to the gradient."}
      </p>
      <div className="formula">
        m = {ADAM_BETA1} × previous m + {(1 - ADAM_BETA1).toFixed(1)} × gradient({formatNumber(gradient, 5)}) ={" "}
        {formatNumber(m, 6)}
        <br />v = {ADAM_BETA2} × previous v + {(1 - ADAM_BETA2).toFixed(3)} × gradient² = {formatNumber(v, 8)}
        <br />
        m̂ = m ÷ (1 − {ADAM_BETA1}^{step}) = {formatNumber(mHat, 6)}, v̂ = v ÷ (1 − {ADAM_BETA2}^{step}) ={" "}
        {formatNumber(vHat, 8)}
        <br />
        after = before − learningRate × m̂ ÷ (√v̂ + ε)
        <br />= {formatNumber(before, 5)} − {optimizer.learningRate} × {formatNumber(mHat, 6)} ÷{" "}
        {formatNumber(Math.sqrt(vHat), 6)} = {formatNumber(before, 5)} − {formatNumber(update, 5)} ={" "}
        <strong>{formatNumber(after, 5)}</strong>
      </div>
    </>
  );
}
