import { useMemo, useState } from "react";
import { listParameters, type Model, type TrainingStepReport } from "@math-llm/core";
import { MatrixHeatmap } from "../../components/MatrixHeatmap";
import { ProbabilityBars } from "../../components/ProbabilityBars";
import { StageCard } from "../../components/StageCard";
import { displayToken, formatNumber, formatPercent } from "../../utils/format";
import { parameterAxisLabels, vocabularyLabels } from "../../utils/labels";
import { WeightUpdateExplorer } from "./WeightUpdateExplorer";

interface TrainingStepReportViewProps {
  model: Model;
  report: TrainingStepReport;
}

interface WeightChange {
  parameterName: string;
  cellLabel: string;
  before: number;
  gradient: number;
  after: number;
}

function largestChanges(model: Model, report: TrainingStepReport, count: number): WeightChange[] {
  const gradients = listParameters(report.gradients);
  const after = listParameters(report.parametersAfter);
  const changes: WeightChange[] = [];

  listParameters(report.parametersBefore).forEach(({ name, matrix }, index) => {
    const { rowLabels, columnLabels } = parameterAxisLabels(name, matrix, model);
    for (let i = 0; i < matrix.data.length; i += 1) {
      const row = Math.floor(i / matrix.cols);
      const col = i % matrix.cols;
      changes.push({
        parameterName: name,
        cellLabel: `[${rowLabels[row] || row}, ${columnLabels[col]}]`,
        before: matrix.data[i],
        gradient: gradients[index].matrix.data[i],
        after: after[index].matrix.data[i],
      });
    }
  });

  return changes
    .sort((a, b) => Math.abs(b.gradient) - Math.abs(a.gradient))
    .slice(0, count);
}

export function TrainingStepReportView({ model, report }: TrainingStepReportViewProps) {
  const [parameterName, setParameterName] = useState("outputWeight");
  const target = report.sample.targetToken;
  const targetId = report.sample.targetTokenId;
  const probabilityBefore = report.traceBefore.probabilities.data[targetId];
  const probabilityAfter = report.traceAfter.probabilities.data[targetId];

  const gradientSizes = useMemo(
    () =>
      listParameters(report.gradients).map(({ name, matrix }) => ({
        name,
        largest: Math.max(...Array.from(matrix.data, Math.abs)),
        nonZero: matrix.data.filter((value) => value !== 0).length,
        total: matrix.data.length,
      })),
    [report],
  );
  const largestGradient = Math.max(...gradientSizes.map(({ largest }) => largest), 1e-12);
  const topChanges = useMemo(() => largestChanges(model, report, 12), [model, report]);

  return (
    <div className="trace">
      <StageCard
        step="A"
        title="Forward pass: what does the model predict right now?"
        explanation={
          <>
            The sample is <code>{report.sample.contextText}</code> and the correct next character is{" "}
            <strong>{displayToken(target)}</strong>. Before learning anything from it, the model gives that character a
            probability of {formatPercent(probabilityBefore)}. (The &quot;Ask&quot; tab shows every matrix of this
            forward pass.)
          </>
        }
      >
        <ProbabilityBars
          tokens={model.tokenizer.vocabulary}
          probabilities={report.traceBefore.probabilities.data}
          highlightToken={target}
          highlightLabel="correct answer"
        />
      </StageCard>

      <StageCard
        step="B"
        title="Loss: one number that measures how wrong the prediction is"
        explanation={
          <>
            Cross-entropy loss only looks at the probability given to the correct character. 100% gives a loss of 0.
            The smaller the probability, the larger the loss. Training means: change the weights to make this number
            smaller.
          </>
        }
        formula={
          <>
            loss = −ln(probability of correct character) = −ln({formatNumber(probabilityBefore, 4)}) ={" "}
            <strong>{formatNumber(report.lossBefore, 4)}</strong>
          </>
        }
      >
        {null}
      </StageCard>

      <StageCard
        step="C"
        title="Error signal at the output"
        explanation={
          <>
            Backpropagation starts here. For softmax with cross-entropy, the gradient of the loss with respect to each
            logit is simply <em>probability − 1</em> for the correct character and <em>probability − 0</em> for every
            other one. A negative value means &quot;this score should go up&quot; (only the correct character), a
            positive value means &quot;this score should go down&quot;. Wrong characters with high probability get
            pushed down the most.
          </>
        }
        formula="logitGradient = probabilities − oneHot(correct character)"
      >
        <MatrixHeatmap
          matrix={report.logitGradient}
          title="∂loss / ∂logits"
          rowLabels={["gradient"]}
          columnLabels={vocabularyLabels(model)}
          highlightedColumns={[targetId]}
          digits={3}
        />
      </StageCard>

      <StageCard
        step="D"
        title="Backpropagation: how much blame each parameter gets"
        explanation={
          <>
            The chain rule carries the error signal backwards through every layer: output → feed-forward → attention →
            embeddings. Each weight receives a gradient: how much the loss would change if that weight were nudged up a
            little. Note how only the rows of <code>tokenEmbedding</code> for characters present in the input get a
            gradient: the other characters were not involved.
          </>
        }
      >
        <table className="data-table compact">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Largest |gradient|</th>
              <th />
              <th>Weights with non-zero gradient</th>
            </tr>
          </thead>
          <tbody>
            {gradientSizes.map(({ name, largest, nonZero, total }) => (
              <tr
                key={name}
                className={`clickable-row ${name === parameterName ? "row-highlight" : ""}`}
                onClick={() => setParameterName(name)}
              >
                <td>
                  <code>{name}</code>
                </td>
                <td>{formatNumber(largest, 5)}</td>
                <td className="bar-cell">
                  <div className="bar-track">
                    <div className="bar-fill gradient" style={{ width: `${(largest / largestGradient) * 100}%` }} />
                  </div>
                </td>
                <td>
                  {nonZero} / {total}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </StageCard>

      <StageCard
        step="E"
        title="Weight update: before, gradient, change, after"
        explanation={
          <>
            Every weight takes a small step against its gradient. Pick a parameter (or click a row in the table above),
            then click any cell to see the exact arithmetic for that one weight.
          </>
        }
      >
        <WeightUpdateExplorer
          key={parameterName}
          model={model}
          report={report}
          parameterName={parameterName}
          onParameterChange={setParameterName}
        />
      </StageCard>

      <StageCard
        step="F"
        title="The 12 weights with the largest gradient"
        explanation="These are the weights that matter most for this sample: changing them moves the loss the most."
      >
        <table className="data-table compact">
          <thead>
            <tr>
              <th>Weight</th>
              <th>Before</th>
              <th>Gradient</th>
              <th>After</th>
              <th>Direction</th>
            </tr>
          </thead>
          <tbody>
            {topChanges.map((change) => (
              <tr
                key={`${change.parameterName}${change.cellLabel}`}
                className="clickable-row"
                onClick={() => setParameterName(change.parameterName)}
              >
                <td>
                  <code>
                    {change.parameterName}
                    {change.cellLabel}
                  </code>
                </td>
                <td>{formatNumber(change.before, 5)}</td>
                <td className={change.gradient > 0 ? "positive" : "negative"}>{formatNumber(change.gradient, 5)}</td>
                <td>{formatNumber(change.after, 5)}</td>
                <td>{change.after > change.before ? "increased ↑" : "decreased ↓"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </StageCard>

      <StageCard
        step="G"
        title="Result: the same input, after the update"
        explanation={
          <>
            Running the same sample again with the new weights: the probability of{" "}
            <strong>{displayToken(target)}</strong> went from {formatPercent(probabilityBefore)} to{" "}
            {formatPercent(probabilityAfter)}, and the loss from {formatNumber(report.lossBefore, 4)} to{" "}
            {formatNumber(report.lossAfter, 4)}. One step only nudges the weights a little. Training repeats this
            thousands of times over all samples.
          </>
        }
      >
        <ProbabilityBars
          tokens={model.tokenizer.vocabulary}
          probabilities={report.traceAfter.probabilities.data}
          highlightToken={target}
          highlightLabel="correct answer"
        />
      </StageCard>
    </div>
  );
}
