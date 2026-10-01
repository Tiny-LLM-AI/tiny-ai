import { useState } from "react";
import type { OptimizerName } from "@math-llm/core";
import { LineChart } from "../../components/LineChart";
import type { TrainingSessionController } from "../../state/useTrainingSession";
import { displayToken, formatNumber, formatPercent } from "../../utils/format";
import { TrainingStepReportView } from "./TrainingStepReportView";

interface TrainingViewProps {
  controller: TrainingSessionController;
}

export function TrainingView({ controller }: TrainingViewProps) {
  const { session, isRunning, lastReport } = controller;
  const { trainingConfig } = session;
  const [epochsToRun, setEpochsToRun] = useState(trainingConfig.epochs);
  const [sampleIndex, setSampleIndex] = useState(0);

  const lossHistory = session.history.map(({ averageLoss }) => averageLoss);
  const accuracyHistory = session.history.map(({ accuracy }) => accuracy);

  return (
    <div className="view">
      <section className="panel">
        <h2>How the model learns</h2>
        <p>
          Training repeats four steps: <strong>forward pass</strong> (make a prediction), <strong>loss</strong> (measure
          how wrong it is), <strong>backward pass</strong> (compute, for every weight, which direction reduces the loss)
          and <strong>update</strong> (move every weight a small step in that direction). A batch averages the gradients
          of several samples before updating. An epoch is one pass over all {session.samples.length} samples.
        </p>

        <div className="controls">
          <label>
            Optimizer
            <select
              value={trainingConfig.optimizer}
              disabled={isRunning}
              onChange={(event) => controller.updateTrainingConfig({ optimizer: event.target.value as OptimizerName })}
            >
              <option value="adam">Adam (fast, standard for Transformers)</option>
              <option value="sgd">SGD (simplest: weight − learningRate × gradient)</option>
            </select>
          </label>
          <label>
            Learning rate
            <input
              type="number"
              step={0.001}
              min={0.0001}
              value={trainingConfig.learningRate}
              disabled={isRunning}
              onChange={(event) => controller.updateTrainingConfig({ learningRate: Number(event.target.value) })}
            />
          </label>
          <label>
            Batch size
            <input
              type="number"
              min={1}
              max={session.samples.length}
              value={trainingConfig.batchSize}
              disabled={isRunning}
              onChange={(event) => controller.updateTrainingConfig({ batchSize: Math.max(1, Number(event.target.value)) })}
            />
          </label>
          <label>
            Epochs to run
            <input
              type="number"
              min={1}
              value={epochsToRun}
              disabled={isRunning}
              onChange={(event) => setEpochsToRun(Math.max(1, Number(event.target.value)))}
            />
          </label>
          {isRunning ? (
            <button type="button" onClick={controller.stopTraining}>
              Stop
            </button>
          ) : (
            <button type="button" onClick={() => void controller.trainEpochs(epochsToRun)}>
              Train {epochsToRun} epochs
            </button>
          )}
          <button
            type="button"
            className="secondary"
            disabled={isRunning}
            onClick={() => controller.resetModel(trainingConfig)}
          >
            Reset to random weights
          </button>
        </div>

        <p className="muted">
          Epochs trained: {session.epoch} · weight updates: {session.optimizer.stepCount}
          {session.history.length > 0 &&
            ` · last loss ${formatNumber(lossHistory.at(-1)!, 4)} · accuracy ${formatPercent(accuracyHistory.at(-1)!)}`}
        </p>

        <div className="chart-row">
          <LineChart
            title="Average loss per epoch (lower is better)"
            values={lossHistory}
            formatValue={(value) => value.toFixed(3)}
            color="#dc2626"
          />
          <LineChart
            title="Questions answered exactly right"
            values={accuracyHistory}
            maxValue={1}
            formatValue={(value) => formatPercent(value, 0)}
            color="#16a34a"
          />
        </div>
      </section>

      <section className="panel">
        <h2>Train on one sample and see exactly what changes</h2>
        <p>
          Pick one training sample and apply a single update with only that sample. Every number involved is shown
          below. Try it on an untrained model first, then again after training.
        </p>
        <div className="controls">
          <label>
            Sample
            <select value={sampleIndex} disabled={isRunning} onChange={(event) => setSampleIndex(Number(event.target.value))}>
              {session.samples.map((sample, index) => (
                <option key={index} value={index}>
                  &quot;{sample.contextText}&quot; → {displayToken(sample.targetToken)}
                </option>
              ))}
            </select>
          </label>
          <button type="button" disabled={isRunning} onClick={() => controller.trainOneSample(session.samples[sampleIndex])}>
            Train on this sample
          </button>
          <button
            type="button"
            className="secondary"
            disabled={isRunning}
            onClick={() => {
              const index = Math.floor(Math.random() * session.samples.length);
              setSampleIndex(index);
              controller.trainOneSample(session.samples[index]);
            }}
          >
            Train on a random sample
          </button>
        </div>
      </section>

      {lastReport && <TrainingStepReportView model={session.model} report={lastReport} />}
    </div>
  );
}
