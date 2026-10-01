import type { Model, PredictionStep } from "@math-llm/core";
import { MatMulView } from "../../components/MatMulView";
import { MatrixHeatmap } from "../../components/MatrixHeatmap";
import { ProbabilityBars } from "../../components/ProbabilityBars";
import { SoftmaxExplainer } from "../../components/SoftmaxExplainer";
import { StageCard } from "../../components/StageCard";
import { displayToken, formatPercent, indexLabels } from "../../utils/format";
import { vocabularyLabels } from "../../utils/labels";

interface OutputStageProps {
  model: Model;
  step: PredictionStep;
  stepOffset: number;
}

export function OutputStage({ model, step, stepOffset }: OutputStageProps) {
  const { trace } = step;
  const { embeddingSize: d } = model.config;
  const vocabulary = vocabularyLabels(model);
  const chosenIndex = model.tokenizer.vocabulary.indexOf(step.chosenToken);

  return (
    <>
      <StageCard
        step={`${stepOffset}`}
        title="Last position → one score per character (logits)"
        explanation={
          <>
            Only the last row is used to predict the next character, because it is the only position that has seen the
            whole input. It is multiplied by <code>outputWeight</code>, which has one column per vocabulary character.
            Each result is a raw score called a logit: higher means &quot;this character is more likely next&quot;.
          </>
        }
        formula={`logits = lastRow · outputWeight + outputBias   [1×${d}] · [${d}×${vocabulary.length}] → [1×${vocabulary.length}]`}
      >
        <div className="matrix-row">
          <MatrixHeatmap
            matrix={trace.finalVector}
            title="last row of block output"
            rowLabels={["last"]}
            columnLabels={indexLabels(d)}
          />
          <MatrixHeatmap
            matrix={model.parameters.outputWeight}
            title="outputWeight (learned)"
            rowLabels={indexLabels(d)}
            columnLabels={vocabulary}
            highlightedColumns={[chosenIndex]}
          />
        </div>
        <MatMulView
          resultName="logit"
          title="logits"
          result={trace.logits}
          rowLabels={["last position"]}
          columnLabels={vocabulary}
          left={trace.finalVector}
          leftName="block output"
          right={model.parameters.outputWeight}
          rightName="outputWeight"
          bias={model.parameters.outputBias}
          biasName="outputBias"
        />
      </StageCard>

      <StageCard
        step={`${stepOffset + 1}`}
        title="Softmax → probabilities → pick the most likely character"
        explanation={
          <>
            Softmax turns the logits into probabilities. The model then simply takes the character with the highest
            probability (greedy decoding). Here it picks <strong>{displayToken(step.chosenToken)}</strong> with{" "}
            {formatPercent(trace.probabilities.data[chosenIndex])}.
            {step.chosenToken === "<END>" && " END means: the answer is complete, stop generating."}
          </>
        }
        formula="probabilities = softmax(logits);   next character = the one with the highest probability"
      >
        <div className="matrix-row">
          <ProbabilityBars
            tokens={model.tokenizer.vocabulary}
            probabilities={trace.probabilities.data}
            logits={trace.logits.data}
            highlightToken={step.chosenToken}
            highlightLabel="chosen"
          />
          <SoftmaxExplainer
            title="Softmax of the logits"
            labels={vocabulary}
            scores={Array.from(trace.logits.data)}
            highlightIndex={chosenIndex}
          />
        </div>
      </StageCard>
    </>
  );
}
