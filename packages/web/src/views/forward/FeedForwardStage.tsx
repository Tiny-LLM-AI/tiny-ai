import type { BlockParameters, BlockTrace, Model } from "@math-llm/core";
import { MatMulView } from "../../components/MatMulView";
import { MatrixHeatmap } from "../../components/MatrixHeatmap";
import { StageCard } from "../../components/StageCard";
import { indexLabels } from "../../utils/format";

interface FeedForwardStageProps {
  model: Model;
  block: BlockTrace;
  parameters: BlockParameters;
  rowLabels: string[];
  stepOffset: number;
}

export function FeedForwardStage({ model, block, parameters, rowLabels, stepOffset }: FeedForwardStageProps) {
  const { embeddingSize: d, feedForwardSize: f } = model.config;
  const dimensionLabels = indexLabels(d);
  const hiddenLabels = indexLabels(f);
  const zeroedCount = block.feedForwardHidden.data.filter((value) => value <= 0).length;

  return (
    <>
      <StageCard
        step={`${stepOffset}`}
        title="Feed-forward network, layer 1 + ReLU"
        explanation={
          <>
            Attention moved information between positions. The feed-forward network (MLP) now processes each position
            on its own. It widens every row from {d} to {f} numbers, then ReLU replaces every negative number with 0.
            ReLU is what makes the network non-linear: without it, all layers together would collapse into a single
            matrix multiplication. In this input, ReLU zeroed {zeroedCount} of {block.feedForwardHidden.data.length}{" "}
            numbers.
          </>
        }
        formula={`hidden = afterAttention · feedForwardInputWeight + feedForwardInputBias   [${rowLabels.length}×${d}] · [${d}×${f}] → [${rowLabels.length}×${f}];   activated = max(0, hidden)`}
      >
        <MatMulView
          resultName="hidden"
          title="hidden (before ReLU)"
          result={block.feedForwardHidden}
          rowLabels={rowLabels}
          columnLabels={hiddenLabels}
          left={block.attentionNorm.output}
          leftName="afterAttention"
          right={parameters.feedForwardInputWeight}
          rightName="feedForwardInputWeight"
          bias={parameters.feedForwardInputBias}
          biasName="feedForwardInputBias"
        />
        <MatrixHeatmap
          matrix={block.feedForwardActivated}
          title="activated = ReLU(hidden): negatives become 0"
          rowLabels={rowLabels}
          columnLabels={hiddenLabels}
        />
      </StageCard>

      <StageCard
        step={`${stepOffset + 1}`}
        title="Feed-forward network, layer 2 + residual + layer norm"
        explanation={
          <>
            A second matrix shrinks each row back to {d} numbers. Like after attention, the result is added to its
            input (residual) and layer-normalized. This is the output of the Transformer block.
          </>
        }
        formula={`blockOutput = layerNorm(afterAttention + activated · feedForwardOutputWeight + feedForwardOutputBias)   [${rowLabels.length}×${f}] · [${f}×${d}]`}
      >
        <MatMulView
          resultName="feedForwardOutput"
          title="feedForwardOutput"
          result={block.feedForwardOutput}
          rowLabels={rowLabels}
          columnLabels={dimensionLabels}
          left={block.feedForwardActivated}
          leftName="activated"
          right={parameters.feedForwardOutputWeight}
          rightName="feedForwardOutputWeight"
          bias={parameters.feedForwardOutputBias}
          biasName="feedForwardOutputBias"
        />
        <div className="matrix-row">
          <MatrixHeatmap
            matrix={block.feedForwardResidual}
            title="afterAttention + feedForwardOutput"
            rowLabels={rowLabels}
            columnLabels={dimensionLabels}
          />
          <MatrixHeatmap
            matrix={block.output}
            title="block output (after layer norm)"
            rowLabels={rowLabels}
            columnLabels={dimensionLabels}
            highlightedRows={[rowLabels.length - 1]}
          />
        </div>
      </StageCard>
    </>
  );
}
