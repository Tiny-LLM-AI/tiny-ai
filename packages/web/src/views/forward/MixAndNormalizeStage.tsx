import { useState } from "react";
import type { BlockParameters, BlockTrace, Model } from "@math-llm/core";
import { LayerNormExplainer } from "../../components/LayerNormExplainer";
import { MatMulView } from "../../components/MatMulView";
import { MatrixHeatmap } from "../../components/MatrixHeatmap";
import { StageCard } from "../../components/StageCard";
import { indexLabels } from "../../utils/format";

interface MixAndNormalizeStageProps {
  model: Model;
  block: BlockTrace;
  parameters: BlockParameters;
  rowLabels: string[];
  stepOffset: number;
}

export function MixAndNormalizeStage({ model, block, parameters, rowLabels, stepOffset }: MixAndNormalizeStageProps) {
  const [normRow, setNormRow] = useState(rowLabels.length - 1);
  const dimensionLabels = indexLabels(model.config.embeddingSize);
  const { embeddingSize: d } = model.config;

  return (
    <>
      <StageCard
        step={`${stepOffset}`}
        title="Join the heads and mix them"
        explanation={
          <>
            The head outputs are placed side by side to get back {d} columns. Then one more learned matrix mixes them,
            so information found by different heads can be combined.
          </>
        }
        formula={`attentionOutput = concat(heads) · attentionOutputWeight + attentionOutputBias   [${rowLabels.length}×${d}] · [${d}×${d}]`}
      >
        <div className="matrix-row">
          <MatrixHeatmap
            matrix={block.concatenatedHeads}
            title="concatenated heads"
            rowLabels={rowLabels}
            columnLabels={dimensionLabels}
          />
          <MatMulView
            resultName="attentionOutput"
            title="attentionOutput"
            result={block.attentionOutput}
            rowLabels={rowLabels}
            columnLabels={dimensionLabels}
            left={block.concatenatedHeads}
            leftName="concatenated heads"
            right={parameters.attentionOutputWeight}
            rightName="attentionOutputWeight"
            bias={parameters.attentionOutputBias}
            biasName="attentionOutputBias"
          />
        </div>
      </StageCard>

      <StageCard
        step={`${stepOffset + 1}`}
        title="Residual connection + layer norm"
        explanation={
          <>
            The attention result is added on top of the original X instead of replacing it (a &quot;residual&quot; or
            skip connection). The model keeps what it already knew and only adds a correction. Layer norm then rescales
            each row. Click a row to see the layer norm numbers.
          </>
        }
        formula="afterAttention = layerNorm(X + attentionOutput)"
      >
        <div className="matrix-row">
          <MatrixHeatmap matrix={block.input} title="X (block input)" rowLabels={rowLabels} columnLabels={dimensionLabels} />
          <MatrixHeatmap
            matrix={block.attentionResidual}
            title="X + attentionOutput"
            rowLabels={rowLabels}
            columnLabels={dimensionLabels}
            highlightedRows={[normRow]}
            onCellClick={({ row }) => setNormRow(row)}
          />
          <MatrixHeatmap
            matrix={block.attentionNorm.output}
            title="afterAttention (after layer norm)"
            rowLabels={rowLabels}
            columnLabels={dimensionLabels}
            highlightedRows={[normRow]}
            onCellClick={({ row }) => setNormRow(row)}
          />
        </div>
        <LayerNormExplainer
          input={block.attentionResidual}
          result={block.attentionNorm}
          scale={parameters.attentionNormScale}
          shift={parameters.attentionNormShift}
          row={normRow}
          rowLabel={rowLabels[normRow]}
        />
      </StageCard>
    </>
  );
}
