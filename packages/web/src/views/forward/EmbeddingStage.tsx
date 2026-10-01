import { useState } from "react";
import { getValue, type ForwardTrace, type Model } from "@math-llm/core";
import type { CellPosition } from "../../components/MatrixHeatmap";
import { MatrixHeatmap } from "../../components/MatrixHeatmap";
import { StageCard } from "../../components/StageCard";
import { TokenStrip } from "../../components/TokenStrip";
import { formatNumber, indexLabels } from "../../utils/format";
import { contextRowLabels, vocabularyLabels } from "../../utils/labels";

interface EmbeddingStageProps {
  model: Model;
  trace: ForwardTrace;
}

export function EmbeddingStage({ model, trace }: EmbeddingStageProps) {
  const [selected, setSelected] = useState<CellPosition>({ row: trace.tokenIds.length - 1, col: 0 });
  const rowLabels = contextRowLabels(model, trace.tokenIds);
  const dimensionLabels = indexLabels(model.config.embeddingSize);
  const usedTokenIds = [...new Set(trace.tokenIds)];

  const tokenValue = getValue(trace.tokenVectors, selected.row, selected.col);
  const positionValue = getValue(trace.positionVectors, selected.row, selected.col);

  return (
    <>
      <StageCard
        step="1"
        title="Text → token ids"
        explanation={
          <>
            The model cannot read characters, only numbers. Each character is replaced by its id in the vocabulary. The
            context window always has {model.config.contextLength} positions, so shorter text is padded on the left
            with PAD (id 0).
          </>
        }
      >
        <TokenStrip model={model} tokenIds={trace.tokenIds} />
      </StageCard>

      <StageCard
        step="2"
        title="Token ids → vectors (embedding lookup)"
        explanation={
          <>
            <code>tokenEmbedding</code> is a learned table with one row of {model.config.embeddingSize} numbers per
            vocabulary character. For every position we copy the row of that character. No multiplication happens here,
            it is just a lookup. Highlighted rows are the characters used in this input.
          </>
        }
        formula="tokenVectors[position] = tokenEmbedding[tokenId of that position]"
      >
        <div className="matrix-row">
          <MatrixHeatmap
            matrix={model.parameters.tokenEmbedding}
            title="tokenEmbedding (learned)"
            rowLabels={vocabularyLabels(model)}
            columnLabels={dimensionLabels}
            highlightedRows={usedTokenIds}
          />
          <MatrixHeatmap
            matrix={trace.tokenVectors}
            title="tokenVectors"
            rowLabels={rowLabels}
            columnLabels={dimensionLabels}
          />
        </div>
      </StageCard>

      <StageCard
        step="3"
        title="Add position information"
        explanation={
          <>
            The token vectors alone do not say where a character is: &quot;2-3&quot; and &quot;3-2&quot; would look the
            same. <code>positionEmbedding</code> is another learned table with one row per position. Adding it gives
            every position its own fingerprint.
          </>
        }
        formula="X = tokenVectors + positionEmbedding   (element by element)"
      >
        <div className="matrix-row">
          <MatrixHeatmap
            matrix={trace.positionVectors}
            title="positionEmbedding (learned)"
            rowLabels={indexLabels(model.config.contextLength).map((i) => `pos ${i}`)}
            columnLabels={dimensionLabels}
            selectedCell={selected}
            onCellClick={setSelected}
          />
          <MatrixHeatmap
            matrix={trace.embeddingSum}
            title="X = input to block 1"
            rowLabels={rowLabels}
            columnLabels={dimensionLabels}
            selectedCell={selected}
            onCellClick={setSelected}
          />
        </div>
        <div className="explainer">
          <div className="explainer-title">
            X[{rowLabels[selected.row]}, {selected.col}]
          </div>
          <div className="formula">
            {formatNumber(tokenValue, 4)} (token) + {formatNumber(positionValue, 4)} (position) ={" "}
            <strong>{formatNumber(tokenValue + positionValue, 4)}</strong>
          </div>
        </div>
      </StageCard>
    </>
  );
}
