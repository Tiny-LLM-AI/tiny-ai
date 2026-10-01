import { useState } from "react";
import { getRow, type BlockParameters, type BlockTrace, type Model } from "@math-llm/core";
import { MatMulView } from "../../components/MatMulView";
import { SoftmaxExplainer } from "../../components/SoftmaxExplainer";
import { StageCard } from "../../components/StageCard";
import { MatrixHeatmap } from "../../components/MatrixHeatmap";
import { indexLabels } from "../../utils/format";

interface AttentionStageProps {
  model: Model;
  block: BlockTrace;
  parameters: BlockParameters;
  rowLabels: string[];
  stepOffset: number;
}

type Projection = "query" | "key" | "value";

const PROJECTION_TEXT: Record<Projection, { letter: string; meaning: string }> = {
  query: { letter: "Q", meaning: "what each position is looking for" },
  key: { letter: "K", meaning: "what each position offers to others" },
  value: { letter: "V", meaning: "the information each position hands over when it is attended to" },
};

export function AttentionStage({ model, block, parameters, rowLabels, stepOffset }: AttentionStageProps) {
  const [projection, setProjection] = useState<Projection>("query");
  const [headIndex, setHeadIndex] = useState(0);
  const [softmaxRow, setSoftmaxRow] = useState(rowLabels.length - 1);

  const headSize = model.config.embeddingSize / model.config.headCount;
  const head = block.heads[headIndex];
  const headStart = headIndex * headSize;
  const headLabels = indexLabels(headSize).map((i) => String(headStart + Number(i)));
  const dimensionLabels = indexLabels(model.config.embeddingSize);
  const positionLabels = rowLabels;
  const { letter, meaning } = PROJECTION_TEXT[projection];

  return (
    <>
      <StageCard
        step={`${stepOffset}`}
        title="Queries, keys and values"
        explanation={
          <>
            Attention lets each position collect information from earlier positions. To do that, X is multiplied by
            three learned matrices. {letter} = {meaning}. All three are ordinary matrix multiplications.
          </>
        }
        formula={`${letter} = X · W${letter.toLowerCase()} + b${letter.toLowerCase()}   [${rowLabels.length}×${model.config.embeddingSize}] · [${model.config.embeddingSize}×${model.config.embeddingSize}] → [${rowLabels.length}×${model.config.embeddingSize}]`}
      >
        <div className="segmented">
          {(Object.keys(PROJECTION_TEXT) as Projection[]).map((option) => (
            <button
              key={option}
              type="button"
              className={option === projection ? "active" : ""}
              onClick={() => setProjection(option)}
            >
              {PROJECTION_TEXT[option].letter}
            </button>
          ))}
        </div>
        <div className="matrix-row">
          <MatrixHeatmap
            matrix={parameters[`${projection}Weight`]}
            title={`${projection}Weight (learned)`}
            rowLabels={dimensionLabels}
            columnLabels={dimensionLabels}
          />
          <MatMulView
            resultName={letter}
            title={letter}
            result={block[projection]}
            rowLabels={positionLabels}
            columnLabels={dimensionLabels}
            left={block.input}
            leftName="X"
            right={parameters[`${projection}Weight`]}
            rightName={`${projection}Weight`}
            bias={parameters[`${projection}Bias`]}
            biasName={`${projection}Bias`}
          />
        </div>
      </StageCard>

      <StageCard
        step={`${stepOffset + 1}`}
        title="Attention scores: who should look at whom"
        explanation={
          <>
            The {model.config.embeddingSize} columns are split into {model.config.headCount} heads of {headSize}{" "}
            columns each. Every head works independently, so it can learn a different pattern (for example one head
            looks at the operator, another at the digits). Score[i, j] is the dot product of the query of position i
            with the key of position j: a large score means &quot;position i finds position j relevant&quot;. Scores are
            divided by √{headSize} to keep them in a reasonable range. Positions after i are masked with −∞.
          </>
        }
        formula={`scores = Q_head · K_headᵀ ÷ √${headSize}   [${rowLabels.length}×${headSize}] · [${headSize}×${rowLabels.length}] → [${rowLabels.length}×${rowLabels.length}]`}
      >
        <div className="segmented">
          {block.heads.map((_, index) => (
            <button
              key={index}
              type="button"
              className={index === headIndex ? "active" : ""}
              onClick={() => setHeadIndex(index)}
            >
              Head {index + 1} (columns {index * headSize}–{(index + 1) * headSize - 1})
            </button>
          ))}
        </div>
        <div className="matrix-row">
          <MatrixHeatmap matrix={head.query} title="Q for this head" rowLabels={positionLabels} columnLabels={headLabels} />
          <MatrixHeatmap matrix={head.key} title="K for this head" rowLabels={positionLabels} columnLabels={headLabels} />
          <MatMulView
            key={`scores-${headIndex}`}
            resultName="score"
            title="masked scores (row = looking position, column = looked-at position)"
            result={head.maskedScores}
            rowLabels={positionLabels}
            columnLabels={positionLabels}
            left={head.query}
            leftName="Q"
            right={head.key}
            rightName="K"
            rightIsTransposed
            scaleFactor={1 / Math.sqrt(headSize)}
            scaleDescription={`divide by √${headSize}`}
            termLabels={headLabels}
            isMasked={({ row, col }) => col > row}
          />
        </div>
      </StageCard>

      <StageCard
        step={`${stepOffset + 2}`}
        title="Softmax: scores → attention weights"
        explanation={
          <>
            Each row of scores is turned into percentages that add up to 100%. Row i now says how much position i
            listens to each earlier position. Click a row to see the softmax calculation.
          </>
        }
        formula="attentionWeights[i] = softmax(scores[i])"
      >
        <div className="matrix-row">
          <MatrixHeatmap
            matrix={head.attentionWeights}
            title="attention weights"
            rowLabels={positionLabels}
            columnLabels={positionLabels}
            colorMode="probability"
            highlightedRows={[softmaxRow]}
            onCellClick={({ row }) => setSoftmaxRow(row)}
          />
          <SoftmaxExplainer
            title={`Softmax of row ${positionLabels[softmaxRow]}`}
            labels={positionLabels}
            scores={getRow(head.maskedScores, softmaxRow)}
          />
        </div>
      </StageCard>

      <StageCard
        step={`${stepOffset + 3}`}
        title="Weighted average of the values"
        explanation={
          <>
            Each position now builds its output as a mix of the value vectors, using the attention weights as the
            mixing ratios. If position 3 gives 80% weight to position 1, its output is mostly position 1&apos;s value.
            This is how information moves between characters.
          </>
        }
        formula={`headOutput = attentionWeights · V_head   [${rowLabels.length}×${rowLabels.length}] · [${rowLabels.length}×${headSize}] → [${rowLabels.length}×${headSize}]`}
      >
        <div className="matrix-row">
          <MatrixHeatmap matrix={head.value} title="V for this head" rowLabels={positionLabels} columnLabels={headLabels} />
          <MatMulView
            key={`output-${headIndex}`}
            resultName="headOutput"
            title="head output"
            result={head.output}
            rowLabels={positionLabels}
            columnLabels={headLabels}
            left={head.attentionWeights}
            leftName="attention weights"
            right={head.value}
            rightName="V"
            termLabels={positionLabels}
          />
        </div>
      </StageCard>
    </>
  );
}
