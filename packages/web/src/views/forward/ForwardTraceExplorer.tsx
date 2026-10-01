import { useState } from "react";
import type { Model, PredictionStep } from "@math-llm/core";
import { contextRowLabels } from "../../utils/labels";
import { AttentionStage } from "./AttentionStage";
import { EmbeddingStage } from "./EmbeddingStage";
import { FeedForwardStage } from "./FeedForwardStage";
import { MixAndNormalizeStage } from "./MixAndNormalizeStage";
import { OutputStage } from "./OutputStage";

interface ForwardTraceExplorerProps {
  model: Model;
  step: PredictionStep;
}

/** Every stage of one forward pass, from characters to the chosen next character. */
export function ForwardTraceExplorer({ model, step }: ForwardTraceExplorerProps) {
  const [blockIndex, setBlockIndex] = useState(0);
  const rowLabels = contextRowLabels(model, step.contextTokenIds);
  const block = step.trace.blocks[Math.min(blockIndex, step.trace.blocks.length - 1)];
  const parameters = model.parameters.blocks[Math.min(blockIndex, model.parameters.blocks.length - 1)];

  return (
    <div className="trace">
      <EmbeddingStage model={model} trace={step.trace} />

      {step.trace.blocks.length > 1 && (
        <div className="segmented">
          {step.trace.blocks.map((_, index) => (
            <button
              key={index}
              type="button"
              className={index === blockIndex ? "active" : ""}
              onClick={() => setBlockIndex(index)}
            >
              Transformer block {index + 1}
            </button>
          ))}
        </div>
      )}

      <div className="block-frame">
        <div className="block-frame-title">
          Transformer block {blockIndex + 1} of {step.trace.blocks.length}
        </div>
        <AttentionStage model={model} block={block} parameters={parameters} rowLabels={rowLabels} stepOffset={4} />
        <MixAndNormalizeStage model={model} block={block} parameters={parameters} rowLabels={rowLabels} stepOffset={8} />
        <FeedForwardStage model={model} block={block} parameters={parameters} rowLabels={rowLabels} stepOffset={10} />
      </div>

      <OutputStage model={model} step={step} stepOffset={12} />
    </div>
  );
}
