import type { Model } from "@math-llm/core";
import { displayToken } from "../utils/format";

interface TokenStripProps {
  model: Model;
  tokenIds: number[];
}

/** Shows each position of the context window as a box: position, character and token id. */
export function TokenStrip({ model, tokenIds }: TokenStripProps) {
  return (
    <div className="token-strip">
      {tokenIds.map((id, position) => {
        const token = model.tokenizer.vocabulary[id];
        return (
          <div key={position} className={`token-box ${id === 0 ? "padding" : ""}`}>
            <span className="token-position">pos {position}</span>
            <span className="token-character">{displayToken(token)}</span>
            <span className="token-id">id {id}</span>
          </div>
        );
      })}
    </div>
  );
}
