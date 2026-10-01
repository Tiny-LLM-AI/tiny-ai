import { countParameters, listParameters, type TrainingSession } from "@math-llm/core";
import { displayToken } from "../utils/format";

interface DataViewProps {
  session: TrainingSession;
}

export function DataView({ session }: DataViewProps) {
  const { model, examples, samples } = session;
  const exampleSamples = samples.filter((sample) => sample.contextText.startsWith("2-3="));

  return (
    <div className="view">
      <section className="panel">
        <h2>What this model is</h2>
        <p>
          A Transformer, the same kind of neural network as GPT, shrunk to {countParameters(model.parameters)} learnable
          numbers. It never sees a rule like &quot;if the operator is + then add&quot;. It only sees text such as{" "}
          <code>2+3=5</code> and learns to predict the next character. Everything it &quot;knows&quot; about arithmetic
          ends up stored in the numbers of its weight matrices.
        </p>
        <div className="pipeline">
          {[
            "characters",
            "token ids",
            "embedding vectors",
            "attention",
            "feed-forward",
            "logits",
            "probabilities",
            "next character",
          ].map((label, index, all) => (
            <span key={label} className="pipeline-item">
              {label}
              {index < all.length - 1 && <span className="pipeline-arrow">→</span>}
            </span>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Vocabulary: every character the model knows</h2>
        <p>
          Each character gets an id. PAD fills empty positions on the left, END means &quot;the answer is
          finished&quot;. The model can only ever output one of these {model.tokenizer.vocabulary.length} tokens.
        </p>
        <div className="token-strip">
          {model.tokenizer.vocabulary.map((token, id) => (
            <div key={token} className="token-box">
              <span className="token-character">{displayToken(token)}</span>
              <span className="token-id">id {id}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>From one example to training samples</h2>
        <p>
          The model always predicts one character at a time, so each example becomes one training sample per answer
          character, plus one sample that teaches it to stop. Here is <code>2-3=-1</code>:
        </p>
        <table className="data-table">
          <thead>
            <tr>
              <th>Input the model sees</th>
              <th>Context window ({model.config.contextLength} token ids)</th>
              <th>Correct next character</th>
            </tr>
          </thead>
          <tbody>
            {exampleSamples.map((sample) => (
              <tr key={sample.contextText}>
                <td>
                  <code>{sample.contextText}</code>
                </td>
                <td>
                  <code>[{sample.contextTokenIds.join(", ")}]</code>
                </td>
                <td>
                  <code>{displayToken(sample.targetToken)}</code> (id {sample.targetTokenId})
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted">
          {examples.length} examples produce {samples.length} training samples in total.
        </p>
      </section>

      <section className="panel">
        <h2>All learnable parameters</h2>
        <p>These matrices start as small random numbers. Training changes them; nothing else in the model is learned.</p>
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Shape</th>
              <th>Numbers</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {listParameters(model.parameters).map(({ name, matrix, description }) => (
              <tr key={name}>
                <td>
                  <code>{name}</code>
                </td>
                <td>
                  {matrix.rows} × {matrix.cols}
                </td>
                <td>{matrix.data.length}</td>
                <td>{description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>Dataset</h2>
        <div className="example-grid">
          {examples.map(({ question, answer }) => (
            <code key={question}>
              {question}
              {answer}
            </code>
          ))}
        </div>
      </section>
    </div>
  );
}
