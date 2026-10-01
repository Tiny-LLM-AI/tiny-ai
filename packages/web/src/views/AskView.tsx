import { useMemo, useState } from "react";
import { generateAnswer, type TrainingSession } from "@math-llm/core";
import { displayToken, formatPercent } from "../utils/format";
import { ForwardTraceExplorer } from "./forward/ForwardTraceExplorer";

interface AskViewProps {
  session: TrainingSession;
  revision: number;
}

function generateSafely(session: TrainingSession, question: string) {
  try {
    return { result: generateAnswer(session.model, question), error: null };
  } catch (error) {
    return { result: null, error: error instanceof Error ? error.message : String(error) };
  }
}

export function AskView({ session, revision }: AskViewProps) {
  const [question, setQuestion] = useState("2+3=");
  const [selectedStep, setSelectedStep] = useState(0);

  // `revision` changes whenever training updates the weights in place.
  const { result, error } = useMemo(() => generateSafely(session, question), [session, question, revision]);
  const step = result?.steps[Math.min(selectedStep, result.steps.length - 1)];
  const expected = session.examples.find((example) => example.question === question)?.answer;

  return (
    <div className="view">
      <section className="panel">
        <h2>Ask the model and watch every calculation</h2>
        <p>
          The model writes its answer one character at a time. Each character requires one full pass through the
          network (a &quot;forward pass&quot;). Pick a step below to inspect its forward pass.{" "}
          {session.epoch === 0 && (
            <strong>The model is untrained, so its answers are random. Train it in step 4, then come back.</strong>
          )}
        </p>
        <div className="controls">
          <label>
            Question
            <input
              type="text"
              value={question}
              onChange={(event) => {
                setQuestion(event.target.value);
                setSelectedStep(0);
              }}
            />
          </label>
          <div className="quick-picks">
            {["2+3=", "3*3=", "1-3=", "6/2=", "0*2="].map((option) => (
              <button
                key={option}
                type="button"
                className="secondary"
                onClick={() => {
                  setQuestion(option);
                  setSelectedStep(0);
                }}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        {error && <p className="error">{error}</p>}

        {result && (
          <>
            <div className="answer-line">
              <code>
                {result.question}
                <span className="generated">{result.answer}</span>
              </code>
              {expected !== undefined && (
                <span className={result.answer === expected ? "badge good" : "badge bad"}>
                  {result.answer === expected ? "correct" : `wrong, expected ${expected}`}
                </span>
              )}
            </div>
            <div className="generation-steps">
              {result.steps.map((generationStep, index) => (
                <button
                  key={index}
                  type="button"
                  className={`generation-step ${index === selectedStep ? "active" : ""}`}
                  onClick={() => setSelectedStep(index)}
                >
                  <span className="muted">Step {index + 1}</span>
                  <code>{generationStep.contextText}</code>
                  <span>
                    → <strong>{displayToken(generationStep.chosenToken)}</strong> (
                    {formatPercent(generationStep.ranking[0].probability)})
                  </span>
                </button>
              ))}
            </div>
          </>
        )}
      </section>

      {step && <ForwardTraceExplorer key={`${question}-${selectedStep}`} model={session.model} step={step} />}
    </div>
  );
}
