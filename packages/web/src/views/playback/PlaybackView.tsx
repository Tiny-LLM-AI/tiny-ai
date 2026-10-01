import { useMemo, useState } from "react";
import { generateAnswer, type TrainingSession } from "@math-llm/core";
import { buildForwardPlayback } from "../../playback/build-forward-playback";
import { displayToken, formatPercent } from "../../utils/format";
import { PlaybackPlayer } from "./PlaybackPlayer";

interface PlaybackViewProps {
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

export function PlaybackView({ session, revision }: PlaybackViewProps) {
  const [question, setQuestion] = useState("2+3=");
  const [selectedStep, setSelectedStep] = useState(0);

  const { result, error } = useMemo(() => generateSafely(session, question), [session, question, revision]);
  const step = result?.steps[Math.min(selectedStep, result.steps.length - 1)];
  const stages = useMemo(() => (step ? buildForwardPlayback(session.model, step) : []), [session, step]);

  const choose = (value: string) => {
    setQuestion(value);
    setSelectedStep(0);
  };

  return (
    <div className="view">
      <section className="panel">
        <h2>Watch the model compute, one number at a time</h2>
        <p>
          Press <strong>Play</strong> and follow the input from text to ids, to vectors, through every matrix
          multiplication, until one character is chosen. Slow it down to read every multiplication, speed it up to see
          the big picture. Keys: <kbd>space</kbd> play/pause, <kbd>←</kbd> <kbd>→</kbd> step.{" "}
          {session.epoch === 0 && <strong>The model is untrained, so the numbers are random. Train it first for a meaningful answer.</strong>}
        </p>
        <div className="controls">
          <label>
            Question
            <input type="text" value={question} onChange={(event) => choose(event.target.value)} />
          </label>
          <div className="quick-picks">
            {["2+3=", "3*3=", "1-3=", "6/2="].map((option) => (
              <button key={option} type="button" className="secondary" onClick={() => choose(option)}>
                {option}
              </button>
            ))}
          </div>
        </div>
        {error && <p className="error">{error}</p>}
        {result && (
          <div className="generation-steps">
            {result.steps.map((generationStep, index) => (
              <button
                key={index}
                type="button"
                className={`generation-step ${index === selectedStep ? "active" : ""}`}
                onClick={() => setSelectedStep(index)}
              >
                <span className="muted">Character {index + 1}</span>
                <code>{generationStep.contextText}</code>
                <span>
                  → <strong>{displayToken(generationStep.chosenToken)}</strong> (
                  {formatPercent(generationStep.ranking[0].probability)})
                </span>
              </button>
            ))}
          </div>
        )}
      </section>
      {step && <PlaybackPlayer key={`${question}-${selectedStep}-${revision}`} stages={stages} />}
    </div>
  );
}
