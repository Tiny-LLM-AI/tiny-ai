import type { ReactNode } from "react";

interface StageCardProps {
  step: string;
  title: string;
  formula?: ReactNode;
  children: ReactNode;
  explanation: ReactNode;
}

/** One numbered stage of the computation: what happens, the formula, and the matrices involved. */
export function StageCard({ step, title, formula, explanation, children }: StageCardProps) {
  return (
    <section className="stage">
      <header className="stage-header">
        <span className="stage-step">{step}</span>
        <h3>{title}</h3>
      </header>
      <div className="stage-explanation">{explanation}</div>
      {formula && <div className="formula">{formula}</div>}
      <div className="stage-body">{children}</div>
    </section>
  );
}
