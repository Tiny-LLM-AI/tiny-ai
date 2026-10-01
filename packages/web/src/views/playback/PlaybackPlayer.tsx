import { useEffect, useMemo, useState } from "react";
import { MatrixHeatmap } from "../../components/MatrixHeatmap";
import type { PlaybackStage } from "../../playback/frame-types";

const SPEEDS = [
  { label: "0.25×", delay: 2400 },
  { label: "0.5×", delay: 1200 },
  { label: "1×", delay: 600 },
  { label: "2×", delay: 300 },
  { label: "4×", delay: 150 },
  { label: "8×", delay: 75 },
  { label: "16×", delay: 35 },
];
const DEFAULT_SPEED = 2;

export function PlaybackPlayer({ stages }: { stages: PlaybackStage[] }) {
  const frames = useMemo(
    () => stages.flatMap((stage, stageIndex) => stage.frames.map((frame) => ({ frame, stageIndex }))),
    [stages],
  );
  const stageStarts = useMemo(() => {
    const starts: number[] = [];
    let total = 0;
    for (const stage of stages) {
      starts.push(total);
      total += stage.frames.length;
    }
    return starts;
  }, [stages]);

  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedIndex, setSpeedIndex] = useState(DEFAULT_SPEED);
  const last = frames.length - 1;
  const current = frames[Math.min(index, last)];
  const stage = stages[current.stageIndex];

  useEffect(() => {
    if (!isPlaying) return;
    if (index >= last) {
      setIsPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => setIndex((value) => value + 1), SPEEDS[speedIndex].delay);
    return () => window.clearTimeout(timer);
  }, [isPlaying, index, last, speedIndex]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.key === " ") {
        event.preventDefault();
        setIsPlaying((value) => !value);
      } else if (event.key === "ArrowRight") {
        setIsPlaying(false);
        setIndex((value) => Math.min(last, value + 1));
      } else if (event.key === "ArrowLeft") {
        setIsPlaying(false);
        setIndex((value) => Math.max(0, value - 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last]);

  const goTo = (value: number) => {
    setIsPlaying(false);
    setIndex(Math.max(0, Math.min(last, value)));
  };

  return (
    <section className="panel player">
      <div className="player-controls">
        <button type="button" className="secondary" onClick={() => goTo(0)} title="Restart">
          ⏮
        </button>
        <button type="button" className="secondary" onClick={() => goTo(index - 1)} title="Previous frame (←)">
          ◀
        </button>
        <button
          type="button"
          onClick={() => {
            if (index >= last) setIndex(0);
            setIsPlaying((value) => !value);
          }}
          title="Play / pause (space)"
        >
          {isPlaying ? "⏸ Pause" : "▶ Play"}
        </button>
        <button type="button" className="secondary" onClick={() => goTo(index + 1)} title="Next frame (→)">
          ▶
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => goTo(stageStarts[current.stageIndex + 1] ?? last)}
          title="Skip to next stage"
        >
          ⏭
        </button>
        <label className="speed">
          Speed
          <input
            type="range"
            min={0}
            max={SPEEDS.length - 1}
            value={speedIndex}
            onChange={(event) => setSpeedIndex(Number(event.target.value))}
          />
          <strong>{SPEEDS[speedIndex].label}</strong>
        </label>
        <input
          className="scrubber"
          type="range"
          min={0}
          max={last}
          value={index}
          onChange={(event) => goTo(Number(event.target.value))}
        />
        <span className="muted">
          frame {index + 1} / {frames.length}
        </span>
      </div>

      <div className="stage-chips">
        {stages.map((item, stageIndex) => (
          <button
            key={stageIndex}
            type="button"
            className={`stage-chip ${stageIndex === current.stageIndex ? "active" : ""} ${stageIndex < current.stageIndex ? "done" : ""}`}
            onClick={() => goTo(stageStarts[stageIndex])}
            title={item.title}
          >
            {stageIndex + 1}. {item.shortTitle}
          </button>
        ))}
      </div>

      <div className="player-header">
        <h3>
          {current.stageIndex + 1}. {stage.title}
        </h3>
        <code className="formula">{stage.formula}</code>
      </div>

      <p className="narration">{current.frame.narration}</p>
      <div className="calculation">
        {current.frame.calculation.length === 0 ? (
          <span className="muted">—</span>
        ) : (
          current.frame.calculation.map((line, lineIndex) => <div key={lineIndex}>{line}</div>)
        )}
      </div>

      <div className="player-legend muted">
        <span className="legend-swatch reading" /> being read
        <span className="legend-swatch writing" /> being written
        <span className="legend-swatch pending" /> not computed yet
        <span>· red = positive, blue = negative, green = probability</span>
      </div>

      <div className="player-panels">
        {current.frame.panels.map((item) => (
          <MatrixHeatmap
            key={item.key}
            matrix={item.matrix}
            title={item.title}
            rowLabels={item.rowLabels}
            columnLabels={item.columnLabels}
            colorMode={item.colorMode}
            digits={item.digits}
            highlightedRows={item.highlightedRows}
            highlightedColumns={item.highlightedColumns}
            highlightedCells={item.highlightedCells}
            activeCells={item.activeCells}
            isCellRevealed={item.isCellRevealed}
          />
        ))}
      </div>
    </section>
  );
}
