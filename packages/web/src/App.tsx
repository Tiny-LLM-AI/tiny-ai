import { useState } from "react";
import { validateDataset, type ArithmeticDatasetFile } from "@math-llm/core";
import datasetFile from "../../../train-data/arithmetic.json";
import { DisplaySettingsContext } from "./components/DisplaySettings";
import { useTrainingSession } from "./state/useTrainingSession";
import { AskView } from "./views/AskView";
import { DataView } from "./views/DataView";
import { PlaybackView } from "./views/playback/PlaybackView";
import { TrainingView } from "./views/training/TrainingView";

const examples = validateDataset(datasetFile as ArithmeticDatasetFile);

const TABS = [
  { id: "data", label: "1. Data & model" },
  { id: "watch", label: "2. Watch it compute" },
  { id: "ask", label: "3. Ask: inspect every matrix" },
  { id: "train", label: "4. Train: backward pass" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>("data");
  const [showCellValues, setShowCellValues] = useState(true);
  const controller = useTrainingSession(examples);

  return (
    <DisplaySettingsContext.Provider value={{ showCellValues }}>
      <div className="app">
        <header className="header">
          <div>
            <h1>Tiny Arithmetic Transformer</h1>
            <p>
              A GPT-style model with {controller.session.model.config.blockCount} block and a few thousand weights,
              learning single-digit arithmetic. Every matrix, gradient and weight update is visible.
            </p>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={showCellValues} onChange={(event) => setShowCellValues(event.target.checked)} />
            Show numbers in matrix cells
          </label>
        </header>

        <nav className="tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
          <span className="model-status">
            {controller.session.epoch === 0
              ? "Model: untrained (random weights)"
              : `Model: trained ${controller.session.epoch} epochs`}
          </span>
        </nav>

        {activeTab === "data" && <DataView session={controller.session} />}
        {activeTab === "watch" && <PlaybackView session={controller.session} revision={controller.revision} />}
        {activeTab === "ask" && <AskView session={controller.session} revision={controller.revision} />}
        {activeTab === "train" && <TrainingView controller={controller} />}
      </div>
    </DisplaySettingsContext.Provider>
  );
}
