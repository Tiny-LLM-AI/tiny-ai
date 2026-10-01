import { useCallback, useRef, useState } from "react";
import {
  createOptimizer,
  createTrainingSession,
  DEFAULT_MODEL_CONFIG,
  DEFAULT_TRAINING_CONFIG,
  trainOneEpoch,
  trainOneSampleWithReport,
  type ArithmeticExample,
  type TrainingConfig,
  type TrainingSample,
  type TrainingSession,
  type TrainingStepReport,
} from "@math-llm/core";

const waitForNextFrame = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/**
 * Owns the mutable training session. The core mutates weights in place for speed,
 * so `revision` is bumped after every change to tell React to re-render.
 */
export function useTrainingSession(examples: ArithmeticExample[]) {
  const sessionRef = useRef<TrainingSession | null>(null);
  if (sessionRef.current === null) {
    sessionRef.current = createTrainingSession(examples, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG);
  }

  const [revision, setRevision] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [lastReport, setLastReport] = useState<TrainingStepReport | null>(null);
  const stopRequested = useRef(false);

  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  const session = sessionRef.current;

  const resetModel = useCallback(
    (trainingConfig: TrainingConfig) => {
      sessionRef.current = createTrainingSession(examples, DEFAULT_MODEL_CONFIG, trainingConfig);
      setLastReport(null);
      refresh();
    },
    [examples, refresh],
  );

  const updateTrainingConfig = useCallback(
    (changes: Partial<TrainingConfig>) => {
      const current = sessionRef.current!;
      const next = { ...current.trainingConfig, ...changes };
      current.trainingConfig = next;
      if (changes.optimizer && changes.optimizer !== current.optimizer.name) {
        current.optimizer = createOptimizer(next.optimizer, next.learningRate, current.model.parameters);
      }
      current.optimizer.learningRate = next.learningRate;
      refresh();
    },
    [refresh],
  );

  const trainOneSample = useCallback(
    (sample: TrainingSample) => {
      setLastReport(trainOneSampleWithReport(sessionRef.current!, sample));
      refresh();
    },
    [refresh],
  );

  const trainEpochs = useCallback(
    async (epochCount: number) => {
      stopRequested.current = false;
      setIsRunning(true);
      setLastReport(null);
      try {
        for (let i = 0; i < epochCount && !stopRequested.current; i += 1) {
          trainOneEpoch(sessionRef.current!);
          refresh();
          await waitForNextFrame();
        }
      } finally {
        setIsRunning(false);
      }
    },
    [refresh],
  );

  const stopTraining = useCallback(() => {
    stopRequested.current = true;
  }, []);

  return { session, revision, isRunning, lastReport, resetModel, updateTrainingConfig, trainOneSample, trainEpochs, stopTraining };
}

export type TrainingSessionController = ReturnType<typeof useTrainingSession>;
