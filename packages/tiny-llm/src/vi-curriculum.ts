/** Persisted progress is separate from total model steps (old wiki runs do not count). */
export const VI_FOUNDATION_ID = "vie_news_2020_100K:117757c4ee804a1f34a414d200d6df92474ac564d840b686a46c52ea0694929c";
export const VI_FOUNDATION_STEPS = 3000;
export interface ViCurriculum {
  corpusId: string;
  phase: "foundation" | "wikipedia" | "chat";
  foundationSteps: number;
  targetSteps: number;
  wikiSeed: string;
  chatSteps: number;
  chatTargetSteps: number;
  plannedEndStep?: number;
}

export function newViCurriculum(targetSteps = VI_FOUNDATION_STEPS): ViCurriculum {
  if (!Number.isSafeInteger(targetSteps) || targetSteps < 1) throw new Error("Foundation steps must be a positive integer.");
  return { corpusId: VI_FOUNDATION_ID, phase: "foundation", foundationSteps: 0, targetSteps, wikiSeed: "https://vi.wikipedia.org/wiki/Tiếng_Việt", chatSteps: 0, chatTargetSteps: 1000 };
}

export function restoreViCurriculum(value: unknown): ViCurriculum {
  const v = value as Partial<ViCurriculum> | undefined;
  if (!v || v.corpusId !== VI_FOUNDATION_ID ||
      !Number.isSafeInteger(v.foundationSteps) || v.foundationSteps! < 0 ||
      !Number.isSafeInteger(v.targetSteps) || v.targetSteps! < 1 ||
      (v.phase !== "foundation" && v.phase !== "wikipedia" && v.phase !== "chat")) return newViCurriculum();
  return {
    corpusId: VI_FOUNDATION_ID,
    plannedEndStep: Number.isSafeInteger(v.plannedEndStep) && v.plannedEndStep! >= 0 ? v.plannedEndStep : undefined,
    phase: v.phase,
    foundationSteps: v.foundationSteps!,
    targetSteps: v.targetSteps!,
    wikiSeed: typeof v.wikiSeed === "string" && v.wikiSeed.trim() ? v.wikiSeed : newViCurriculum().wikiSeed,
    chatSteps: Number.isSafeInteger(v.chatSteps) && v.chatSteps! >= 0 ? v.chatSteps! : 0,
    chatTargetSteps: Number.isSafeInteger(v.chatTargetSteps) && v.chatTargetSteps! >= 1 ? v.chatTargetSteps! : 1000,
  };
}

export function chatReady(state: ViCurriculum): boolean {
  return foundationReady(state) && state.chatSteps >= state.chatTargetSteps;
}

export function foundationReady(state: ViCurriculum): boolean {
  return state.corpusId === VI_FOUNDATION_ID && state.foundationSteps >= state.targetSteps;
}

/** Resume an unfinished phase; completed phases schedule another foundation-sized block. */
export function planViPhase(state: ViCurriculum, phase: ViCurriculum["phase"], step: number): void {
  if (state.phase !== phase || state.plannedEndStep == null || state.plannedEndStep <= step) {
    state.plannedEndStep = step + (phase === "foundation"
      ? Math.max(0, state.targetSteps - state.foundationSteps) : state.targetSteps);
    if (phase === "chat") state.chatTargetSteps = state.chatSteps + state.plannedEndStep - step;
  }
  state.phase = phase;
}
