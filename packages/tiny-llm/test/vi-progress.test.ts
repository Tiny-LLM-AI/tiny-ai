import { expect, it } from "vitest";
import { newViCurriculum, planViPhase, restoreViCurriculum } from "../src/vi-curriculum.js";

it("adds one foundation-sized block per phase, resumes saved partial progress", () => {
  let state = newViCurriculum(3000);
  planViPhase(state, "foundation", 0);
  expect(state.plannedEndStep).toBe(3000);
  state.foundationSteps = 3000;
  planViPhase(state, "chat", 6172);
  expect(state.plannedEndStep).toBe(9172);
  expect(state.chatTargetSteps).toBe(3000);
  state.chatSteps = 500;
  state = restoreViCurriculum(JSON.parse(JSON.stringify(state)));
  planViPhase(state, "chat", 6672);
  expect(state.plannedEndStep).toBe(9172);
  state.chatSteps = 3000;
  planViPhase(state, "wikipedia", 9172);
  expect(state.plannedEndStep).toBe(12172);
  planViPhase(state, "wikipedia", 10000);
  expect(state.plannedEndStep).toBe(12172);
  planViPhase(state, "wikipedia", 12172);
  expect(state.plannedEndStep).toBe(15172);
});
