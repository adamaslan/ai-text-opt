// lib/workflow.ts
// TraderQueryWorkflow — routes each query to the appropriate engine:
//   1. Explicit UI toggle (T1 / T2)
//   2. Heuristic keyword detection
//   3. Comparative / both-trader question → SubQuestionEngine
//   4. Default broad search

import { Workflow, StartEvent, StopEvent, step } from "llamaindex";
import { broadEngine, subQuestionEngine, traderEngine } from "./rag";

/** Shape of the data passed to traderWorkflow.run() */
interface TraderQueryInput {
  query: string;
  /** Explicit trader filter from the frontend toggle.  null = auto-detect. */
  traderFilter?: "T1" | "T2" | null;
}

class TraderQueryWorkflow extends Workflow {
  @step()
  async route(ev: StartEvent): Promise<StopEvent> {
    const { query, traderFilter } = ev.data as TraderQueryInput;

    // 1. Explicit trader filter from UI toggle ────────────────────────────────
    if (traderFilter === "T1" || traderFilter === "T2") {
      const response = await traderEngine(traderFilter).query({ query });
      return new StopEvent({ result: response });
    }

    // 2. Heuristic: detect trader mention in query text ───────────────────────
    const mentionsT1 = /tactical|opportunist|\bt1\b/i.test(query);
    const mentionsT2 = /structured|growth investor|\bt2\b/i.test(query);

    if (mentionsT1 && !mentionsT2) {
      const response = await traderEngine("T1").query({ query });
      return new StopEvent({ result: response });
    }
    if (mentionsT2 && !mentionsT1) {
      const response = await traderEngine("T2").query({ query });
      return new StopEvent({ result: response });
    }

    // 3. Comparative → sub-question decomposition ─────────────────────────────
    if (/both|compare|differ|vs\.?|versus/i.test(query)) {
      const response = await subQuestionEngine.query({ query });
      return new StopEvent({ result: response });
    }

    // 4. Default: broad search + Voyage rerank ────────────────────────────────
    const response = await broadEngine.query({ query });
    return new StopEvent({ result: response });
  }
}

export const traderWorkflow = new TraderQueryWorkflow();
