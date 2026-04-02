// app/api/chat/route.ts
// POST /api/chat — wires the request body to the TraderQueryWorkflow.

import { NextRequest, NextResponse } from "next/server";
import { StartEvent } from "llamaindex";
import { traderWorkflow } from "@/lib/workflow";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const message: string = body.message ?? "";
  const traderFilter: "T1" | "T2" | null = body.trader_filter ?? null;

  if (!message.trim()) {
    return NextResponse.json({ error: "message is required" }, { status: 400 });
  }

  // LlamaIndex Workflow: run() accepts a StartEvent and returns a StopEvent.
  const stopEvent = await traderWorkflow.run(
    new StartEvent({ query: message, traderFilter })
  );
  const response = stopEvent.data.result;

  return NextResponse.json({
    answer: response.response,
    /** Surface active LLM for debugging / the UI badge */
    llm_provider: process.env.LLM_PROVIDER ?? "gemini",
    sources: response.sourceNodes?.map((node: any) => ({
      text_preview:  node.node.text.slice(0, 200),
      chunk_summary: node.node.metadata.chunk_summary ?? "",
      source_file:   node.node.metadata.source_file ?? "",
      theme_name:    node.node.metadata.theme_name ?? "",
      rerank_score:  node.score ?? null,
    })) ?? [],
  });
}
