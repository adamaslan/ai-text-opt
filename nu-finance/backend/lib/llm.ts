// lib/llm.ts
// LLM abstraction — returns Gemini or Mistral based on LLM_PROVIDER env var.
// To swap LLMs, change ONE env var: LLM_PROVIDER=mistral (no code changes needed).

import { Gemini, MistralAI } from "llamaindex";

export type LLMProvider = "gemini" | "mistral";

export function getLLM(provider?: LLMProvider) {
  const active = (provider ?? process.env.LLM_PROVIDER ?? "gemini") as LLMProvider;

  switch (active) {
    // ── Default: Google Gemini 2.5 Flash (free tier via Google AI Studio) ──
    case "gemini":
      return new Gemini({
        apiKey: process.env.GEMINI_API_KEY!,
        model: (process.env.GEMINI_MODEL as any) ?? "gemini-2.5-flash",
      });

    // ── Swap: Mistral (free tier: mistral-small-latest) ────────────────────
    case "mistral":
      return new MistralAI({
        apiKey: process.env.MISTRAL_API_KEY!,
        model: process.env.MISTRAL_MODEL ?? "mistral-small-latest",
      });

    default:
      throw new Error(
        `Unknown LLM_PROVIDER: "${active}". Valid values: "gemini" | "mistral".`
      );
  }
}
