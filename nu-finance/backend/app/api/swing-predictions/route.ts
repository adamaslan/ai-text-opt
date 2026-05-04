import { NextRequest, NextResponse } from "next/server";
import { readFileSync } from "fs";
import { join } from "path";

const GCP3_BACKEND_URL = process.env.GCP3_BACKEND_URL || "http://localhost:8080";
const FETCH_TIMEOUT_MS = 90_000; // yfinance scan of 250 stocks takes ~60s cold
const MAX_RETRIES = 1;
const RETRY_DELAY_MS = 2_000;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
    });
  } finally {
    clearTimeout(id);
  }
}

function loadFallback(): NextResponse {
  try {
    const fallbackPath = join(process.cwd(), "data", "swing-predictions-fallback.json");
    const raw = readFileSync(fallbackPath, "utf-8");
    const data = JSON.parse(raw);
    // Tag it so the UI can show a stale-data notice
    data._fallback = true;
    data._fallback_reason = "upstream timeout — showing last cached scan";
    return NextResponse.json(data, {
      headers: { "X-Data-Source": "local-fallback" },
    });
  } catch {
    return NextResponse.json(
      { error: "Upstream unavailable and no local fallback found", retryable: true },
      { status: 503 }
    );
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const forceRefresh = searchParams.get("force_refresh") ?? "false";
  const topN = searchParams.get("top_n") ?? "10";
  const period = searchParams.get("period") ?? "3mo";
  const universe = searchParams.get("universe") ?? "sp500";

  const upstream = `${GCP3_BACKEND_URL}/swing-predictions?universe=${universe}&top_n=${topN}&period=${period}&force_refresh=${forceRefresh}`;

  let lastError = "Unknown error";

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) await sleep(RETRY_DELAY_MS);

    try {
      const response = await fetchWithTimeout(upstream, FETCH_TIMEOUT_MS);

      if (!response.ok) {
        const body = await response.text().catch(() => "");
        lastError = `Upstream ${response.status}: ${body.slice(0, 200)}`;
        // 4xx → not retryable, serve fallback immediately
        if (response.status >= 400 && response.status < 500) {
          console.error(`[swing-predictions] 4xx from upstream: ${lastError}`);
          return loadFallback();
        }
        continue; // retry 5xx once
      }

      const data = await response.json();

      // Persist as new fallback in background (best-effort, non-blocking)
      try {
        const { writeFileSync } = await import("fs");
        const { join: pathJoin } = await import("path");
        writeFileSync(
          pathJoin(process.cwd(), "data", "swing-predictions-fallback.json"),
          JSON.stringify(data)
        );
      } catch { /* non-fatal */ }

      return NextResponse.json(data, {
        headers: {
          "Cache-Control": "public, s-maxage=60, stale-while-revalidate=840",
          "X-Data-Source": "live",
        },
      });
    } catch (err: unknown) {
      const isTimeout = err instanceof Error && err.name === "AbortError";
      lastError = isTimeout
        ? `Upstream timed out after ${FETCH_TIMEOUT_MS / 1000}s (attempt ${attempt + 1})`
        : `Fetch error: ${err instanceof Error ? err.message : String(err)}`;
      console.error(`[swing-predictions] attempt=${attempt + 1} ${lastError}`);
    }
  }

  // All attempts exhausted — serve local fallback
  console.warn("[swing-predictions] falling back to local JSON:", lastError);
  return loadFallback();
}
