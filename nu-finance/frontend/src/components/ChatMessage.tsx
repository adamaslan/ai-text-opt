import React from "react";

export interface Source {
  text_preview:  string;
  chunk_summary: string;
  source_file:   string;
  theme_name:    string;
  rerank_score:  number | null;
}

interface Props {
  role:    "user" | "assistant";
  content: string;
  sources?: Source[];
  llmProvider?: string;
}

const TRADER_COLORS: Record<string, string> = {
  gemini:  "#3b82f6",
  mistral: "#f59e0b",
};

export default function ChatMessage({ role, content, sources, llmProvider }: Props) {
  const isUser = role === "user";

  return (
    <div
      className="animate-fade-in-up"
      style={{
        display:        "flex",
        flexDirection:  isUser ? "row-reverse" : "row",
        gap:            "12px",
        alignItems:     "flex-start",
        marginBottom:   "20px",
      }}
    >
      {/* Avatar */}
      <div
        aria-hidden
        style={{
          width:           "36px",
          height:          "36px",
          borderRadius:    "50%",
          flexShrink:      0,
          display:         "flex",
          alignItems:      "center",
          justifyContent:  "center",
          fontSize:        "16px",
          background:      isUser
            ? "linear-gradient(135deg, #3b82f6, #8b5cf6)"
            : "linear-gradient(135deg, #1a2235, #223)",
          border:          "1px solid rgba(255,255,255,0.1)",
          boxShadow:       isUser ? "0 0 12px rgba(59,130,246,0.3)" : "none",
        }}
      >
        {isUser ? "👤" : "🤖"}
      </div>

      {/* Bubble */}
      <div style={{ maxWidth: "75%", minWidth: "60px" }}>
        <div
          style={{
            padding:      "14px 18px",
            borderRadius: isUser ? "18px 4px 18px 18px" : "4px 18px 18px 18px",
            background:   isUser
              ? "linear-gradient(135deg, #2563eb, #7c3aed)"
              : "var(--bg-glass)",
            backdropFilter: isUser ? undefined : "blur(16px)",
            border:       isUser ? "none" : "1px solid var(--border)",
            color:        "var(--text-primary)",
            lineHeight:   1.65,
            fontSize:     "0.9375rem",
            whiteSpace:   "pre-wrap",
            wordBreak:    "break-word",
          }}
        >
          {content}
        </div>

        {/* LLM badge */}
        {!isUser && llmProvider && (
          <div style={{ marginTop: "6px", display: "flex", gap: "6px", alignItems: "center" }}>
            <span
              style={{
                fontSize:        "0.7rem",
                fontWeight:      500,
                color:           TRADER_COLORS[llmProvider] ?? "var(--text-secondary)",
                background:      "rgba(255,255,255,0.05)",
                border:          "1px solid var(--border)",
                borderRadius:    "99px",
                padding:         "2px 8px",
                letterSpacing:   "0.04em",
                textTransform:   "uppercase",
              }}
            >
              {llmProvider}
            </span>
            {sources && sources.length > 0 && (
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                {sources.length} source{sources.length !== 1 ? "s" : ""}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
