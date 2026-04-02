import { useState } from "react";
import type { Source } from "./ChatMessage";

interface Props {
  sources: Source[];
}

const scoreColor = (s: number | null) => {
  if (s === null) return "var(--text-muted)";
  if (s >= 0.85)  return "#10b981";   // green
  if (s >= 0.65)  return "#f59e0b";   // amber
  return "#ef4444";                    // red
};

export default function SourcesPanel({ sources }: Props) {
  const [open, setOpen] = useState(true);
  const [expanded, setExpanded] = useState<number | null>(null);

  if (!sources.length) return null;

  return (
    <aside
      className="glass"
      style={{
        width:        open ? "300px" : "44px",
        flexShrink:   0,
        borderLeft:   "1px solid var(--border)",
        transition:   "width 0.3s cubic-bezier(0.4,0,0.2,1)",
        overflow:     "hidden",
        display:      "flex",
        flexDirection:"column",
      }}
      aria-label="Sources panel"
    >
      {/* Header */}
      <button
        id="sources-panel-toggle"
        onClick={() => setOpen((o) => !o)}
        title={open ? "Collapse sources" : "Expand sources"}
        style={{
          display:        "flex",
          alignItems:     "center",
          gap:            "8px",
          padding:        "14px 14px",
          background:     "transparent",
          color:          "var(--text-secondary)",
          borderBottom:   open ? "1px solid var(--border)" : "none",
          whiteSpace:     "nowrap",
          width:          "100%",
          textAlign:      "left",
        }}
      >
        <span style={{ fontSize: "14px" }}>{open ? "◀" : "▶"}</span>
        {open && (
          <span style={{ fontSize: "0.8125rem", fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Sources · {sources.length}
          </span>
        )}
      </button>

      {/* Source list */}
      {open && (
        <ul
          style={{
            overflowY:   "auto",
            flex:        1,
            padding:     "12px",
            listStyle:   "none",
            display:     "flex",
            flexDirection:"column",
            gap:         "8px",
          }}
        >
          {sources.map((src, i) => (
            <li
              key={i}
              className="animate-fade-in-up"
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <button
                id={`source-item-${i}`}
                onClick={() => setExpanded(expanded === i ? null : i)}
                style={{
                  width:          "100%",
                  textAlign:      "left",
                  background:     expanded === i ? "rgba(59,130,246,0.08)" : "rgba(255,255,255,0.03)",
                  border:         `1px solid ${expanded === i ? "var(--border-accent)" : "var(--border)"}`,
                  borderRadius:   "var(--radius-md)",
                  padding:        "10px 12px",
                  color:          "var(--text-primary)",
                  display:        "flex",
                  flexDirection:  "column",
                  gap:            "6px",
                  transition:     "all var(--transition)",
                }}
              >
                {/* File + score row */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
                  <span
                    style={{
                      fontSize:     "0.7rem",
                      fontFamily:   "JetBrains Mono, monospace",
                      color:        "var(--accent)",
                      overflow:     "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace:   "nowrap",
                      flex:         1,
                    }}
                  >
                    {src.source_file || "—"}
                  </span>
                  {src.rerank_score !== null && (
                    <span
                      style={{
                        fontSize:    "0.68rem",
                        fontWeight:  700,
                        color:       scoreColor(src.rerank_score),
                        flexShrink:  0,
                      }}
                    >
                      {(src.rerank_score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>

                {/* Theme name */}
                {src.theme_name && (
                  <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                    {src.theme_name}
                  </span>
                )}

                {/* Summary */}
                <span
                  style={{
                    fontSize:     "0.78rem",
                    color:        "var(--text-muted)",
                    lineHeight:   1.5,
                    overflow:     "hidden",
                    display:      "-webkit-box",
                    WebkitLineClamp: expanded === i ? 20 : 2,
                    WebkitBoxOrient: "vertical",
                    transition:   "all var(--transition)",
                  }}
                >
                  {src.chunk_summary || src.text_preview}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
