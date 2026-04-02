import { useState, useRef, useEffect, FormEvent } from "react";
import ChatMessage, { Source } from "../components/ChatMessage";
import SourcesPanel from "../components/SourcesPanel";

type TraderFilter = "T1" | "T2" | "both";

interface Message {
  role:        "user" | "assistant";
  content:     string;
  sources?:    Source[];
  llmProvider?: string;
}

const FILTER_CONFIG: Record<TraderFilter, { label: string; color: string; desc: string }> = {
  T1:   { label: "T1 · Tactical",  color: "var(--accent-t1)", desc: "Tactical Opportunist" },
  T2:   { label: "T2 · Structured", color: "var(--accent-t2)", desc: "Structured Growth" },
  both: { label: "Both",            color: "var(--accent-both)", desc: "Both traders (comparative)" },
};

export default function ChatPage() {
  const [messages,  setMessages]  = useState<Message[]>([]);
  const [input,     setInput]     = useState("");
  const [loading,   setLoading]   = useState(false);
  const [filter,    setFilter]    = useState<TraderFilter>("both");
  const [latestSrcs, setLatestSrcs] = useState<Source[]>([]);

  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Auto-resize textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  };

  const sendMessage = async (e?: FormEvent) => {
    e?.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          message:       text,
          trader_filter: filter === "both" ? null : filter,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error ?? `Server error ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg: Message = {
        role:        "assistant",
        content:     data.answer,
        sources:     data.sources ?? [],
        llmProvider: data.llm_provider,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setLatestSrcs(data.sources ?? []);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `⚠️ ${err.message || "Something went wrong. Please try again."}` },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setLatestSrcs([]);
  };

  const activeConfig = FILTER_CONFIG[filter];

  return (
    <div style={{ display: "flex", width: "100%", height: "100dvh", overflow: "hidden" }}>

      {/* ── Main chat column ─────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Header */}
        <header
          className="glass"
          style={{
            padding:        "16px 24px",
            borderBottom:   "1px solid var(--border)",
            display:        "flex",
            alignItems:     "center",
            justifyContent: "space-between",
            gap:            "16px",
            flexShrink:     0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span
              style={{
                fontSize:   "22px",
                background: "linear-gradient(135deg,#3b82f6,#8b5cf6)",
                borderRadius:"8px",
                padding:    "4px 8px",
              }}
            >
              📈
            </span>
            <div>
              <h1 style={{ fontSize: "1rem", fontWeight: 700, lineHeight: 1.2 }}>
                Nu-Finance
              </h1>
              <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>
                Trader Q&amp;A · LlamaIndex + Zilliz
              </p>
            </div>
          </div>

          {/* Trader filter toggle */}
          <div
            role="group"
            aria-label="Trader filter"
            style={{
              display:      "flex",
              gap:          "4px",
              background:   "rgba(255,255,255,0.04)",
              border:       "1px solid var(--border)",
              borderRadius: "var(--radius-lg)",
              padding:      "3px",
            }}
          >
            {(Object.keys(FILTER_CONFIG) as TraderFilter[]).map((key) => {
              const cfg  = FILTER_CONFIG[key];
              const isOn = filter === key;
              return (
                <button
                  key={key}
                  id={`filter-btn-${key.toLowerCase()}`}
                  onClick={() => setFilter(key)}
                  title={cfg.desc}
                  style={{
                    padding:      "5px 14px",
                    borderRadius: "var(--radius-lg)",
                    background:   isOn ? cfg.color : "transparent",
                    color:        isOn ? "#fff" : "var(--text-secondary)",
                    fontWeight:   isOn ? 600 : 400,
                    fontSize:     "0.8rem",
                    boxShadow:    isOn ? `0 0 12px ${cfg.color}55` : "none",
                  }}
                >
                  {cfg.label}
                </button>
              );
            })}
          </div>

          {/* Clear button */}
          <button
            id="clear-chat-btn"
            onClick={clearChat}
            disabled={messages.length === 0}
            style={{
              padding:      "6px 14px",
              borderRadius: "var(--radius-md)",
              background:   "transparent",
              border:       "1px solid var(--border)",
              color:        messages.length === 0 ? "var(--text-muted)" : "var(--text-secondary)",
              fontSize:     "0.8rem",
            }}
          >
            Clear
          </button>
        </header>

        {/* Messages area */}
        <main
          style={{
            flex:        1,
            overflowY:   "auto",
            padding:     "24px",
            display:     "flex",
            flexDirection:"column",
          }}
        >
          {messages.length === 0 && (
            <div
              style={{
                flex:           1,
                display:        "flex",
                flexDirection:  "column",
                alignItems:     "center",
                justifyContent: "center",
                gap:            "16px",
                color:          "var(--text-muted)",
                textAlign:      "center",
                padding:        "40px",
              }}
            >
              <span style={{ fontSize: "48px" }}>💬</span>
              <p style={{ fontSize: "1rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                Ask anything about trading
              </p>
              <p style={{ fontSize: "0.85rem", maxWidth: "360px", lineHeight: 1.7 }}>
                Try&nbsp;
                <em>"How does the Tactical Opportunist trade earnings plays?"</em>
                &nbsp;or&nbsp;
                <em>"Compare both traders on NVDA."</em>
              </p>
              <div
                style={{
                  display:       "flex",
                  flexWrap:      "wrap",
                  gap:           "8px",
                  justifyContent:"center",
                  marginTop:     "8px",
                }}
              >
                {[
                  "What is IV crush?",
                  "How does T1 use covered calls?",
                  "Compare both traders on AI stocks",
                  "Best position sizing approach?",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); inputRef.current?.focus(); }}
                    style={{
                      padding:      "7px 14px",
                      borderRadius: "var(--radius-lg)",
                      background:   "rgba(255,255,255,0.04)",
                      border:       "1px solid var(--border)",
                      color:        "var(--text-secondary)",
                      fontSize:     "0.8rem",
                    }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatMessage
              key={i}
              role={msg.role}
              content={msg.content}
              sources={msg.sources}
              llmProvider={msg.llmProvider}
            />
          ))}

          {/* Loading indicator */}
          {loading && (
            <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "4px 0 12px" }}>
              <div className="spinner" />
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Thinking…</span>
            </div>
          )}

          <div ref={bottomRef} />
        </main>

        {/* Input bar */}
        <form
          onSubmit={sendMessage}
          className="glass"
          style={{
            padding:      "14px 20px",
            borderTop:    "1px solid var(--border)",
            display:      "flex",
            alignItems:   "flex-end",
            gap:          "12px",
            flexShrink:   0,
          }}
        >
          {/* Active filter dot */}
          <span
            style={{
              width:        "8px",
              height:       "8px",
              borderRadius: "50%",
              background:   activeConfig.color,
              flexShrink:   0,
              marginBottom: "12px",
              boxShadow:    `0 0 8px ${activeConfig.color}`,
            }}
            title={`Filtering: ${activeConfig.desc}`}
          />

          <textarea
            ref={inputRef}
            id="chat-input"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask a trading question… (Shift+Enter for newline)"
            rows={1}
            style={{
              flex:        1,
              resize:      "none",
              background:  "rgba(255,255,255,0.04)",
              border:      "1px solid var(--border)",
              borderRadius:"var(--radius-md)",
              padding:     "10px 14px",
              color:       "var(--text-primary)",
              fontSize:    "0.9375rem",
              lineHeight:  1.5,
              outline:     "none",
              fontFamily:  "Inter, sans-serif",
              transition:  "border-color var(--transition)",
              maxHeight:   "160px",
              overflowY:   "auto",
            }}
            onFocus={(e) => (e.target.style.borderColor = "var(--border-accent)")}
            onBlur={(e)  => (e.target.style.borderColor = "var(--border)")}
          />

          <button
            id="send-btn"
            type="submit"
            disabled={!input.trim() || loading}
            style={{
              width:        "42px",
              height:       "42px",
              borderRadius: "var(--radius-md)",
              background:   input.trim() && !loading
                ? "linear-gradient(135deg, #2563eb, #7c3aed)"
                : "rgba(255,255,255,0.06)",
              color:        input.trim() && !loading ? "#fff" : "var(--text-muted)",
              fontSize:     "18px",
              flexShrink:   0,
              display:      "flex",
              alignItems:   "center",
              justifyContent:"center",
              boxShadow:    input.trim() && !loading ? "var(--shadow-glow)" : "none",
            }}
            aria-label="Send message"
          >
            {loading ? <span className="spinner" style={{ width:"18px",height:"18px" }} /> : "↑"}
          </button>
        </form>
      </div>

      {/* ── Sources sidebar ───────────────────────────────────────────────── */}
      <SourcesPanel sources={latestSrcs} />
    </div>
  );
}
