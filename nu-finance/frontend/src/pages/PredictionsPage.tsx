import { useState, useEffect, useCallback } from "react";

interface Candidate {
  symbol: string;
  score: number;
  price: number;
  rsi: number;
  adx: number;
  plus_di: number;
  minus_di: number;
  macd_hist: number;
  momentum_10d: number;
  momentum_21d: number;
  volume_ratio: number;
  atr: number;
  stop_loss: number;
  target: number;
  bb_position: number;
  sector_etf_momentum?: number;
  reason: string;
}

interface SwingPredictions {
  date: string;
  generated_at: string;
  horizon: string;
  methodology: string;
  period_used: string;
  universe: string;
  buy_candidates: Candidate[];
  sell_candidates: Candidate[];
  total_analyzed: number;
  from_cache: number;
  live_fetched: number;
  errors: number;
  industry_sectors_loaded: number;
  _fallback?: boolean;
  _fallback_reason?: string;
}

type FetchStatus = "idle" | "loading" | "success" | "error";

function ScoreBar({ score, direction }: { score: number; direction: "buy" | "sell" }) {
  const pct = Math.min(100, Math.max(0, score));
  const color = direction === "buy" ? "var(--accent-t2)" : "#ef4444";
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="score-bar-label">{score.toFixed(0)}</span>
    </div>
  );
}

function IndicatorPill({
  label,
  value,
  className = "",
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className={`indicator-pill ${className}`}>
      <span className="pill-label">{label}</span>
      <span className="pill-value">{value}</span>
    </div>
  );
}

function CandidateCard({
  cand,
  rank,
  direction,
}: {
  cand: Candidate;
  rank: number;
  direction: "buy" | "sell";
}) {
  const rsiClass =
    cand.rsi > 70 ? "overbought" : cand.rsi < 30 ? "oversold" : "";
  const adxClass = cand.adx > 25 ? "trending" : "ranging";
  const mom10Class = cand.momentum_10d > 0 ? "positive" : "negative";
  const mom21Class = cand.momentum_21d > 0 ? "positive" : "negative";
  const macdClass = cand.macd_hist > 0 ? "positive" : "negative";
  const volClass =
    cand.volume_ratio >= 1.5 ? "volume-spike" : cand.volume_ratio < 0.8 ? "volume-low" : "";

  const rr = cand.atr > 0
    ? Math.abs(cand.target - cand.price) / Math.abs(cand.price - cand.stop_loss)
    : null;

  return (
    <div className={`candidate-card ${direction}`}>
      <div className="card-header">
        <span className="rank">#{rank}</span>
        <span className="symbol">{cand.symbol}</span>
        <div className="card-header-right">
          <ScoreBar score={cand.score} direction={direction} />
        </div>
      </div>

      <div className="card-body">
        <div className="price-row">
          <span className="price">${cand.price.toFixed(2)}</span>
          {rr !== null && (
            <span className={`rr-badge ${rr >= 2 ? "rr-good" : rr >= 1.5 ? "rr-ok" : "rr-weak"}`}>
              R:R {rr.toFixed(1)}×
            </span>
          )}
        </div>

        <div className="indicators-grid">
          <IndicatorPill label="RSI" value={cand.rsi.toFixed(0)} className={rsiClass} />
          <IndicatorPill label="ADX" value={cand.adx.toFixed(0)} className={adxClass} />
          <IndicatorPill
            label="MACD"
            value={cand.macd_hist > 0 ? `+${cand.macd_hist.toFixed(3)}` : cand.macd_hist.toFixed(3)}
            className={macdClass}
          />
          <IndicatorPill
            label="10D"
            value={`${cand.momentum_10d > 0 ? "+" : ""}${cand.momentum_10d.toFixed(1)}%`}
            className={mom10Class}
          />
          <IndicatorPill
            label="21D"
            value={`${cand.momentum_21d > 0 ? "+" : ""}${cand.momentum_21d.toFixed(1)}%`}
            className={mom21Class}
          />
          <IndicatorPill
            label="VOL"
            value={`${cand.volume_ratio.toFixed(1)}×`}
            className={volClass}
          />
        </div>

        <div className="stop-target-row">
          <div className="stop-block">
            <span className="st-label">Stop</span>
            <span className="st-value stop">${cand.stop_loss.toFixed(2)}</span>
          </div>
          <div className="atr-block">
            <span className="st-label">ATR</span>
            <span className="st-value">${cand.atr.toFixed(2)}</span>
          </div>
          <div className="target-block">
            <span className="st-label">Target</span>
            <span className="st-value target">${cand.target.toFixed(2)}</span>
          </div>
        </div>

        <div className="bb-bar-wrap">
          <span className="bb-label">BB</span>
          <div className="bb-track">
            <div
              className="bb-marker"
              style={{ left: `${Math.min(100, Math.max(0, cand.bb_position * 100))}%` }}
            />
          </div>
          <span className="bb-label">{(cand.bb_position * 100).toFixed(0)}%</span>
        </div>

        <div className="di-row">
          <span className={`di-pill ${cand.plus_di > cand.minus_di ? "positive" : ""}`}>
            +DI {cand.plus_di.toFixed(0)}
          </span>
          <span className={`di-pill ${cand.minus_di > cand.plus_di ? "negative" : ""}`}>
            −DI {cand.minus_di.toFixed(0)}
          </span>
          {cand.sector_etf_momentum !== undefined && (
            <span className={`di-pill ${cand.sector_etf_momentum > 0 ? "positive" : cand.sector_etf_momentum < 0 ? "negative" : ""}`}>
              Sector {cand.sector_etf_momentum > 0 ? "+" : ""}{cand.sector_etf_momentum.toFixed(1)}%
            </span>
          )}
        </div>

        <div className="reason">{cand.reason}</div>
      </div>
    </div>
  );
}

export default function PredictionsPage() {
  const [data, setData] = useState<SwingPredictions | null>(null);
  const [status, setStatus] = useState<FetchStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"buy" | "sell">("buy");
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  const load = useCallback((forceRefresh = false) => {
    setStatus("loading");
    setError(null);
    const url = `/api/swing-predictions?universe=sp500&top_n=10&force_refresh=${forceRefresh}`;
    fetch(url)
      .then((res) => {
        if (!res.ok) {
          return res.json().then((body) => {
            throw new Error(body?.error ?? `HTTP ${res.status}`);
          });
        }
        return res.json();
      })
      .then((d: SwingPredictions) => {
        setData(d);
        setStatus("success");
        setLastFetched(new Date());
      })
      .catch((err: Error) => {
        setError(err.message);
        setStatus("error");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const candidates =
    activeTab === "buy" ? data?.buy_candidates : data?.sell_candidates;

  const headerSubtitle = data
    ? `${data.horizon} • ${data.universe?.toUpperCase() ?? "SP500"} • ${data.total_analyzed} stocks scanned`
    : null;

  return (
    <div className="predictions-page">
      <header className="page-header">
        <div className="header-top">
          <div>
            <h1>Swing Trade Predictions</h1>
            {headerSubtitle && <p className="subtitle">{headerSubtitle}</p>}
            {lastFetched && data && (
              <p className="last-fetched">
                Updated {lastFetched.toLocaleTimeString()}
                {" • "}{data.from_cache ?? 0} from GCP cache
                {data.live_fetched ? ` • ${data.live_fetched} live` : ""}
                {data.industry_sectors_loaded ? ` • ${data.industry_sectors_loaded} sectors` : ""}
                {data.errors ? ` • ${data.errors} skipped` : ""}
              </p>
            )}
          </div>
          <div className="header-actions">
            <button
              className="refresh-btn"
              disabled={status === "loading"}
              onClick={() => load(true)}
            >
              {status === "loading" ? (
                <span className="spinner-sm" />
              ) : (
                "↻ Refresh"
              )}
            </button>
          </div>
        </div>
      </header>

      {status === "loading" && !data && (
        <div className="loading-state">
          <div className="spinner" />
          <span>Scanning {data ? data.total_analyzed : "50+"} stocks…</span>
        </div>
      )}

      {status === "error" && (
        <div className="error-card">
          <div className="error-title">Failed to load predictions</div>
          <div className="error-detail">{error}</div>
          <button className="retry-btn" onClick={() => load()}>
            Retry
          </button>
        </div>
      )}

      {data && (
        <>
          <div className="tabs">
            <button
              className={`tab ${activeTab === "buy" ? "active buy-tab" : ""}`}
              onClick={() => setActiveTab("buy")}
            >
              Buy Candidates ({data.buy_candidates.length})
            </button>
            <button
              className={`tab ${activeTab === "sell" ? "active sell-tab" : ""}`}
              onClick={() => setActiveTab("sell")}
            >
              Sell / Short Candidates ({data.sell_candidates.length})
            </button>
          </div>

          {status === "loading" && (
            <div className="refreshing-banner">Refreshing data…</div>
          )}
          {data._fallback && (
            <div className="fallback-banner">
              Showing last saved scan — live fetch timed out. Hit Refresh to retry.
            </div>
          )}

          <div className="candidates-grid">
            {candidates?.map((cand, idx) => (
              <CandidateCard
                key={cand.symbol}
                cand={cand}
                rank={idx + 1}
                direction={activeTab}
              />
            ))}
            {candidates?.length === 0 && (
              <div className="empty-state">No candidates met the filter criteria.</div>
            )}
          </div>

          <div className="summary-stats">
            <div className="stat">
              <span className="stat-value">{data.total_analyzed}</span>
              <span className="stat-label">Analyzed</span>
            </div>
            <div className="stat">
              <span className="stat-value buy-color">{data.buy_candidates.length}</span>
              <span className="stat-label">Buy Signals</span>
            </div>
            <div className="stat">
              <span className="stat-value sell-color">{data.sell_candidates.length}</span>
              <span className="stat-label">Sell Signals</span>
            </div>
            <div className="stat">
              <span className="stat-value">{data.date}</span>
              <span className="stat-label">Analysis Date</span>
            </div>
          </div>

          <div className="methodology-note">
            Methodology: RSI 15 · MACD 15 · ADX 12 · Mom10d 18 · Mom21d 15 · Bollinger 10 · Volume 15 · Sector±5.
            Data: GCP Firestore pre-computed signals ({data.from_cache} cached) + live fetch ({data.live_fetched} symbols) + {data.industry_sectors_loaded} industry-return sectors.
            Stop = 2×ATR · Target = 3×ATR.
          </div>
        </>
      )}
    </div>
  );
}
