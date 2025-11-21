import React, { useState, useEffect, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
} from "recharts";
import { marked } from "marked";
import DOMPurify from "dompurify";

// Auto-detect API URL for HF Spaces
const getApiBaseUrl = () => {
  // Check if we're in production (HF Spaces)
  if (import.meta.env.PROD) {
    // In production, use same origin (HF Spaces serves both frontend and backend)
    return window.location.origin;
  }
  // Development: use env var or default
  return import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
};

const API_BASE_URL = getApiBaseUrl();

const horizonLabels = {
  h1: "1-Day",
  h5: "5-Day",
  h20: "20-Day",
};

const horizonColors = {
  h1: "#60a5fa",
  h5: "#f59e0b",
  h20: "#10b981",
};

marked.setOptions({
  breaks: true,
  gfm: true,
});

const renderMarkdown = (text) => {
  if (!text) return "";
  const raw = marked.parse(text);
  return DOMPurify.sanitize(raw);
};

export default function RiskDashboard() {
  // ---------------------- PREDICTIONS / NEWS ----------------------
  const [predictions, setPredictions] = useState([]);
  const [news, setNews] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [symbol, setSymbol] = useState("SPY");
  const [generatedAt, setGeneratedAt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    async function fetchPredictions() {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`${API_BASE_URL}/predict`);
        if (!res.ok) {
          throw new Error(`Predict endpoint failed (${res.status})`);
        }
        const payload = await res.json();
        if (!isMounted) return;

        const preds = payload?.predictions ?? [];
        const newsItems = payload?.recent_news ?? [];

        setSymbol(payload?.symbol ?? "SPY");
        setGeneratedAt(payload?.generated_at ?? null);
        setPredictions(preds);
        setNews(newsItems);

        const formattedChartData = preds.map((item) => ({
          date: item.date,
          h1: item.h1_prediction * 100,
          h5: item.h5_prediction * 100,
          h20: item.h20_prediction * 100,
        }));
        setChartData(formattedChartData);
      } catch (err) {
        if (isMounted) {
          setError(err.message);
          setPredictions([]);
          setNews([]);
          setChartData([]);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    fetchPredictions();

    return () => {
      isMounted = false;
    };
  }, []);

  const latestPrediction = useMemo(
    () => (predictions.length ? predictions[predictions.length - 1] : null),
    [predictions]
  );

  const horizonSummary = useMemo(() => {
    if (!predictions.length) return [];
    const latest = predictions[predictions.length - 1];

    return ["h1", "h5", "h20"].map((key) => {
      const prediction = latest?.[`${key}_prediction`] ?? 0;
      const signal = latest?.[`${key}_signal`] ?? "N/A";
      return {
        key,
        label: horizonLabels[key],
        signal,
        prediction,
        color: horizonColors[key],
        intensity: Math.abs(prediction) * 100,
      };
    });
  }, [predictions]);

  // ---------------------- CHAT / RAG PANEL ----------------------
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [chatError, setChatError] = useState(null);
  const [isSending, setIsSending] = useState(false);

  async function sendMessage() {
    if (!input.trim()) return;

    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setChatError(null);
    setIsSending(true);

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input.trim(), top_k: 3 }),
      });

      if (!res.ok) {
        throw new Error(`Chat endpoint failed (${res.status})`);
      }

      const data = await res.json();
      const summary = data?.answer ?? "No answer received.";

      const context = (data?.context ?? [])
        .map((doc) => `• ${doc.title}`)
        .join("\n")
        .trim();

      const combinedText = context ? `${summary}\n\nContext:\n${context}` : summary;

      const botMsg = { sender: "bot", text: combinedText };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setChatError(err.message);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `Error processing request: ${err.message}` },
      ]);
    } finally {
      setInput("");
      setIsSending(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="h-screen w-full overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-black text-slate-100 p-4 md:p-6">
      <div className="h-full flex flex-col gap-4 md:gap-6">
        {/* ---------------- HEADER SUMMARY ---------------- */}
        <div className="grid grid-cols-12 gap-4 md:gap-6 flex-none">
          <div className="col-span-12 md:col-span-4 bg-slate-900/80 border border-slate-800 rounded-2xl p-4 md:p-5 shadow-lg shadow-slate-950/40">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-3">Symbol</p>
            <p className="text-4xl font-semibold tracking-wider">{symbol}</p>
          </div>
          <div className="col-span-12 md:col-span-4 bg-slate-900/80 border border-slate-800 rounded-2xl p-4 md:p-5 shadow-lg shadow-slate-950/40">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-3">Latest h20 Signal</p>
            {latestPrediction ? (
              <div className="space-y-1">
                <p className="text-3xl font-semibold">{latestPrediction.h20_signal}</p>
                <p className="text-sm text-slate-400">
                  {(latestPrediction.h20_prediction * 100).toFixed(2)}% · Close $
                  {latestPrediction.actual_close.toFixed(2)}
                </p>
                <span className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-emerald-300 bg-emerald-500/10 px-3 py-1 rounded-full">
                  Trend
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-300 animate-pulse" />
                </span>
              </div>
            ) : (
              <p className="text-slate-500 text-sm">No data available.</p>
            )}
          </div>

          <div className="col-span-12 md:col-span-4 bg-slate-900/80 border border-slate-800 rounded-2xl p-4 md:p-5 shadow-lg shadow-slate-950/40">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-3">Last Updated</p>
            <p className="text-xl font-semibold">
              {generatedAt ? new Date(generatedAt).toLocaleString() : "Awaiting data"}
            </p>
          </div>
        </div>

        {/* ---------------- HORIZON SUMMARY ---------------- */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-none overflow-x-auto md:overflow-visible pb-1">
          {horizonSummary.map(({ key, label, signal, prediction, color, intensity }) => (
            <div
              key={key}
              className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/70 p-4 md:p-5 shadow-lg shadow-slate-950/30 min-w-[240px]"
            >
              <div
                className="absolute inset-x-0 top-0 h-1 opacity-60"
                style={{ background: `linear-gradient(90deg, transparent, ${color}, transparent)` }}
              />
              <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-3">{label}</p>
              <p className="text-3xl font-semibold">
                {signal}
                <span className="ml-2 text-base text-slate-400">
                  {(prediction * 100).toFixed(2)}%
                </span>
              </p>
              <div className="mt-4">
                <div className="flex justify-between text-xs text-slate-500 mb-1">
                  <span>Momentum</span>
                  <span>{intensity.toFixed(2)}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min(intensity, 100)}%`,
                      background: color,
                    }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 md:gap-6 flex-1 min-h-0">
          {/* ---------------- CHART PANEL ---------------- */}
          <div className="xl:col-span-2 bg-slate-900/80 border border-slate-800 rounded-3xl p-4 md:p-6 shadow-xl shadow-slate-950/40 flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-4 flex-none">
              <div>
                <h2 className="text-2xl font-semibold tracking-tight">Return Forecasts</h2>
                <p className="text-sm text-slate-400">Predicted % change · positive → BUY bias</p>
              </div>
              <div className="flex items-center gap-3 text-xs uppercase tracking-[0.3em] text-slate-500">
                <span className="flex items-center gap-2">
                  <span className="h-2 w-4 rounded bg-[#60a5fa]" />
                  h1
                </span>
                <span className="flex items-center gap-2">
                  <span className="h-2 w-4 rounded bg-[#f59e0b]" />
                  h5
                </span>
                <span className="flex items-center gap-2">
                  <span className="h-2 w-4 rounded bg-[#10b981]" />
                  h20
                </span>
              </div>
            </div>

            <div className="w-full flex-1 min-h-[220px] bg-slate-950/60 rounded-2xl border border-slate-800/60 backdrop-blur">
              {loading ? (
                <div className="w-full h-full flex flex-col items-center justify-center text-slate-400 gap-2">
                  <div className="h-10 w-10 rounded-full border-2 border-slate-700 border-t-emerald-400 animate-spin" />
                  Fetching predictions...
                </div>
              ) : error ? (
                <div className="w-full h-full flex items-center justify-center text-red-400">
                  {error}
                </div>
              ) : chartData.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="date" stroke="#94a3b8" />
                    <YAxis
                      stroke="#94a3b8"
                      tickFormatter={(value) => `${value.toFixed(2)}%`}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        border: "1px solid rgba(148, 163, 184, 0.2)",
                      }}
                      formatter={(value) => [`${value.toFixed(3)}%`, "Return"]}
                      labelFormatter={(label) => `Date: ${label}`}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="h1" stroke={horizonColors.h1} strokeWidth={2} dot />
                    <Line type="monotone" dataKey="h5" stroke={horizonColors.h5} strokeWidth={2} dot />
                    <Line
                      type="monotone"
                      dataKey="h20"
                      stroke={horizonColors.h20}
                      strokeWidth={3}
                      dot
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-slate-400">
                  No prediction data available.
                </div>
              )}
            </div>

            {/* ---------------- HEADLINES PANEL ---------------- */}
            <div className="mt-6 flex-1 flex flex-col min-h-0">
              <div className="flex items-center justify-between mb-3 flex-none">
                <h2 className="text-xl font-semibold tracking-tight">Recent Headlines</h2>
                <span className="text-xs uppercase tracking-[0.3em] text-slate-500">
                  sentiment cues
                </span>
              </div>

              <div className="grid lg:grid-cols-2 gap-4 flex-1 min-h-0 overflow-y-auto pr-2">
                {loading && !news.length ? (
                  <div className="text-slate-400 text-sm">Loading headlines...</div>
                ) : news.length ? (
                  news.map((item, idx) => (
                    <div
                      key={`${item.title}-${idx}`}
                      className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4 transition hover:border-emerald-500/40 hover:shadow-lg hover:shadow-emerald-500/10"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <h3 className="text-base font-semibold leading-snug">{item.title}</h3>
                        <span className="text-[10px] uppercase tracking-[0.3em] text-emerald-300">
                          {item.publisher ?? "Source"}
                        </span>
                      </div>
                      <p className="text-xs text-slate-500 mt-2">
                        {item.date ? new Date(item.date).toLocaleString() : ""}
                      </p>
                      {item.link && (
                        <a
                          href={item.link}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-2 text-xs text-emerald-300 mt-3 hover:underline"
                        >
                          View source →
                        </a>
                      )}
                    </div>
                  ))
                ) : error ? (
                  <div className="text-red-400">{error}</div>
                ) : (
                  <div className="text-slate-400">No recent headlines available.</div>
                )}
              </div>
            </div>
          </div>

          {/* ---------------- RAG CHAT PANEL ---------------- */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-4 md:p-6 shadow-xl shadow-slate-950/40 flex flex-col min-h-0">
            <div className="mb-4 flex-none">
              <h2 className="text-2xl font-semibold tracking-tight">Post your questions here!</h2>
              <p className="text-sm text-slate-400">
                Ask about signals, risk posture, or key headlines. Responses come from the RAG layer.
              </p>
            </div>

            <div className="flex-1 bg-slate-950/60 border border-slate-800/70 rounded-2xl p-4 overflow-y-auto space-y-3 min-h-0">
            {messages.map((m, idx) => {
              const html = renderMarkdown(m.text);
              const isUser = m.sender === "user";
              return (
                <div
                  key={idx}
                  className={`max-w-[85%] p-4 rounded-2xl leading-relaxed tracking-wide shadow-md ${
                    isUser
                      ? "bg-blue-600/80 border border-blue-400/40 self-end ml-auto"
                      : "bg-emerald-600/20 border border-emerald-400/40 self-start"
                  }`}
                  dangerouslySetInnerHTML={{ __html: html }}
                />
              );
            })}
              {!messages.length && (
                <div className="text-slate-500 text-sm">
                  Ask the assistant for context—e.g., “Summarize today’s h5 outlook with supporting
                  news.”
                </div>
              )}
            </div>

            {chatError && (
              <div className="mt-3 text-sm text-red-400 bg-red-500/10 border border-red-400/30 rounded-xl px-3 py-2 flex-none">
                Chat error: {chatError}
              </div>
            )}

            <div className="mt-4 flex gap-3 flex-none">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={2}
                className="flex-1 bg-slate-950/60 border border-slate-800 rounded-2xl p-3 resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                placeholder="Type your query... Press Enter to send"
              />
              <button
                onClick={sendMessage}
                disabled={isSending}
                className={`h-12 px-6 rounded-2xl font-semibold shadow-lg transition ${
                  isSending
                    ? "bg-emerald-500/40 text-emerald-100 cursor-not-allowed"
                    : "bg-emerald-500 hover:bg-emerald-400 text-slate-900"
                }`}
              >
                {isSending ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}