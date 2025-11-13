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

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

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
    <div className="w-screen h-screen bg-black text-white p-6 grid grid-cols-2 gap-4">
      {/* ---------------- METRIC SUMMARY ---------------- */}
      <div className="col-span-2 grid grid-cols-3 gap-4 mb-2">
        <div className="bg-gray-900 rounded-2xl p-4">
          <h3 className="text-sm uppercase text-gray-400 mb-1">Symbol</h3>
          <p className="text-2xl font-bold">{symbol}</p>
        </div>
        <div className="bg-gray-900 rounded-2xl p-4">
          <h3 className="text-sm uppercase text-gray-400 mb-1">Latest h20 Signal</h3>
          {latestPrediction ? (
            <div>
              <p className="text-2xl font-bold">{latestPrediction.h20_signal}</p>
              <p className="text-gray-400 text-sm">
                {`Prediction: ${(latestPrediction.h20_prediction * 100).toFixed(2)}%`}
              </p>
              <p className="text-gray-400 text-sm">
                Close: ${latestPrediction.actual_close.toFixed(2)}
              </p>
            </div>
          ) : (
            <p className="text-gray-500">No data</p>
          )}
        </div>
        <div className="bg-gray-900 rounded-2xl p-4">
          <h3 className="text-sm uppercase text-gray-400 mb-1">Last Updated</h3>
          <p className="text-lg font-bold">
            {generatedAt ? new Date(generatedAt).toLocaleString() : "—"}
          </p>
        </div>
      </div>

      {/* ---------------- CHART PANEL ---------------- */}
      <div className="bg-gray-900 rounded-2xl p-4 relative flex flex-col">
        <h2 className="text-xl font-bold mb-4">Return Forecasts (Predictions)</h2>
        <div className="w-full h-72 bg-gray-800 rounded-xl p-4">
          {loading ? (
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              Loading predictions...
            </div>
          ) : error ? (
            <div className="w-full h-full flex items-center justify-center text-red-400">
              {error}
            </div>
          ) : chartData.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="date" stroke="#aaa" />
                <YAxis
                  stroke="#aaa"
                  tickFormatter={(value) => `${value.toFixed(2)}%`}
                  domain={["auto", "auto"]}
                />
                <Tooltip
                  formatter={(value) => [`${value.toFixed(3)}%`, ""]}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Legend />
                <Line type="monotone" dataKey="h1" stroke="#60a5fa" strokeWidth={2} dot />
                <Line type="monotone" dataKey="h5" stroke="#f59e0b" strokeWidth={2} dot />
                <Line type="monotone" dataKey="h20" stroke="#10b981" strokeWidth={3} dot />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              No prediction data available.
            </div>
          )}
        </div>
        {/* ---------------- HEADLINES PANEL ---------------- */}
        <div className="bg-gray-900 rounded-2xl p-4 flex flex-col mt-4">
          <h2 className="text-xl font-bold mb-4">Recent Headlines</h2>
          {loading && !news.length ? (
            <div className="text-gray-400">Loading headlines...</div>
          ) : news.length ? (
            <div className="flex-1 overflow-y-auto space-y-3 max-h-64 pr-1">
              {news.map((item, idx) => (
                <div key={`${item.title}-${idx}`} className="bg-gray-800 p-3 rounded-xl">
                  <p className="font-semibold">{item.title}</p>
                  <p className="text-sm text-gray-400 mt-1">{item.publisher ?? "Unknown"}</p>
                  <p className="text-xs text-gray-500">
                    {item.date ? new Date(item.date).toLocaleString() : ""}
                  </p>
                  {item.link && (
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noreferrer"
                      className="text-blue-400 text-xs mt-2 inline-block"
                    >
                      View source
                    </a>
                  )}
                </div>
              ))}
            </div>
          ) : error ? (
            <div className="text-red-400">{error}</div>
          ) : (
            <div className="text-gray-400">No recent headlines available.</div>
          )}
        </div>
      </div>

      {/* ---------------- RAG CHAT PANEL ---------------- */}
      <div className="bg-gray-900 rounded-2xl p-4 flex flex-col">
        <h2 className="text-xl font-bold mb-4">Post your questions here!</h2>
        <div className="flex-1 bg-gray-800 rounded-xl p-3 overflow-y-auto space-y-3">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`max-w-[80%] p-3 rounded-xl ${
                m.sender === "user" ? "bg-blue-600 self-end ml-auto" : "bg-green-600 self-start"
              }`}
            >
              {m.text.split("\n").map((line, lineIdx) => (
                <React.Fragment key={lineIdx}>
                  {line}
                  <br />
                </React.Fragment>
              ))}
            </div>
          ))}
          {!messages.length && (
            <div className="text-gray-400 text-sm">Ask the RAG assistant about the predictions.</div>
          )}
        </div>

        {chatError && (
          <div className="mt-2 text-sm text-red-400">Chat error: {chatError}</div>
        )}

        <div className="mt-4 flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-gray-800 rounded-xl p-3"
            placeholder="Type your query..."
          />
          <button
            onClick={sendMessage}
            disabled={isSending}
            className={`rounded-xl px-4 py-2 font-semibold ${
              isSending ? "bg-blue-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {isSending ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}