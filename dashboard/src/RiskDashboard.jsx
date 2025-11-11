import React, { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function RiskDashboard() {
  // ---------------------- HEADLINES ----------------------
  const [headlines, setHeadlines] = useState([]);
  useEffect(() => {
    async function fetchHeadlines() {
      const res = await fetch("https://jsonplaceholder.typicode.com/posts?_limit=5");
      const data = await res.json();
      setHeadlines(data.map(d => d.title));
    }
    fetchHeadlines();
  }, []);

  // ---------------------- CHART DATA ----------------------
  const [chartData, setChartData] = useState([]);
  useEffect(() => {
    async function fetchChart() {
      const res = await fetch("https://jsonplaceholder.typicode.com/comments?_limit=6");
      const data = await res.json();
      const parsed = data.map((d, idx) => ({ name: `P${idx}`, value: (idx + 1) * 10 }));
      parsed.push({ name: "Pred", value: 85 });
      setChartData(parsed);
    }
    fetchChart();
  }, []);

  // ---------------------- CHAT / RAG PANEL ----------------------
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  async function sendMessage() {
    if (!input.trim()) return;

    const userMsg = { sender: "user", text: input };
    setMessages(prev => [...prev, userMsg]);

    const res = await fetch("https://jsonplaceholder.typicode.com/posts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: input })
    });
    const data = await res.json();

    const botMsg = { sender: "bot", text: data.title || "Sample response from API" };
    setMessages(prev => [...prev, botMsg]);

    setInput("");
  }

  return (
    <div className="w-screen h-screen bg-black text-white p-6 grid grid-cols-2 gap-4">

      {/* ---------------- CHART PANEL ---------------- */}
      <div className="bg-gray-900 rounded-2xl p-4 relative flex flex-col items-center">
        <h2 className="text-xl font-bold mb-4">Stock Price Forecast</h2>
        <div className="w-full h-full bg-gray-800 rounded-xl p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <XAxis dataKey="name" stroke="#aaa" />
              <YAxis stroke="#aaa" />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#00ff99" strokeWidth={3} dot />
            </LineChart>
          </ResponsiveContainer>
        </div>
        {/* ---------------- HEADLINES PANEL ---------------- */}
        <div className="bg-gray-900 rounded-2xl p-4 flex flex-col">
            <h2 className="text-xl font-bold mb-4">Headlines</h2>
            <div className="flex-1 overflow-y-auto space-y-2">
            {headlines.map((h, i) => (
                <div key={i} className="bg-gray-800 p-3 rounded-xl">{h}</div>
            ))}
            </div>
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
                m.sender === "user" ? "bg-blue-600 self-end" : "bg-green-600 self-start"
              }`}
            >
              {m.text}
            </div>
          ))}
        </div>

        <div className="mt-4 flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 bg-gray-800 rounded-xl p-3"
            placeholder="Type your query..."
          />
          <button
            onClick={sendMessage}
            className="bg-blue-600 hover:bg-blue-700 rounded-xl px-4 py-2 font-semibold"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}