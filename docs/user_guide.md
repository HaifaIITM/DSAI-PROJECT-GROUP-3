# **User Documentation (Non-Technical Guide)**

---

# **1. App Overview**

This application provides **AI-powered financial forecasts**, **next-day to 20-day return predictions**, and **news-aware insights** for a selected stock or index (e.g., SPY). It is designed for **non-technical users**, including investors, analysts, and students who want an easy interface to:

* View **forecasted returns** for h1 (1-day), h5 (5-day), and h20 (20-day) horizons
* Understand the **latest AI-generated buy/sell signal**
* Browse **recent news headlines** affecting the asset
* Ask the built-in **RAG Assistant** natural-language questions about the predictions

The system uses a hybrid ESN–Ridge model to generate numerical predictions and an NLP engine to interpret news and answer questions.

---

# **2. Use Cases**

The system is suitable for:

* **Insightful Investing**: Quickly assess price momentum and short-term market direction.
* **Market Monitoring**: Track how news headlines affect AI risk predictions.
* **Education**: Learn how advanced machine learning forecasting works.
* **Decision Support**: Use directional signals (Buy/Sell) as part of a broader analysis.

---

# **3. App Layout & Output Description**

Below is a reference to your provided screenshot.
The interface contains the following sections:

---

## **3.1 Symbol Panel (Top Left)**

Displays the **currently loaded asset** (e.g., SPY).

**What the user sees:**

* The ticker symbol
* The asset being analyzed

---

## **3.2 Latest H20 Signal (Top Middle)**

Shows the **AI’s most confident long-horizon trading recommendation**, including:

* **BUY / SELL / HOLD**
* The predicted 20-day return (e.g., 0.47%)
* The current closing price

This helps users quickly understand the system's strongest signal.

---

## **3.3 Last Updated Timestamp (Top Right)**

Shows the most recent timestamp when:

* Data was fetched
* Forecasts were generated

Useful to verify recency.

---

## **3.4 Forecast Chart (Center Left)**

A dynamic line chart displaying **predicted returns** for:

* **h1** → 1-day prediction
* **h5** → 5-day prediction
* **h20** → 20-day prediction

Each horizon is color-coded.
Users can visually compare how short-term vs long-term predictions differ.

---

## **3.5 Recent Headlines (Bottom Left)**

Real news headlines fetched from market-relevant sources.

Each item shows:

* Headline text
* Publisher
* Timestamp
* “View Source” link

These headlines influence the risk index and predictions.

---

## **3.6 RAG Question Panel (Right Section)**

Users can ask natural-language questions like:

* “Why is the model predicting a BUY?”
* “How did recent news affect SPY?”
* “Explain the h20 forecast trend.”
* “Is volatility increasing?”

The assistant responds using:

* Forecast data
* News sentiment
* Stored model artifacts

---

## **3.7 Input Box (Bottom Right)**

Users type their question here and click **Send**.

---

# **4. How to Use the App (Step-by-Step)**

### **Step 1 — Launch the Application**

Open the deployed web app via your browser or internal URL.

---

### **Step 2 — Select or Confirm the Ticker**

The default asset (SPY) loads automatically.
Some versions may allow selecting a new symbol from a dropdown.

---

### **Step 3 — Review the Forecast Signals**

Check the top-middle box:

* If it shows **BUY**, the model forecasts upward movement
* **SELL** means downward trend expected
* **HOLD** means mixed or weak signals

---

### **Step 4 — Observe Multi-Horizon Forecast Chart**

Interpret the short vs. long horizon predictions visually:

* **h1 (1-day)** small or zero movement
* **h5 (5-day)** moderate movement
* **h20 (20-day)** larger trend (usually more predictive)

The chart helps identify:

* Trend acceleration
* Momentum
* Mean-reversion possibilities

---

### **Step 5 — Read Recent Market Headlines**

Scroll through the headlines to understand the **context** behind predictions:

* Positive earnings news → bullish signals
* Economic slowdown → bearish sentiment

---

### **Step 6 — Ask the RAG Assistant**

Use the panel on the right to ask questions like:

**Example Questions:**

* “Why is SPY showing a BUY signal today?”
* “Summarize the last 3 headlines.”
* “What is the risk index based on news?”
* “How confident is the h20 prediction?”
* “Explain the shape of the prediction chart.”

Press **Send**, and the assistant will respond with a detailed explanation.

---

# **5. Screenshot Reference**

![WhatsApp Image 2025-11-13 at 13 52 30_f9fca9d2](https://github.com/user-attachments/assets/b5f8fd28-a827-470a-ad3a-9a8a67e4453c)


Your app contains:

* Symbol: SPY
* Latest H20 Signal: BUY
* Prediction Chart: 1-day, 5-day, 20-day returns
* Recent Headlines: list with timestamps
* RAG Assistant panel
* Query input box

---

# **6. Summary**

This app is designed to give **everyday users**—not just technical ML experts—clear insights into:

* Where the market might move
* Which news stories matter
* Why the AI model is giving a specific signal

It is intuitive, fast, and built around transparency using explainable AI.

