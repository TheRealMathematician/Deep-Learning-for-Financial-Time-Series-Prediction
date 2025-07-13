# 🧠 Deep Learning for Financial Time Series Prediction

## 📘 Background & Overview

This project investigates the power of deep learning (DL) in forecasting financial time series across diverse asset classes:

* 📈 **Cryptocurrency**: Bitcoin (BTC-USD)
* 📊 **Equity**: META Stock
* 🏛️ **Fixed Income**: 10-Year US Treasury Yields

We implement and compare three deep learning models:

* 🤖 **Multi-Layer Perceptron (MLP)**
* 🔁 **Long Short-Term Memory (LSTM)**
* 🧩 **Convolutional Neural Networks (CNN)** with **Gramian Angular Fields (GAF)**

The core goals are:

1. 🎯 Accurate price movement prediction
2. ✅ Reliable backtesting using walk-forward validation
3. 🧼 Reducing data leakage for real-world deployment

---

## 📂 Data Structure Overview

### 💾 Datasets

| Asset      | Time Period           | Source        | Key Traits                     |
| ---------- | --------------------- | ------------- | ------------------------------ |
| Bitcoin    | 6 months (daily)      | Yahoo Finance | High volatility                |
| META Stock | 1 year (daily/weekly) | Yahoo Finance | Bullish trend with corrections |
| 10Y Bonds  | 5 years (daily)       | Yahoo Finance | Smooth with crisis spikes      |

### 🔧 Preprocessing Pipeline

* **Log Return Calculation**: Stationarizing prices
* **Train-Test Splits**:

  * 70/30 baseline
  * Walk-forward (500 train / 100 test)
  * Walk-forward + 50-day buffer (leakage mitigation)
* **Normalization**: Min-max scaling
* **Target**: 1-day ahead log returns

---

## 🚀 Executive Summary

### 🔍 Key Findings

1. **Best Models by Asset**

   * 🧠 **LSTM** captured temporal dependencies for BTC & Bonds
   * 🖼️ **CNN-GAF** handled META’s volatility patterns
   * ⚡ MLP was fast but less accurate on sequential data

2. **Backtesting Results**

   * Walk-forward validation improved robustness:

     * MSE reduced from **9.1–11.3** to **0.5–4.1**
   * 50-observation buffer cut leakage and lowered MSE by **\~15%**

3. **Trading Strategy Performance**

| Model   | Sharpe Ratio (BTC) | Max Drawdown (META)     |
| ------- | ------------------ | ----------------------- |
| MLP     | 0.71               | -30.8%                  |
| LSTM    | 0.91               | -6.8%                   |
| CNN-GAF | **1.02**           | **+8.8%** (vs baseline) |

4. **Model Insights**

   * CNN-GAF visual encoding boosts volatility detection 📊
   * LSTM models longer-term dependencies 🌊
   * MLP is a decent benchmark but underfits volatile assets 🚧

---

## 🔬 Insights Deep Dive

### 📈 Bitcoin

* 🥇 LSTM = lowest MSE (0.0011), strongest long-memory performance
* 🧠 CNN-GAF captured volatility clusters visually

### 📉 META Stock

* 📊 CNN-GAF excelled at detecting reversals
* 🛡️ Leak mitigation essential: early MLP models overfit massively

### 💸 Treasury Yields

* LSTM best captured yield curve trends
* CNN overfit due to smoother target

---

## 🧩 Final Recommendations

### ✅ Model Selection

| Use Case           | Recommended Model |
| ------------------ | ----------------- |
| Trend detection    | LSTM              |
| Volatility/Pattern | CNN-GAF           |
| Low-compute needs  | MLP               |

### 🧪 Testing Framework

* Use **non-anchored walk-forward splits**
* Insert **buffer gaps** to avoid data leakage
* Test across **multiple horizons** (100–500 obs)

### 🛠️ Deployment Tips

* 🕓 Retrain weekly for crypto, quarterly for equities
* 🔔 Trigger model retraining based on drift in performance
* 📉 Monitor Sharpe ratio and max drawdown continuously

---

## 🧰 Repository Structure

```bash
├── data/                # Raw and processed datasets
├── notebooks/           # MLP, LSTM, CNN-GAF development
├── models/              # Saved model weights
├── results/             # Backtesting plots, metrics
└── README.md            # Project overview (this file)
```

---

## 📦 Dependencies

```bash
Python 3.8+
pip install -r requirements.txt
```

Main Libraries:

* `TensorFlow`
* `scikit-learn`
* `yfinance`
* `pandas`, `numpy`
* `matplotlib`, `seaborn`

---

## 👨‍💻 Contributors

* Tumelo Ranoto 🇿🇦
* Wycliffe Kipkoech Cheruiyot 🇰🇪
* Azeez Arisekola Sebiotimo 🇳🇬

🔬 MScFE 642 — Deep Learning for Finance
📜 License: MIT

---

## 🌟 Future Enhancements

1. Integrate transformer models (e.g., **Time Series BERT**)
2. Deploy models with **TensorFlow Serving** or **FastAPI**
3. Use **SHAP** for explainability
4. Add **multi-asset allocation strategy** using predicted returns
5. Combine GAF with **attention mechanisms**

---
