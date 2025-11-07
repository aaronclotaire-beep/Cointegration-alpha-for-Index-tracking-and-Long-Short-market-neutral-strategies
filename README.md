# Cointegration-alpha-for-Index-tracking-and-Long-Short-market-neutral-strategies

This project implements two main quantitative equity strategies based on **cointegration**:

1. **Index Tracking using Cointegration**  
2. **Market-Neutral Long-Short Strategy**

The objective is to replicate and extend the methodology originally developed for the **DJIA**, now applied to the **CAC 40**. Both approaches rely on long-term equilibrium relationships between stock prices and the index, rather than on short-term correlations.

---

## 1. Index Tracking via Cointegration

### Concept
Instead of using simple correlation, the index-tracking strategy exploits **cointegration** between the index and the prices of its component stocks. This approach offers several advantages:

- The **tracking error** is mean-reverting by construction.  
- Portfolio **weights are more stable**, reducing rebalancing needs.  
- It captures **long-term information** contained in asset price levels.

---

### Methodology

**Stock selection**  
A subset of index components is chosen (e.g., top 10–30 stocks) based on statistical or economic criteria.

**Weight estimation**  
Portfolio weights are obtained from an OLS regression of the log-index on the log-prices of selected stocks:

$$
\log(\text{Index}_t) = c_0 + \sum_{k=1}^N c_k \log(P_{k,t}) + \varepsilon_t,
$$

where the estimated coefficients $c_k$ are normalized and used as portfolio weights $w_k$.

**Validation & backtesting**
- Use the **Engle–Granger** procedure (ADF test on residuals) to confirm cointegration.
- Backtest with rolling calibration windows (e.g., 1–5 years).
- Rebalance every 10 trading days to limit transaction costs.

**Performance metrics**
- Annualized volatility and tracking error  
- Correlation with the benchmark  
- Skewness and kurtosis of tracking residuals  
- Sharpe and Information Ratios

---

## 2. Long-Short Strategy Based on Cointegration

### Concept
Extend index tracking to a **self-financing long-short framework** by constructing two cointegrated portfolios:

- A **“plus”** portfolio tracking an index adjusted by $+x\%$.  
- A **“minus”** portfolio tracking an index adjusted by $-x\%$.

The long-short position captures the return differential between the two while aiming to remain market-neutral.

---

### Methodology

Estimate two cointegrating regressions:

$$
\log(\text{Index}_t^{+}) = a_0 + \sum_{k=1}^N a_k \log(P_{k,t}) + u_t,
$$

$$
\log(\text{Index}_t^{-}) = b_0 + \sum_{k=1}^N b_k \log(P_{k,t}) + v_t.
$$

Define the long-short portfolio weights as:

\[
w_k = w_k^{+} - w_k^{-},
\]

which yields a self-financing, (approximately) market-neutral exposure.
