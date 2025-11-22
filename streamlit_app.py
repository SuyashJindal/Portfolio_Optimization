import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from optimizer import PortfolioManager, Optimizer

# --- Streamlit UI ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📊 Portfolio Optimizer (Streamlit-Only Deployment)")

# Sidebar Inputs
tickers = st.sidebar.text_input(
    "Tickers (comma separated)", 
    "AAPL,MSFT,GOOGL,AMZN,JPM,JNJ"
).replace(" ", "").split(",")

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

method = st.sidebar.selectbox(
    "Optimization Method",
    ["mvo_sharpe", "mvo_min_vol", "cvar", "risk_parity", 
     "tracking_error", "info_ratio", "kelly", "sortino", 
     "omega", "max_drawdown"]
)

benchmark = st.sidebar.text_input("Benchmark (optional)", "")

rf = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.2, 0.02)
mar = st.sidebar.number_input("MAR", -0.2, 0.5, 0.0)
conf = st.sidebar.slider("Confidence Level", 0.8, 0.99, 0.95)

min_w = st.sidebar.number_input("Min Weight", -1.0, 1.0, 0.0)
max_w = st.sidebar.number_input("Max Weight", 0.0, 1.0, 1.0)
sum_is_one = st.sidebar.checkbox("Weights Sum to 1", True)

if st.sidebar.button("🚀 Optimize"):

    st.info("Fetching data...")

    df, bench = PortfolioManager.get_data(
        tickers, str(start_date), str(end_date),
        benchmark if benchmark != "" else None
    )

    if df.empty:
        st.error("No price data found.")
        st.stop()

    returns = df.pct_change().dropna()

    benchmark_returns = None
    if bench is not None:
        benchmark_returns = bench.pct_change().dropna()

    constraints = {
        "min_weight": min_w,
        "max_weight": max_w,
        "sum_is_one": sum_is_one
    }

    optimizer = Optimizer(
        returns=returns,
        constraints=constraints,
        risk_free_rate=rf,
        mar=mar,
        benchmark_returns=benchmark_returns
    )

    st.info("Running optimizer...")

    weights = optimizer.optimize(method, conf)

    st.success("Optimization complete!")

    # ------- Display Results -------
    st.subheader("🔢 Portfolio Weights")

    df_weights = pd.DataFrame({
        "Asset": list(weights.keys()),
        "Weight": list(weights.values())
    })

    st.dataframe(df_weights)

    # Metrics
    st.subheader("📈 Portfolio Metrics")

    w_arr = np.array([weights[t] for t in df.columns])

    metrics = PortfolioManager.get_metrics(
        w_arr, returns, rf, mar, benchmark_returns, conf
    )

    st.json(metrics)

    # Chart
    st.subheader("📉 Cumulative Returns")

    port_rets = returns.dot(w_arr)
    port_cum = (1 + port_rets).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values,
                             mode='lines', name='Portfolio'))

    st.plotly_chart(fig, use_container_width=True)

