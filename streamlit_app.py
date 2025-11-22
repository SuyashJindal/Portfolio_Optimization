import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# === IMPORT CORE LOGIC DIRECTLY (No API) ===
# Ensure optimizer.py is in the same directory
from optimizer import PortfolioManager, Optimizer

st.set_page_config(
    layout="wide", 
    page_title="Advanced Portfolio Optimizer",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 10px;}
    .sub-header {font-size: 1.1rem; color: #666; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# === SIDEBAR: INPUTS ===
st.sidebar.title("⚙️ Configuration")

# 1. Assets
st.sidebar.subheader("📊 Portfolio Assets")
tickers_input = st.sidebar.text_area(
    "Tickers (comma separated)", 
    "AAPL,MSFT,GOOGL,AMZN,JPM,JNJ",
    help="Enter stock tickers separated by commas",
    height=80
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# 2. Date Range
st.sidebar.subheader("📅 Historical Period")
default_start = datetime.now() - timedelta(days=3*365)
default_end = datetime.now() - timedelta(days=1)

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Start", default_start)
with col_d2:
    end_date = st.date_input("End", default_end)

# 3. Method Selection
st.sidebar.subheader("🎯 Optimization Method")
method_map = {
    "Max Sharpe (MVO)": "mvo_sharpe",
    "Min Variance (MVO)": "mvo_min_vol",
    "CVaR Minimization": "cvar",
    "Risk Parity": "risk_parity",
    "Tracking Error Min": "tracking_error",
    "Information Ratio Max": "info_ratio",
    "Kelly Criterion": "kelly",
    "Sortino Ratio Max": "sortino",
    "Omega Ratio Max": "omega",
    "Min Max Drawdown": "max_drawdown"
}

method_label = st.sidebar.selectbox("Select Method", list(method_map.keys()))
method_key = method_map[method_label]

# 4. Parameters
st.sidebar.subheader("📈 Risk Parameters")
rf = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.0, 0.5) / 100
mar = st.sidebar.number_input("Min Acceptable Return - MAR (%)", -10.0, 50.0, 0.0, 1.0) / 100
conf = st.sidebar.slider("CVaR Confidence Level", 0.80, 0.99, 0.95, 0.01)

benchmark = None
if method_key in ["tracking_error", "info_ratio"]:
    benchmark = st.sidebar.text_input("Benchmark Ticker", "SPY").strip().upper()

# 5. Constraints
st.sidebar.subheader("🔒 Portfolio Constraints")
long_only = st.sidebar.checkbox("Long Only (No Shorts)", True)

col_w1, col_w2 = st.sidebar.columns(2)
with col_w1:
    min_w = st.number_input("Min Weight (%)", -100, 100, 0 if long_only else -50, 5) / 100
with col_w2:
    max_w = st.number_input("Max Weight (%)", 0, 200, 100, 5) / 100

sum_one = st.sidebar.checkbox("Weights Sum to 100%", True)

st.sidebar.divider()

# === MAIN PAGE ===
st.markdown('<p class="main-header">📊 Advanced Portfolio Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional multi-strategy asset allocation with real-time market data</p>', unsafe_allow_html=True)

# Info boxes
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("Method", method_label.split("(")[0].strip())
with col_info2:
    st.metric("Assets", len(tickers))
with col_info3:
    days_diff = (end_date - start_date).days
    st.metric("Period", f"{days_diff} days")
with col_info4:
    st.metric("Risk-Free", f"{rf*100:.1f}%")

st.divider()

# === RUN OPTIMIZATION ===
run_button = st.sidebar.button("🚀 RUN OPTIMIZATION", type="primary", use_container_width=True)

if run_button:
    # Validation
    errors = []
    if len(tickers) < 2: errors.append("⚠️ Enter at least 2 tickers")
    if start_date >= end_date: errors.append("⚠️ Start date must be before end date")
    if max_w < min_w: errors.append("⚠️ Max weight must be ≥ Min weight")
    
    if errors:
        for error in errors: st.error(error)
        st.stop()

    with st.spinner("🔄 Fetching data and optimizing portfolio..."):
        try:
            # 1. DIRECT DATA FETCH (Replaces API /load-data)
            df, bench_series = PortfolioManager.get_data(tickers, str(start_date), str(end_date), benchmark)
            returns = df.pct_change().dropna()
            bench_rets = bench_series.pct_change().dropna() if bench_series is not None else None

            if returns.empty:
                st.error("No data found for the given tickers/dates.")
                st.stop()

            # 2. OPTIMIZATION (Replaces API /optimize)
            constraints = {'min_weight': min_w, 'max_weight': max_w, 'sum_is_one': sum_one}
            opt = Optimizer(returns, constraints, rf, mar, bench_rets)
            
            weights = opt.optimize(method_key, conf)

            # 3. METRICS CALCULATION
            final_weights_arr = np.array([weights[t] for t in df.columns])
            metrics = PortfolioManager.get_metrics(
                final_weights_arr, returns, rf, mar, bench_rets, conf
            )
            
            # 4. CHART DATA PREPARATION
            port_cum_ret = (1 + returns.dot(final_weights_arr)).cumprod()
            chart_series = (port_cum_ret / port_cum_ret.iloc[0]) * 100
            
            # Calculate Risk Contributions if needed
            risk_contribs = {}
            if method_key == "risk_parity":
                cov = returns.cov() * 252
                port_vol = np.sqrt(np.dot(final_weights_arr.T, np.dot(cov, final_weights_arr)))
                mrc = np.dot(cov, final_weights_arr) / port_vol
                rc = final_weights_arr * mrc
                risk_contribs = dict(zip(df.columns, rc))

            st.success("✅ Optimization Complete!")
            
            # === VISUALIZATION LOGIC ===
            
            # Weights Table & Pie
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("💼 Optimized Portfolio Weights")
                df_weights = pd.DataFrame([
                    {"Ticker": t, "Weight": w, "Weight %": w * 100}
                    for t, w in sorted(weights.items(), key=lambda x: -x[1])
                ])
                st.dataframe(df_weights.style.format({"Weight": "{:.4f}", "Weight %": "{:.2f}%"})
                             .background_gradient(subset=['Weight %'], cmap='RdYlGn'), use_container_width=True)
            with col2:
                fig_pie = go.Figure(data=[go.Pie(labels=df_weights['Ticker'], values=df_weights['Weight'], hole=0.4)])
                fig_pie.update_layout(title="Asset Allocation", height=350)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            
            # Performance Metrics
            st.subheader("📊 Portfolio Performance Metrics")
            metric_cols = st.columns(4)
            for idx, (k, v) in enumerate(metrics.items()):
                with metric_cols[idx % 4]:
                    st.metric(k, f"{v:.4f}")
            
            st.divider()
            
            # Cumulative Returns Chart
            st.subheader("📈 Cumulative Portfolio Returns")
            fig_cumret = go.Figure()
            fig_cumret.add_trace(go.Scatter(x=chart_series.index, y=chart_series.values, mode='lines', 
                                            name='Portfolio Value', fill='tozeroy'))
            fig_cumret.update_layout(title="Portfolio Growth (Base 100)", height=400)
            fig_cumret.add_hline(y=100, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_cumret, use_container_width=True)
            
            st.divider()
            
            # Efficient Frontier (MVO Only) - Calculated Locally
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if "mvo" in method_key:
                    st.subheader("🎯 Efficient Frontier")
                    with st.spinner("Calculating frontier..."):
                        # Helper to calculate frontier points
                        frontier_vol, frontier_ret = [], []
                        
                        # 1. Find Min Vol and Max Sharpe range
                        w_min = np.array(list(opt.optimize("mvo_min_vol").values()))
                        ret_min = np.sum(opt.mean_rets * w_min)
                        
                        w_max = np.array(list(opt.optimize("mvo_sharpe").values()))
                        ret_max = np.sum(opt.mean_rets * w_max) * 1.2 # Extend slightly
                        
                        target_returns = np.linspace(ret_min, ret_max, 20)
                        
                        for target in target_returns:
                            opt.cons.append({'type': 'eq', 'fun': lambda x: np.sum(opt.mean_rets * x) - target})
                            try:
                                # Reuse min_vol optimizer with added constraint
                                from scipy.optimize import minimize
                                res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(opt.cov_matrix, w))), 
                                               opt._get_start_guess(), method='SLSQP', bounds=opt.bounds, constraints=opt.cons)
                                if res.success:
                                    frontier_vol.append(res.fun) # Volatility
                                    frontier_ret.append(target)  # Return
                            except: pass
                            opt.cons.pop() # Remove constraint
                            
                        fig_ef = go.Figure()
                        fig_ef.add_trace(go.Scatter(x=frontier_vol, y=frontier_ret, mode='lines', name='Efficient Frontier'))
                        fig_ef.add_trace(go.Scatter(x=[metrics['Volatility']], y=[metrics['Expected Return']], 
                                                    mode='markers', marker=dict(size=15, color='red', symbol='star'), 
                                                    name='Your Portfolio'))
                        fig_ef.update_layout(title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Return", height=450)
                        st.plotly_chart(fig_ef, use_container_width=True)

                elif method_key == "risk_parity" and risk_contribs:
                    st.subheader("⚖️ Risk Contributions")
                    fig_rc = go.Figure(data=[go.Bar(x=list(risk_contribs.keys()), y=list(risk_contribs.values()))])
                    fig_rc.update_layout(title="Risk Contribution by Asset", height=450)
                    st.plotly_chart(fig_rc, use_container_width=True)

            with viz_col2:
                st.subheader("📊 Returns Distribution")
                fig_hist = px.histogram(returns.dot(final_weights_arr), nbins=50, title="Daily Returns Distribution")
                fig_hist.update_layout(showlegend=False, height=450)
                st.plotly_chart(fig_hist, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
else:
    st.info("👈 Configure your portfolio in the sidebar and click 'RUN OPTIMIZATION'")
