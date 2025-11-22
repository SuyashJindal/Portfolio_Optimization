import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

API_URL = "http://127.0.0.1:8000"

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

st.sidebar.subheader("📅 Historical Period")
default_start = datetime.now() - timedelta(days=3*365)
default_end = datetime.now() - timedelta(days=1)

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input(
        "Start", 
        default_start,
        help="Data start date"
    )
with col_d2:
    end_date = st.date_input(
        "End", 
        default_end,
        help="Data end date"
    )
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

method_label = st.sidebar.selectbox(
    "Select Method", 
    list(method_map.keys()),
    help="Choose optimization objective"
)
method_key = method_map[method_label]
st.sidebar.subheader("📈 Risk Parameters")
rf = st.sidebar.number_input(
    "Risk-Free Rate (%)", 
    0.0, 20.0, 4.0, 0.5,
    help="Annual risk-free rate (e.g., 10-year T-Bill)"
) / 100

mar = st.sidebar.number_input(
    "Min Acceptable Return - MAR (%)", 
    -10.0, 50.0, 0.0, 1.0,
    help="For Sortino/Omega ratios"
) / 100

conf = st.sidebar.slider(
    "CVaR Confidence Level", 
    0.80, 0.99, 0.95, 0.01,
    help="Confidence level for CVaR"
)

# Benchmark for relative methods
benchmark = None
if method_key in ["tracking_error", "info_ratio"]:
    benchmark = st.sidebar.text_input(
        "Benchmark Ticker", 
        "SPY",
        help="Reference index"
    ).strip().upper()

# 5. Constraints
st.sidebar.subheader("🔒 Portfolio Constraints")
long_only = st.sidebar.checkbox(
    "Long Only (No Shorts)", 
    True,
    help="Restrict to positive weights"
)

col_w1, col_w2 = st.sidebar.columns(2)
with col_w1:
    min_w = st.number_input(
        "Min Weight (%)", 
        -100, 100, 
        0 if long_only else -50, 5
    ) / 100
with col_w2:
    max_w = st.number_input(
        "Max Weight (%)", 
        0, 200, 
        100, 5
    ) / 100

sum_one = st.sidebar.checkbox(
    "Weights Sum to 100%", 
    True,
    help="Full investment constraint"
)

frontier_points = 25  # Fixed for better visualization

st.sidebar.divider()
st.markdown('<p class="main-header">📊 Advanced Portfolio Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional multi-strategy asset allocation with real-time market data</p>', unsafe_allow_html=True)

# Info boxes
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("Method", method_label.split("(")[0].strip(), help="Optimization objective")
with col_info2:
    st.metric("Assets", len(tickers), help="Number of securities")
with col_info3:
    days_diff = (end_date - start_date).days
    st.metric("Period", f"{days_diff} days", help="Historical data length")
with col_info4:
    st.metric("Risk-Free", f"{rf*100:.1f}%", help="Benchmark rate")

st.divider()

# === RUN OPTIMIZATION ===
run_button = st.sidebar.button("🚀 RUN OPTIMIZATION", type="primary", use_container_width=True)

if run_button:
    
    # Validation
    errors = []
    if len(tickers) < 2:
        errors.append("⚠️ Enter at least 2 tickers")
    if start_date >= end_date:
        errors.append("⚠️ Start date must be before end date")
    if max_w < min_w:
        errors.append("⚠️ Max weight must be ≥ Min weight")
    
    if errors:
        for error in errors:
            st.error(error)
        st.stop()
    
    # Build request
    payload = {
        "tickers": tickers,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "method": method_key,
        "benchmark": benchmark if benchmark else None,
        "risk_free_rate": rf,
        "mar": mar,
        "confidence_level": conf,
        "min_weight": min_w,
        "max_weight": max_w,
        "sum_is_one": sum_one,
        "frontier_points": frontier_points
    }

    with st.spinner("🔄 Fetching data and optimizing portfolio..."):
        try:
            # === MAIN OPTIMIZATION ===
            resp = requests.post(f"{API_URL}/optimize", json=payload, timeout=120)
            
            if resp.status_code != 200:
                st.error(f"❌ API Error ({resp.status_code}): {resp.text}")
                st.stop()
            
            data = resp.json()
            weights = data['weights']
            metrics = data['metrics']
            chart_data_dict = data['chart_data']
            risk_contribs = data.get('risk_contributions', {})
            
            # Convert chart data dict to Series
            chart_dates = sorted(chart_data_dict.keys())
            chart_values = [chart_data_dict[d] for d in chart_dates]
            chart_series = pd.Series(chart_values, index=pd.to_datetime(chart_dates))
            
            st.success("✅ Optimization Complete!")
            
            # === SECTION 1: WEIGHTS TABLE ===
            st.subheader("💼 Optimized Portfolio Weights")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Weights DataFrame
                df_weights = pd.DataFrame([
                    {"Ticker": ticker, "Weight": weight, "Weight %": weight * 100}
                    for ticker, weight in sorted(weights.items(), key=lambda x: -x[1])
                ])
                
                # Display with formatting
                st.dataframe(
                    df_weights.style.format({
                        "Weight": "{:.4f}",
                        "Weight %": "{:.2f}%"
                    }).background_gradient(subset=['Weight %'], cmap='RdYlGn', vmin=0, vmax=df_weights['Weight %'].max()),
                    hide_index=True,
                    use_container_width=True,
                    height=300
                )
                
                # Download button
                csv = df_weights.to_csv(index=False)
                st.download_button(
                    "📥 Download Weights (CSV)",
                    csv,
                    f"portfolio_weights_{method_key}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=df_weights['Ticker'],
                    values=df_weights['Weight'],
                    hole=0.4,
                    marker=dict(line=dict(color='white', width=2))
                )])
                fig_pie.update_layout(
                    title="Asset Allocation",
                    height=350,
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            
            # === SECTION 2: PERFORMANCE METRICS ===
            st.subheader("📊 Portfolio Performance Metrics")
            
            # Create metrics grid
            metric_cols = st.columns(4)
            
            metric_items = list(metrics.items())
            for idx, (metric_name, metric_value) in enumerate(metric_items):
                col_idx = idx % 4
                with metric_cols[col_idx]:
                    # Format display
                    if "Return" in metric_name or "Volatility" in metric_name or "Tracking Error" in metric_name:
                        display_val = f"{metric_value*100:.2f}%"
                        delta = None
                    elif "Ratio" in metric_name:
                        display_val = f"{metric_value:.3f}"
                        delta = None
                    elif "Drawdown" in metric_name:
                        display_val = f"{metric_value*100:.2f}%"
                        delta = None
                    elif "CVaR" in metric_name:
                        display_val = f"{metric_value*100:.2f}%"
                        delta = None
                    else:
                        display_val = f"{metric_value:.4f}"
                        delta = None
                    
                    st.metric(metric_name, display_val, delta=delta)
            
            st.divider()
            
            # === SECTION 3: CUMULATIVE RETURNS ===
            st.subheader("📈 Cumulative Portfolio Returns")
            
            fig_cumret = go.Figure()
            fig_cumret.add_trace(go.Scatter(
                x=chart_series.index,
                y=chart_series.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ))
            
            fig_cumret.update_layout(
                title=f"Portfolio Growth (Base 100) - {data['dates']['start']} to {data['dates']['end']}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                hovermode='x unified',
                template='plotly_white',
                height=400,
                showlegend=True
            )
            
            fig_cumret.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_cumret, use_container_width=True)
            
            st.divider()
            
            # === SECTION 4: METHOD-SPECIFIC VISUALIZATIONS ===
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # === EFFICIENT FRONTIER (MVO Methods) ===
                if "mvo" in method_key:
                    st.subheader("🎯 Efficient Frontier")
                    
                    with st.spinner("Calculating efficient frontier..."):
                        try:
                            frontier_resp = requests.post(
                                f"{API_URL}/frontier", 
                                json=payload,
                                timeout=120
                            )
                            
                            if frontier_resp.status_code == 200:
                                f_data = frontier_resp.json()
                                
                                fig_ef = go.Figure()
                                
                                # Frontier curve
                                fig_ef.add_trace(go.Scatter(
                                    x=f_data['volatility'],
                                    y=f_data['return'],
                                    mode='lines',
                                    name='Efficient Frontier',
                                    line=dict(color='blue', width=3)
                                ))
                                
                                # Selected portfolio - STAR marker
                                fig_ef.add_trace(go.Scatter(
                                    x=[metrics['Volatility']],
                                    y=[metrics['Expected Return']],
                                    mode='markers',
                                    marker=dict(
                                        size=20, 
                                        color='red',
                                        symbol='star',
                                        line=dict(color='darkred', width=2)
                                    ),
                                    name='Selected Portfolio',
                                    hovertemplate='<b>Selected Portfolio</b><br>' +
                                                'Return: %{y:.2%}<br>' +
                                                'Volatility: %{x:.2%}<br>' +
                                                f'Sharpe: {metrics["Sharpe Ratio"]:.3f}<extra></extra>'
                                ))
                                
                                fig_ef.update_layout(
                                    title="Mean-Variance Efficient Frontier",
                                    xaxis_title="Volatility (Annualized)",
                                    yaxis_title="Expected Return (Annualized)",
                                    template='plotly_white',
                                    hovermode='closest',
                                    height=450,
                                    xaxis=dict(tickformat='.1%'),
                                    yaxis=dict(tickformat='.1%')
                                )
                                
                                st.plotly_chart(fig_ef, use_container_width=True)
                                st.caption(f"✨ Red star shows your optimized portfolio on the frontier ({f_data['num_points']} points)")
                            else:
                                st.warning(f"⚠️ Could not generate frontier: {frontier_resp.text}")
                        
                        except requests.exceptions.Timeout:
                            st.warning("⏱️ Frontier calculation timed out. Try fewer assets or shorter period.")
                        except Exception as e:
                            st.warning(f"⚠️ Frontier error: {str(e)}")
                
                # === RISK CONTRIBUTIONS (Risk Parity) ===
                elif method_key == "risk_parity" and risk_contribs:
                    st.subheader("⚖️ Risk Contributions")
                    
                    df_rc = pd.DataFrame([
                        {"Asset": asset, "Risk %": contrib * 100}
                        for asset, contrib in sorted(risk_contribs.items(), key=lambda x: -abs(x[1]))
                    ])
                    
                    fig_rc = go.Figure(data=[
                        go.Bar(
                            x=df_rc['Asset'],
                            y=df_rc['Risk %'],
                            marker=dict(
                                color=df_rc['Risk %'],
                                colorscale='Viridis',
                                showscale=True
                            ),
                            text=df_rc['Risk %'].apply(lambda x: f"{x:.2f}%"),
                            textposition='outside'
                        )
                    ])
                    
                    fig_rc.update_layout(
                        title="Risk Contribution by Asset",
                        xaxis_title="Asset",
                        yaxis_title="Risk Contribution (%)",
                        template='plotly_white',
                        height=450
                    )
                    
                    st.plotly_chart(fig_rc, use_container_width=True)
                    st.caption("📊 Each asset's contribution to total portfolio risk")
                
                # === DRAWDOWN (Max Drawdown Method) ===
                elif method_key == "max_drawdown":
                    st.subheader("📉 Drawdown Analysis")
                    
                    # Calculate drawdown from cumulative returns
                    cum_ret_series = chart_series / 100  # Convert back from base 100
                    running_max = cum_ret_series.cummax()
                    drawdown = (cum_ret_series - running_max) / running_max * 100
                    
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values,
                        mode='lines',
                        name='Drawdown',
                        line=dict(color='red', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.2)'
                    ))
                    
                    fig_dd.update_layout(
                        title="Portfolio Drawdown Over Time",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        template='plotly_white',
                        height=450,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_dd, use_container_width=True)
                    st.caption(f"📉 Maximum Drawdown: {metrics['Max Drawdown']*100:.2f}%")
                
                else:
                    st.info("💡 Efficient Frontier available for MVO methods\n\n"
                           "⚖️ Risk Contributions available for Risk Parity\n\n"
                           "📉 Drawdown chart available for Max Drawdown method")
            
            with viz_col2:
                # === RETURNS DISTRIBUTION ===
                st.subheader("📊 Returns Distribution")
                
                # Calculate daily returns
                daily_rets = chart_series.pct_change().dropna() * 100
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=daily_rets.values,
                    nbinsx=50,
                    name='Daily Returns',
                    marker=dict(color='skyblue', line=dict(color='darkblue', width=1))
                ))
                
                # Add normal distribution overlay
                mean_ret = daily_rets.mean()
                std_ret = daily_rets.std()
                
                fig_hist.update_layout(
                    title="Daily Returns Distribution",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    template='plotly_white',
                    height=450,
                    showlegend=False
                )
                
                # Add vertical lines for mean and +/- 1 std
                fig_hist.add_vline(x=mean_ret, line_dash="dash", line_color="green", 
                                  annotation_text="Mean", annotation_position="top")
                
                st.plotly_chart(fig_hist, use_container_width=True)
                st.caption(f"📈 Mean: {mean_ret:.3f}% | Std Dev: {std_ret:.3f}%")
            
            # === JSON OUTPUT ===
            with st.expander("📄 View Full JSON Response"):
                st.json(data)

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout. Try:\n- Shorter date range\n- Fewer assets\n- Different optimization method")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to API. Ensure backend is running:\n```\npython main.py\n```")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🐛 Debug Info"):
                st.exception(e)

else:
    # === WELCOME SCREEN ===
    st.markdown("""
    ### 👋 Welcome to the Advanced Portfolio Optimizer
    
    Professional quantitative asset allocation powered by modern portfolio theory and beyond.
    
    ---
    
    #### 🎯 Available Optimization Methods:
    
    | Method | Description | Best For |
    |--------|-------------|----------|
    | **Max Sharpe (MVO)** | Maximize risk-adjusted returns | Classic mean-variance optimization |
    | **Min Variance (MVO)** | Minimize portfolio volatility | Conservative risk management |
    | **CVaR Minimization** | Minimize tail risk | Downside protection |
    | **Risk Parity** | Equal risk contribution | Diversified risk allocation |
    | **Tracking Error Min** | Match benchmark closely | Index tracking |
    | **Info Ratio Max** | Maximize active returns | Active management |
    | **Kelly Criterion** | Maximize geometric growth | Long-term compounding |
    | **Sortino Ratio** | Focus on downside risk | Asymmetric risk preference |
    | **Omega Ratio** | Probability-weighted returns | Non-normal distributions |
    | **Min Max Drawdown** | Minimize peak decline | Capital preservation |
    
    ---

    
    1. **Choose Method**:
       - Pick optimization objective
       - Adjust risk parameters (risk-free rate, MAR, confidence)
    
    2. **Set Constraints**:
       - Long-only vs. long-short
       - Min/max position sizes
       - Budget constraint
    
    
    ---
    
    #### 📊 What You'll Get:
    
    ✅ **Optimized Weights** - Asset allocation with downloadable CSV  
    ✅ **Performance Metrics** - Sharpe, Sortino, Omega, CVaR, drawdown  
    ✅ **Cumulative Returns** - Historical backtest visualization  
    ✅ **Efficient Frontier** - Risk-return tradeoff (MVO methods)  
    ✅ **Risk Analysis** - Contributions, distributions, drawdowns  
    

  
# === FOOTER ===
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("📈 Portfolio Optimizer v1.0")
with col_f2:
    st.caption("Built with FastAPI + Streamlit")
with col_f3:
    st.caption("Data: Yahoo Finance")
