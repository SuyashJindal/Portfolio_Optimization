import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import time  # <--- CRITICAL IMPORT FOR RETRIES

# ============================================================================
# PORTFOLIO MANAGER & OPTIMIZER
# ============================================================================

class PortfolioManager:
    """Handles data fetching and metric calculations."""
    
    @staticmethod
    def get_data(tickers: List[str], start: str, end: str, benchmark: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fetch historical price data with retry logic for stability."""
        max_retries = 3
        data = pd.DataFrame()
        bench_data = None
        
        # --- FETCH ASSETS WITH RETRY LOGIC ---
        for attempt in range(max_retries):
            try:
                # auto_adjust=False ensures we get 'Adj Close' if available, or 'Close'
                data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
                
                # Check if data is actually empty (yfinance sometimes returns empty df without error)
                if not data.empty:
                    break
                
                time.sleep(1) # Wait 1 second before retrying
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to download data after {max_retries} attempts: {str(e)}")
                time.sleep(1)

        if data.empty:
            raise ValueError("No data fetched. Check internet connection or Ticker spelling.")

        # --- PROCESS ASSET DATA ---
        # Handle MultiIndex headers (common in new yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            # If MultiIndex, try to access 'Adj Close'
            try:
                data = data['Adj Close']
            except KeyError:
                try:
                    data = data['Close']
                except KeyError:
                    raise ValueError("Could not find 'Adj Close' or 'Close' price data.")
        else:
            # Single level columns
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
        
        # Handle Single Ticker (Series -> DataFrame)
        if isinstance(data, pd.Series):
            data = data.to_frame()
            if len(tickers) == 1:
                data.columns = tickers

        # --- FETCH BENCHMARK (OPTIONAL) ---
        if benchmark:
            for attempt in range(max_retries):
                try:
                    bench_df = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=False)
                    if not bench_df.empty:
                        # Process Benchmark Data
                        if isinstance(bench_df.columns, pd.MultiIndex):
                            try:
                                b_close = bench_df['Adj Close']
                            except KeyError:
                                b_close = bench_df['Close']
                        else:
                            if 'Adj Close' in bench_df.columns:
                                b_close = bench_df['Adj Close']
                            elif 'Close' in bench_df.columns:
                                b_close = bench_df['Close']
                            else:
                                b_close = bench_df.iloc[:, 0]

                        # Ensure it's a Series
                        if isinstance(b_close, pd.DataFrame):
                            b_close = b_close.squeeze()
                        
                        b_close = b_close.rename(benchmark)
                        
                        # Align data
                        data = data.join(b_close, how='inner')
                        bench_data = data[benchmark]
                        data = data.drop(columns=[benchmark])
                        break
                except Exception:
                    pass # Benchmark failure shouldn't crash the app, just ignore it

        return data.dropna(), bench_data

    @staticmethod
    def get_metrics(weights: np.ndarray, returns: pd.DataFrame, rf: float = 0.0, 
                    mar: float = 0.0, benchmark_rets: Optional[pd.Series] = None, 
                    confidence: float = 0.95) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        
        # Portfolio returns series
        port_rets = returns.dot(weights)
        
        # Basic Stats
        mu = port_rets.mean() * 252
        sigma = port_rets.std() * np.sqrt(252)
        sharpe = (mu - rf) / sigma if sigma > 0 else 0
        
        # Drawdown Analysis
        cum_returns = (1 + port_rets).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_dd = drawdown.min()
        
        # Sortino Ratio
        downside = port_rets[port_rets < (mar/252)]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
        sortino = (mu - mar) / downside_std
        
        # Omega Ratio
        threshold = mar / 252
        gains = port_rets[port_rets > threshold].sum()
        losses = abs(port_rets[port_rets < threshold].sum())
        omega = gains / losses if losses > 0 else np.inf
        
        # CVaR (Historical)
        var_cutoff = np.percentile(port_rets, (1 - confidence) * 100)
        cvar = port_rets[port_rets <= var_cutoff].mean()
        
        metrics = {
            "Expected Return": round(mu, 4),
            "Volatility": round(sigma, 4),
            "Sharpe Ratio": round(sharpe, 4),
            "Sortino Ratio": round(sortino, 4),
            "Omega Ratio": round(omega, 4),
            "Max Drawdown": round(max_dd, 4),
            "CVaR (95%)": round(cvar, 4)
        }
        
        # Benchmark-Relative Metrics
        if benchmark_rets is not None:
            active_ret = port_rets - benchmark_rets
            tracking_error = active_ret.std() * np.sqrt(252)
            info_ratio = (active_ret.mean() * 252) / tracking_error if tracking_error > 0 else 0
            metrics["Tracking Error"] = round(tracking_error, 4)
            metrics["Information Ratio"] = round(info_ratio, 4)
            
        return metrics


class Optimizer:
    """Portfolio optimization engine with multiple methods."""
    
    def __init__(self, returns: pd.DataFrame, constraints: Dict, rf: float = 0.0, 
                 mar: float = 0.0, benchmark_rets: Optional[pd.Series] = None):
        self.returns = returns
        self.mean_rets = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        self.num_assets = len(returns.columns)
        self.tickers = returns.columns.tolist()
        self.rf = rf
        self.mar = mar
        self.benchmark_rets = benchmark_rets
        
        # Setup bounds and constraints
        self.bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                           for _ in range(self.num_assets))
        self.cons = []
        if constraints['sum_is_one']:
            self.cons.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

    def _get_start_guess(self) -> np.ndarray:
        """Initial equal-weight guess."""
        return np.array([1.0 / self.num_assets] * self.num_assets)

    def optimize(self, method: str, confidence: float = 0.95) -> Dict[str, float]:
        """Run optimization for specified method."""
        
        # === Objective Functions ===
        
        def neg_sharpe(w):
            r = np.sum(self.mean_rets * w)
            vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            return -(r - self.rf) / (vol + 1e-10)

        def min_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))

        def neg_sortino(w):
            p_ret = self.returns.dot(w)
            mu = p_ret.mean() * 252
            downside = p_ret[p_ret < (self.mar/252)]
            down_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
            return -(mu - self.mar) / down_std

        def risk_parity(w):
            p_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            if p_vol < 1e-10:
                return 1e10
            mrc = np.dot(self.cov_matrix, w) / p_vol
            rc = w * mrc
            target = p_vol / self.num_assets
            return np.sum((rc - target)**2)
            
        def cvar_obj(w):
            p_ret = self.returns.dot(w)
            cutoff = np.percentile(p_ret, (1 - confidence) * 100)
            cvar_val = p_ret[p_ret <= cutoff].mean()
            return -cvar_val  # Minimize negative loss

        def neg_omega(w):
            p_ret = self.returns.dot(w)
            thresh = self.mar / 252
            gains = p_ret[p_ret > thresh].sum()
            losses = abs(p_ret[p_ret < thresh].sum())
            return -(gains / (losses + 1e-10))

        def neg_kelly(w):
            # Maximize E[ln(1+r)] ≈ E[r] - 0.5*Var[r]
            port_ret = np.sum(self.mean_rets * w)
            port_var = np.dot(w.T, np.dot(self.cov_matrix, w))
            return -(port_ret - 0.5 * port_var)

        def max_drawdown_obj(w):
            p_ret = self.returns.dot(w)
            cum = (1 + p_ret).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            return -dd.min()  # Minimize magnitude of MDD

        def tracking_error_obj(w):
            if self.benchmark_rets is None:
                return 1e6
            active = self.returns.dot(w) - self.benchmark_rets
            return active.std()

        def neg_info_ratio(w):
            if self.benchmark_rets is None:
                return 1e6
            active = self.returns.dot(w) - self.benchmark_rets
            te = active.std() * np.sqrt(252)
            ret = active.mean() * 252
            return -(ret / (te + 1e-10))

        # Method Mapping
        objectives = {
            "mvo_sharpe": neg_sharpe,
            "mvo_min_vol": min_vol,
            "risk_parity": risk_parity,
            "cvar": cvar_obj,
            "sortino": neg_sortino,
            "omega": neg_omega,
            "kelly": neg_kelly,
            "max_drawdown": max_drawdown_obj,
            "tracking_error": tracking_error_obj,
            "info_ratio": neg_info_ratio
        }

        if method not in objectives:
            raise ValueError(f"Method '{method}' not implemented")

        # Run Optimization
        res = minimize(
            objectives[method], 
            self._get_start_guess(), 
            method='SLSQP', 
            bounds=self.bounds, 
            constraints=self.cons,
            options={'maxiter': 1000}
        )
        
        if not res.success:
            st.warning(f"Optimization did not fully converge: {res.message}")
        
        return dict(zip(self.tickers, res.x))
    
    def get_risk_contributions(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk contributions for each asset."""
        w = np.array([weights[t] for t in self.tickers])
        port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
        
        if port_vol < 1e-10:
            return {t: 0.0 for t in self.tickers}
        
        mrc = np.dot(self.cov_matrix, w) / port_vol
        risk_contrib = w * mrc
        
        return dict(zip(self.tickers, risk_contrib))
    
    def calculate_frontier(self, num_points: int = 25) -> Tuple[List[float], List[float]]:
        """Calculate efficient frontier for MVO methods."""
        
        # Get boundary portfolios
        try:
            w_min_vol = np.array(list(self.optimize("mvo_min_vol").values()))
            w_max_sharpe = np.array(list(self.optimize("mvo_sharpe").values()))
            
            # Calculate return range
            ret_min = np.sum(self.mean_rets * w_min_vol)
            ret_max = np.sum(self.mean_rets * w_max_sharpe)
            
            # Extend range slightly
            ret_range = ret_max - ret_min
            if ret_range == 0: ret_range = 0.01 # Prevent division by zero
            
            ret_min = ret_min - 0.1 * ret_range
            ret_max = ret_max + 0.3 * ret_range
            
            # Generate frontier points
            target_returns = np.linspace(ret_min, ret_max, num_points)
            
            frontier_vol = []
            frontier_ret = []
            
            for target in target_returns:
                # Create temporary constraints with target return
                temp_cons = self.cons.copy()
                temp_cons.append({
                    'type': 'eq', 
                    'fun': lambda x, t=target: np.sum(self.mean_rets * x) - t
                })
                
                try:
                    # Minimize volatility for target return
                    res = minimize(
                        lambda w: np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w))),
                        self._get_start_guess(),
                        method='SLSQP',
                        bounds=self.bounds,
                        constraints=temp_cons,
                        options={'maxiter': 500, 'ftol': 1e-9}
                    )
                    
                    if res.success:
                        frontier_vol.append(float(res.fun))
                        frontier_ret.append(float(target))
                        
                except Exception:
                    continue
            
            return frontier_vol, frontier_ret
        except Exception as e:
            st.warning(f"Frontier calculation skipped: {str(e)}")
            return [], []


# ============================================================================
# STREAMLIT UI
# ============================================================================

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

method_label = st.sidebar.selectbox(
    "Select Method", 
    list(method_map.keys()),
    help="Choose optimization objective"
)
method_key = method_map[method_label]

# 4. Parameters
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

frontier_points = 25

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
    
    with st.spinner("🔄 Fetching data and optimizing portfolio..."):
        try:
            # === FETCH DATA ===
            df, bench_series = PortfolioManager.get_data(
                tickers, str(start_date), str(end_date), benchmark
            )
            
            if df.empty:
                st.error("❌ No price data available")
                st.stop()
            
            returns = df.pct_change().dropna()
            
            if returns.empty:
                st.error("❌ No return data available after processing")
                st.stop()
            
            bench_rets = None
            if bench_series is not None:
                bench_rets = bench_series.pct_change().dropna()
            
            # === SETUP OPTIMIZER ===
            constraints = {
                'min_weight': min_w, 
                'max_weight': max_w, 
                'sum_is_one': sum_one
            }
            
            opt = Optimizer(returns, constraints, rf, mar, bench_rets)
            
            # === OPTIMIZE WEIGHTS ===
            weights = opt.optimize(method_key, conf)
            
            # === CALCULATE METRICS ===
            final_weights_arr = np.array([weights[t] for t in df.columns])
            metrics = PortfolioManager.get_metrics(
                final_weights_arr, returns, rf, mar, bench_rets, conf
            )
            
            # === GENERATE CUMULATIVE RETURNS ===
            port_daily_rets = returns.dot(final_weights_arr)
            port_cum_ret = (1 + port_daily_rets).cumprod()
            chart_series = (port_cum_ret / port_cum_ret.iloc[0]) * 100
            
            # === RISK CONTRIBUTIONS ===
            risk_contribs = opt.get_risk_contributions(weights)
            
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
                    elif "Ratio" in metric_name:
                        display_val = f"{metric_value:.3f}"
                    elif "Drawdown" in metric_name or "CVaR" in metric_name:
                        display_val = f"{metric_value*100:.2f}%"
                    else:
                        display_val = f"{metric_value:.4f}"
                    
                    st.metric(metric_name, display_val)
            
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
                title=f"Portfolio Growth (Base 100) - {returns.index[0].date()} to {returns.index[-1].date()}",
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
                            frontier_vol, frontier_ret = opt.calculate_frontier(frontier_points)
                            
                            if len(frontier_vol) >= 3:
                                fig_ef = go.Figure()
                                
                                # Frontier curve
                                fig_ef.add_trace(go.Scatter(
                                    x=frontier_vol,
                                    y=frontier_ret,
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
                                st.caption(f"✨ Red star shows your optimized portfolio on the frontier ({len(frontier_vol)} points)")
                            else:
                                st.warning("⚠️ Could not generate enough frontier points")
                        
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
                    cum_ret_series = chart_series / 100
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
                
                # Add mean and std dev info
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
                
                # Add vertical line for mean
                fig_hist.add_vline(x=mean_ret, line_dash="dash", line_color="green", 
                                  annotation_text="Mean", annotation_position="top")
                
                st.plotly_chart(fig_hist, use_container_width=True)
                st.caption(f"📈 Mean: {mean_ret:.3f}% | Std Dev: {std_ret:.3f}%")

        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            with st.expander("🛠 Debug Info"):
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
    
    #### 🚀 Quick Start Guide:
    
    1. **Configure Portfolio** (left sidebar):
       - Enter 2+ tickers (e.g., `AAPL,MSFT,GOOGL,AMZN,JPM,JNJ`)
       - Select date range (3+ years recommended)
    
    2. **Choose Method**:
       - Pick optimization objective
       - Adjust risk parameters (risk-free rate, MAR, confidence)
    
    3. **Set Constraints**:
       - Long-only vs. long-short
       - Min/max position sizes
       - Budget constraint
    
    4. **Run Optimization**:
       - Click "🚀 RUN OPTIMIZATION"
       - View results in ~5-15 seconds
    
    ---
    
    #### 📊 What You'll Get:
    
    ✅ **Optimized Weights** - Asset allocation with downloadable CSV  
    ✅ **Performance Metrics** - Sharpe, Sortino, Omega, CVaR, drawdown  
    ✅ **Cumulative Returns** - Historical backtest visualization  
    ✅ **Efficient Frontier** - Risk-return tradeoff (MVO methods)  
    ✅ **Risk Analysis** - Contributions, distributions, drawdowns  
    
    ---
    
    #### 💡 Pro Tips:
    
    - Use **3+ years** of data for stable covariance estimates
    - For **benchmark methods** (Tracking Error, Info Ratio), specify benchmark ticker
    - Enable **"Long Only"** to prevent short positions
    - **Risk Parity** works well with diverse asset classes (stocks, bonds, commodities)
    - **CVaR** and **Max Drawdown** are conservative for risk-averse investors
    
    ---
    
    #### ⚙️ Configuration in Sidebar →
    
    **Ready to optimize?** Configure your portfolio in the sidebar and click **"🚀 RUN OPTIMIZATION"**
    
    """)
    
    # Sample portfolios
    st.info("""
    **📚 Try These Sample Portfolios:**
    
    - **Tech Growth**: `AAPL,MSFT,GOOGL,AMZN,NVDA,META`
    - **Balanced**: `SPY,TLT,GLD,VNQ,DBC,AGG`
    - **Dividend**: `VYM,SCHD,VIG,DVY,HDV,DGRO`
    - **All-Weather**: `SPY,TLT,IEF,GLD,DBC`
    """)

# === FOOTER ===
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("📈 Portfolio Optimizer v2.0")
with col_f2:
    st.caption("Built with Streamlit")
with col_f3:
    st.caption("Data: Yahoo Finance")
