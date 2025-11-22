import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional

class PortfolioManager:
    """Handles data fetching and metric calculations."""
    
    @staticmethod
    def get_data(tickers: List[str], start: str, end: str, benchmark: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fetch historical price data for assets and optional benchmark."""
        try:
            # Fetch Assets
            data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
            
            if data.empty:
                raise ValueError("No data fetched. Check internet connection, tickers, or date range.")

            # Extract Price Column
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                raise ValueError(f"Could not find price data. Columns: {data.columns}")

            # Handle Single Ticker (Series to DataFrame)
            if isinstance(data, pd.Series):
                data = data.to_frame()
                if len(tickers) == 1:
                    data.columns = tickers

            # Fetch Benchmark
            bench_data = None
            if benchmark:
                bench_df = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=False)
                if not bench_df.empty:
                    if 'Adj Close' in bench_df.columns:
                        b_close = bench_df['Adj Close']
                    elif 'Close' in bench_df.columns:
                        b_close = bench_df['Close']
                    else:
                        b_close = bench_df.iloc[:, 0]
                    
                    if isinstance(b_close, pd.DataFrame):
                        b_close = b_close.squeeze()
                    
                    b_close = b_close.rename(benchmark)
                    data = data.join(b_close, how='inner')
                    bench_data = data[benchmark]
                    data = data.drop(columns=[benchmark])
            
            return data.dropna(), bench_data
        
        except Exception as e:
            raise ValueError(f"Data fetch error: {str(e)}")

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
            print(f"Warning: Optimization did not converge. Message: {res.message}")
        
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
