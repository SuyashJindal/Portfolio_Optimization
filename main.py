from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from optimizer import PortfolioManager, Optimizer
import json

app = FastAPI(title="Portfolio Optimizer API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptimizeRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    method: str
    benchmark: Optional[str] = None
    risk_free_rate: float = 0.02
    mar: float = 0.0
    confidence_level: float = 0.95
    # Constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    sum_is_one: bool = True
    frontier_points: int = 20


@app.get("/")
def root():
    return {
        "message": "Portfolio Optimizer API", 
        "status": "running",
        "endpoints": ["/optimize", "/frontier", "/load-data"]
    }


@app.post("/optimize")
def run_optimization(req: OptimizeRequest):
    """Main optimization endpoint."""
    try:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION REQUEST")
        print(f"{'='*60}")
        print(f"Tickers: {req.tickers}")
        print(f"Period: {req.start_date} to {req.end_date}")
        print(f"Method: {req.method}")
        print(f"{'='*60}\n")
        
        # 1. Fetch Data
        df, bench_series = PortfolioManager.get_data(
            req.tickers, req.start_date, req.end_date, req.benchmark
        )
        
        if df.empty:
            raise HTTPException(400, "No price data available")
        
        returns = df.pct_change().dropna()
        
        if returns.empty:
            raise HTTPException(400, "No return data available after processing")
        
        bench_rets = None
        if bench_series is not None:
            bench_rets = bench_series.pct_change().dropna()
            print(f"Benchmark returns: {len(bench_rets)} periods")

        print(f"Returns shape: {returns.shape}")
        print(f"Return period: {returns.index[0]} to {returns.index[-1]}")
        
        # 2. Setup Optimizer
        constraints = {
            'min_weight': req.min_weight, 
            'max_weight': req.max_weight, 
            'sum_is_one': req.sum_is_one
        }
        
        opt = Optimizer(returns, constraints, req.risk_free_rate, req.mar, bench_rets)
        
        # 3. Optimize Weights
        weights = opt.optimize(req.method, req.confidence_level)
        
        print(f"\nOptimized Weights:")
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight*100:.2f}%")

        # 4. Calculate Portfolio Metrics
        final_weights_arr = np.array([weights[t] for t in df.columns])
        metrics = PortfolioManager.get_metrics(
            final_weights_arr, returns, req.risk_free_rate, 
            req.mar, bench_rets, req.confidence_level
        )
        
        print(f"\nPortfolio Metrics:")
        for key, val in metrics.items():
            print(f"  {key}: {val}")
        
        # 5. Generate Cumulative Returns for Chart
        port_daily_rets = returns.dot(final_weights_arr)
        port_cum_ret = (1 + port_daily_rets).cumprod()
        chart_series = (port_cum_ret / port_cum_ret.iloc[0]) * 100
        
        # Convert to dict with string keys (dates)
        chart_data = {str(date.date()): float(value) for date, value in chart_series.items()}
        
        print(f"\nChart data points: {len(chart_data)}")
        
        # 6. Risk Contributions (for risk parity visualization)
        risk_contributions = opt.get_risk_contributions(weights)
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}\n")
        
        return {
            "weights": weights,
            "metrics": metrics,
            "chart_data": chart_data,
            "risk_contributions": risk_contributions,
            "tickers": req.tickers,
            "dates": {
                "start": str(returns.index[0].date()),
                "end": str(returns.index[-1].date())
            }
        }

    except HTTPException:
        raise
    except ValueError as e:
        print(f"ValueError: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Optimization failed: {str(e)}")


@app.post("/frontier")
def get_frontier(req: OptimizeRequest):
    """Calculate efficient frontier for MVO methods."""
    try:
        print(f"\n{'='*60}")
        print("EFFICIENT FRONTIER CALCULATION")
        print(f"{'='*60}\n")
        
        # Fetch data
        df, _ = PortfolioManager.get_data(req.tickers, req.start_date, req.end_date)
        
        if df.empty:
            raise HTTPException(400, "No price data available")
        
        returns = df.pct_change().dropna()
        
        if returns.empty:
            raise HTTPException(400, "No return data available")
        
        print(f"Data: {returns.shape[0]} days, {returns.shape[1]} assets")
        
        # Setup optimizer
        constraints = {
            'min_weight': req.min_weight, 
            'max_weight': req.max_weight, 
            'sum_is_one': req.sum_is_one
        }
        opt = Optimizer(returns, constraints, req.risk_free_rate)
        
        # Get boundary portfolios
        print("Finding minimum volatility portfolio...")
        w_min_vol = np.array(list(opt.optimize("mvo_min_vol").values()))
        
        print("Finding maximum Sharpe portfolio...")
        w_max_sharpe = np.array(list(opt.optimize("mvo_sharpe").values()))
        
        # Calculate return range
        ret_min = np.sum(returns.mean() * w_min_vol) * 252
        ret_max = np.sum(returns.mean() * w_max_sharpe) * 252
        
        # Extend range slightly
        ret_range = ret_max - ret_min
        ret_min = ret_min - 0.1 * ret_range
        ret_max = ret_max + 0.3 * ret_range
        
        print(f"Return range: {ret_min:.4f} to {ret_max:.4f}")
        
        # Generate frontier points
        target_returns = np.linspace(ret_min, ret_max, req.frontier_points)
        
        frontier_vol = []
        frontier_ret = []
        
        print(f"Calculating {req.frontier_points} frontier points...")
        
        for i, target in enumerate(target_returns):
            # Create temporary constraints with target return
            temp_cons = opt.cons.copy()
            temp_cons.append({
                'type': 'eq', 
                'fun': lambda x, t=target: np.sum(opt.mean_rets * x) - t
            })
            
            try:
                # Minimize volatility for target return
                res = minimize(
                    lambda w: np.sqrt(np.dot(w.T, np.dot(opt.cov_matrix, w))),
                    opt._get_start_guess(),
                    method='SLSQP',
                    bounds=opt.bounds,
                    constraints=temp_cons,
                    options={'maxiter': 500, 'ftol': 1e-9}
                )
                
                if res.success:
                    frontier_vol.append(float(res.fun))
                    frontier_ret.append(float(target))
                    
            except Exception as e:
                print(f"Point {i} failed: {e}")
                continue
        
        if len(frontier_vol) < 3:
            raise HTTPException(500, "Could not generate enough frontier points")
        
        print(f"Successfully generated {len(frontier_vol)} frontier points")
        print(f"{'='*60}\n")
        
        return {
            "volatility": frontier_vol, 
            "return": frontier_ret,
            "num_points": len(frontier_vol)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Frontier error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Frontier calculation failed: {str(e)}")


@app.post("/load-data")
def load_data(req: OptimizeRequest):
    """Load and preview historical data."""
    try:
        print(f"Loading data for validation: {req.tickers}")
        
        df, bench = PortfolioManager.get_data(
            req.tickers, req.start_date, req.end_date, req.benchmark
        )
        
        if df.empty:
            raise HTTPException(400, "No data could be fetched")
        
        returns = df.pct_change().dropna()
        
        return {
            "status": "success",
            "tickers": list(df.columns),
            "data_points": len(df),
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "has_benchmark": bench is not None,
            "mean_returns": {ticker: float(returns[ticker].mean() * 252) 
                           for ticker in df.columns},
            "volatilities": {ticker: float(returns[ticker].std() * np.sqrt(252)) 
                           for ticker in df.columns}
        }
    
    except Exception as e:
        print(f"Data loading error: {e}")
        raise HTTPException(400, f"Data loading failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  PORTFOLIO OPTIMIZER API")
    print("="*60)
    print("Starting server on http://127.0.0.1:8000")
    print("API docs available at http://127.0.0.1:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
