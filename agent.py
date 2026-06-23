import numpy as np
import pandas as pd
import warnings

# Force-silence any environment warnings for the online server
warnings.filterwarnings("ignore")

class InstitutionalAlphaEngine:
    @staticmethod
    def calculate_hurst_exponent(close_prices: np.ndarray, max_lags: int = 10) -> float:
        try:
            if len(close_prices) < max_lags * 2: 
                return 0.50
            lags = np.arange(2, max_lags)
            variances = []
            for lag in lags:
                diffs = close_prices[lag:] - close_prices[:-lag]
                std_dev = np.std(diffs)
                variances.append(std_dev if std_dev > 0 else 1e-6)
            poly = np.polyfit(np.log(lags), np.log(variances), 1)
            return float(np.clip(poly[0] * 2.0, 0.0, 1.0))
        except Exception: 
            return 0.50

    @classmethod
    def evaluate_asset(cls, df: pd.DataFrame) -> dict:
        metrics = {"signal": "HOLD", "alpha_score": 0.0, "atr_pct": 0.01, "price": 0.0}
        try:
            if df is None or len(df) < 30:
                return metrics
            
            col_map = {str(c).lower().strip(): c for c in df.columns}
            close_key = col_map.get('close', col_map.get('price', df.columns[-1]))
            high_key = col_map.get('high', close_key)
            low_key = col_map.get('low', close_key)
            
            closes = df[close_key].to_numpy(dtype=float)
            highs = df[high_key].to_numpy(dtype=float)
            lows = df[low_key].to_numpy(dtype=float)
            
            current_price = closes[-1]
            metrics["price"] = current_price
            prices_series = pd.Series(closes)

            ema_9 = prices_series.ewm(span=9, adjust=False).mean().to_numpy()[-1]
            ema_50 = prices_series.ewm(span=50, adjust=False).mean().to_numpy()[-1]
            ema_100 = prices_series.ewm(span=100, adjust=False).mean().to_numpy()[-1] if len(closes) >= 100 else ema_50

            hl = highs - lows
            hc = np.abs(highs - np.roll(closes, 1))
            lc = np.abs(lows - np.roll(closes, 1))
            hc[0], lc[0] = 0, 0
            atr = pd.Series(np.maximum(hl, np.maximum(hc, lc))).rolling(window=14).mean().to_numpy()[-1]
            if np.isnan(atr) or atr <= 0: atr = current_price * 0.01
            metrics["atr_pct"] = float(atr / current_price)

            momentum = prices_series.pct_change().tail(5).mean()
            if np.isnan(momentum): momentum = 0.0

            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / (loss + 1e-6))))
            rsi_val = rsi.to_numpy()[-1]
            if np.isnan(rsi_val): rsi_val = 50.0

            hurst_val = cls.calculate_hurst_exponent(closes[-30:])

            # ================= COMEBACK UPGRADE: OPTIMIZED REGIMES =================
            if hurst_val >= 0.48:  # Trend & Velocity Expansion Mode
                # Buy when asset is structurally sound (above EMA 50) and building upward velocity
                if current_price > ema_50 and momentum > -0.001 and rsi_val < 78:
                    metrics["signal"] = "BUY"
                    # Weight score by structural momentum strength to capture explosive runners
                    metrics["alpha_score"] = float((momentum * 100.0) + (rsi_val * 0.05))
                # Indicator sell only on true macro trend breakdowns (below EMA 100) or massive overbought peaks
                elif current_price < ema_100 or rsi_val > 85:
                    metrics["signal"] = "SELL"

            else:  # Mean-Reversion / High-Volatility Oscillating Mode
                rolling_mean = prices_series.rolling(window=20).mean().to_numpy()[-1]
                lower_floor = rolling_mean - (1.1 * atr)  # Widened boundary from 1.5 to capture turns faster
                upper_ceiling = rolling_mean + (1.3 * atr)

                if current_price <= lower_floor or rsi_val < 38:
                    metrics["signal"] = "BUY"
                    metrics["alpha_score"] = float(100.0 - rsi_val)
                # Take profit when tagging the ceiling or overextended
                elif current_price >= upper_ceiling or rsi_val > 72:
                    metrics["signal"] = "SELL"
            
            return metrics
        except Exception:
            return metrics

def decide(market_state: dict, portfolio_state: dict, cash: float) -> list:
    try:
        orders = []
        if not market_state:
            return orders

        # 1. Total Portfolio Accounting & Real-Time Exposure Tracking
        total_portfolio_value = float(cash)
        current_gross_exposure = 0.0
        current_positions = {}
        active_prices = {}

        for ticker, p_val in portfolio_state.items():
            qty_held = p_val.get('quantity', p_val.get('qty', 0)) if isinstance(p_val, dict) else p_val
            if qty_held and float(qty_held) > 0:
                current_positions[ticker] = float(qty_held)

        for ticker, bars in market_state.items():
            if bars:
                try:
                    last_bar = bars[-1]
                    price = last_bar.get("close", last_bar.get("price", last_bar.get("open", 0)))
                    active_prices[ticker] = float(price)
                except Exception:
                    continue

        # Calculate exact Net Asset Value (NAV) and starting Gross Exposure
        for ticker, qty in current_positions.items():
            if ticker in active_prices:
                pos_value = qty * active_prices[ticker]
                total_portfolio_value += pos_value
                current_gross_exposure += pos_value

        available_cash = float(cash)
        buy_candidates = []

        # 2. Extract Closures, Risk Triggers, and Safety Liquidations
        for ticker, bars in market_state.items():
            if not bars or len(bars) < 30:
                continue
            
            try:
                df = pd.DataFrame(bars)
                col_map = {str(c).lower().strip(): c for c in df.columns}
                close_key = col_map.get('close', col_map.get('price', df.columns[-1]))
                
                analysis = InstitutionalAlphaEngine.evaluate_asset(df)
                current_price = active_prices.get(ticker, analysis["price"])
                
                if current_price <= 0:
                    continue

                qty_held = current_positions.get(ticker, 0.0)

                # --- UNTOUCHED SAFETY NET: Stateless 4% Trailing Stop Floor ---
                if qty_held > 0 and len(bars) >= 10:
                    closes_lookback = df[close_key].to_numpy(dtype=float)
                    lookback_window = min(len(closes_lookback), 45)
                    structural_peak = float(np.max(closes_lookback[-lookback_window:]))
                    
                    if current_price < (structural_peak * 0.96):
                        orders.append({"ticker": str(ticker), "side": "sell", "quantity": int(qty_held)})
                        current_gross_exposure -= (qty_held * current_price)
                        available_cash += (qty_held * current_price)
                        continue  # Exit processed, pass to next asset

                # Standard Indicator-based Liquidations
                if analysis["signal"] == "SELL" and qty_held > 0:
                    orders.append({"ticker": str(ticker), "side": "sell", "quantity": int(qty_held)})
                    current_gross_exposure -= (qty_held * current_price)
                    available_cash += (qty_held * current_price)
                
                elif analysis["signal"] == "BUY":
                    buy_candidates.append({
                        "ticker": str(ticker),
                        "price": float(current_price),
                        "score": float(analysis["alpha_score"]),
                        "atr_pct": float(analysis["atr_pct"]),
                        "qty_held": float(qty_held)
                    })
            except Exception:
                continue

        # 3. Aggressive Capital Deployment & Leverage Optimization Layer
        max_absolute_exposure = total_portfolio_value * 1.42
        
        # Keep concentration tightly locked to the top 3 high-conviction assets to force max upward velocity
        buy_candidates = sorted(buy_candidates, key=lambda x: x["score"], reverse=True)[:3]

        for candidate in buy_candidates:
            try:
                remaining_leverage_room = max_absolute_exposure - current_gross_exposure
                if remaining_leverage_room <= 0:
                    break

                ticker = candidate["ticker"]
                price = candidate["price"]
                atr_pct = candidate["atr_pct"]
                qty_held = candidate["qty_held"]

                # High-conviction risk coefficient retained at 0.035 for powerful target executions
                base_allocation = total_portfolio_value * (0.035 / (atr_pct + 1e-5))
                
                target_spend = min(base_allocation, total_portfolio_value * 0.22, available_cash * 0.45, remaining_leverage_room)
                max_allowed_spend = (total_portfolio_value * 0.30) - (qty_held * price)
                
                final_spend = min(target_spend, max_allowed_spend)
                if final_spend > 0:
                    quantity = int(final_spend / price)
                    if quantity > 0:
                        orders.append({"ticker": ticker, "side": "buy", "quantity": quantity})
                        available_cash -= (quantity * price)
                        current_gross_exposure += (quantity * price)
            except Exception:
                continue

        return orders

    except Exception:
        return []