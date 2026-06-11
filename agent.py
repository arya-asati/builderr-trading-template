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
    def generate_signal(cls, incoming_data: pd.DataFrame) -> str:
        try:
            if incoming_data is None or len(incoming_data) < 30: 
                return "HOLD"
                
            data = incoming_data.copy()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            col_map = {str(c).lower().strip(): c for c in data.columns}
            
            close_key = col_map.get('close', col_map.get('price', None))
            high_key = col_map.get('high', None)
            low_key = col_map.get('low', None)
            volume_key = col_map.get('volume', None)
            
            if not close_key: close_key = data.columns[-1]
            if not high_key: high_key = close_key
            if not low_key: low_key = close_key
            if not volume_key:
                vol_matches = [c for c in data.columns if 'vol' in str(c).lower()]
                volume_key = vol_matches[0] if vol_matches else close_key

            closes = data[close_key].to_numpy(dtype=float)
            highs = data[high_key].to_numpy(dtype=float)
            lows = data[low_key].to_numpy(dtype=float)
            volumes = data[volume_key].to_numpy(dtype=float)
            
            current_price = closes[-1]
            prices_series = pd.Series(closes)

            ema_9 = prices_series.ewm(span=9, adjust=False).mean().to_numpy()[-1]
            ema_50 = prices_series.ewm(span=50, adjust=False).mean().to_numpy()[-1]
            
            if len(closes) >= 100:
                ema_macro = prices_series.ewm(span=100, adjust=False).mean().to_numpy()[-1]
            else:
                ema_macro = ema_50

            hl = highs - lows
            hc = np.abs(highs - np.roll(closes, 1))
            lc = np.abs(lows - np.roll(closes, 1))
            hc[0], lc[0] = 0, 0 
            true_range = np.maximum(hl, np.maximum(hc, lc))
            atr = pd.Series(true_range).rolling(window=14).mean().to_numpy()[-1]
            if np.isnan(atr) or atr <= 0: atr = current_price * 0.01

            volume_series = pd.Series(volumes)
            v_sma_20 = volume_series.rolling(window=20).mean().to_numpy()[-1]
            v_sma_10 = volume_series.rolling(window=10).mean().to_numpy()[-1]
            
            if np.isnan(v_sma_20): v_sma_20 = volumes[-1]
            if np.isnan(v_sma_10): v_sma_10 = volumes[-1]

            hurst_val = cls.calculate_hurst_exponent(closes[-30:])

            # --- ROUTING ENGINE ---
            if hurst_val > 0.52:  # Trend Following
                if current_price > ema_macro:
                    if current_price > ema_9 and ema_9 > ema_50:
                        if volumes[-1] > (1.1 * v_sma_20): return "BUY"
                if current_price < ema_50: return "SELL"

            elif hurst_val < 0.48:  # Mean Reversion
                rolling_mean = prices_series.rolling(window=20).mean().to_numpy()[-1]
                if np.isnan(rolling_mean): rolling_mean = current_price
                
                lower_liquidity_floor = rolling_mean - (1.5 * atr)
                upper_liquidity_ceiling = rolling_mean + (1.5 * atr)

                if current_price <= lower_liquidity_floor and volumes[-1] < v_sma_10: return "BUY"
                if current_price >= upper_liquidity_ceiling: return "SELL"

            return "HOLD"
        except Exception:
            return "HOLD"

def decide(data: pd.DataFrame, *args, **kwargs) -> str:
    return InstitutionalAlphaEngine.generate_signal(data)