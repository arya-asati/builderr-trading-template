"""Adaptive Calmar Shield — builderr Trading Challenge, Round 1 (Jun 2 – Jul 2 2026).

OBJECTIVE: Maximise 60-day forward Calmar (annualised return / max drawdown).

DESIGN PHILOSOPHY
-----------------
Calmar = annualised return ÷ max drawdown.
The denominator (drawdown) is what kills most competitors.
This agent is built to *control the denominator first* and let
returns come as a consequence of surviving.

  1. DRAWDOWN CIRCUIT BREAKER  (highest priority)
     Track the equity high-water mark every single day.
     -  5%  from peak → HALF_RISK:  50% equities / 50% defensive
     - 10%  from peak → DEFENSIVE:  Full flight-to-quality book

  2. COMPOSITE REGIME DETECTOR
     Reads 6 signals before the loss happens:
       (a) SPY vs 50-day SMA            (b) QQQ vs 50-day SMA
       (c) SPY vs 200-day SMA           (d) QQQ 20-day realised vol
       (e) Sector breadth (11 ETFs)     (f) QQQ 60-day realised vol
     Weighted score → FULL_RISK / HALF_RISK / DEFENSIVE.

  3. QUALITY MOMENTUM RANKING (no lookahead)
     Uses only the bars provided by market_state.
     Primary: 60-day momentum (works with 220-bar warmup from tick 1).
     Secondary: 20-day short-term confirmation.
     Tertiary: trend gap (price vs 50-day SMA).
     Stocks with negative 60-day momentum are excluded.
     Relative strength vs SPY is an additional tiebreaker.
     Position sizes are vol-weighted (lower vol → bigger share).

  4. ALL-WEATHER DEFENSIVE BOOK
     TLT (20yr Treasuries) and GLD (Gold) lead the defensive basket.
     These assets typically rise in risk-off, so the defensive book
     aims for positive returns in crashes, not just lower losses.

  5. SMART REBALANCING
     Rebalances on regime change, significant position drift,
     or every REBALANCE_EVERY_DAYS days — whichever comes first.
     Between rebalances the agent returns [] to save trade budget.

No network calls. No LLM. No API keys. Pure standard-library Python.
Long-only. Beta-adjusted gross ≤ 1.32x (well under the 1.5x DQ threshold).
Max position: 24% (well under the 30% DQ threshold).
"""

from __future__ import annotations

from math import sqrt, log
from statistics import mean, pstdev
from typing import Any

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# Risk-on candidates for scoring + ranking
_RISK_ON_UNIVERSE = (
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
    "SMH", "GLD", "IAU",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "AMD",
)

# 11 SPDR sector ETFs used for market breadth measurement
_BREADTH_UNIVERSE = (
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
)

# Defensive book: (ticker, raw weight) — will be normalised
# TLT and GLD are primary flight-to-quality assets
_DEFENSIVE_BOOK = (
    ("TLT",  0.30),   # 20yr Treasury — top flight-to-quality
    ("GLD",  0.25),   # Gold — crisis/inflation hedge
    ("XLP",  0.18),   # Consumer staples
    ("XLU",  0.14),   # Utilities
    ("XLV",  0.13),   # Healthcare
)

# Beta multipliers for gross-exposure cap
_BETA: dict[str, float] = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
    # Near-zero equity-beta assets barely consume the cap
    "TLT": 0.15, "GLD": 0.05, "IAU": 0.05,
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MAX_WEIGHT          = 0.24    # Hard per-ticker cap (DQ threshold is 0.30)
DRIFT_LIMIT         = 0.27    # Drift rebalance trigger
MAX_BETA_GROSS      = 1.32    # Well under the 1.50 DQ threshold
MIN_TRADE_PCT       = 0.015   # Skip orders smaller than 1.5% of equity
REBALANCE_EVERY     = 5       # Calendar-based rebalance every N trading days

# Drawdown circuit-breaker levels
_DD_HALF    = 0.05    # -5%  from peak → HALF_RISK
_DD_FULL    = 0.10    # -10% from peak → DEFENSIVE

# Momentum look-back (in trading days) — sized to fit 220-bar warmup
_MOM_LONG   = 60      # Primary: ~3 months (works from day 1 with 220-bar warmup)
_MOM_SHORT  = 20      # Confirmation: ~1 month
_VOL_WINDOW = 20      # Realised-vol window (20 trading days)

# Target annualised portfolio volatility for position sizing
_VOL_TARGET = 0.18

# ---------------------------------------------------------------------------
# Module-level state (survives between decide() calls in one session)
# ---------------------------------------------------------------------------
_peak_equity: float = 0.0
_last_date:   str | None = None
_last_regime: str = "UNKNOWN"

# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    out: list[float] = []
    for b in bars:
        try:
            c = float(b["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if c <= 0.0:
            return []
        out.append(c)
    return out


def _sma(v: list[float], n: int) -> float | None:
    if len(v) < n:
        return None
    return mean(v[-n:])


def _mom(v: list[float], n: int) -> float | None:
    """n-bar momentum: v[-1] / v[-(n+1)] - 1."""
    if len(v) <= n:
        return None
    base = v[-(n + 1)]
    return (v[-1] / base - 1.0) if base > 0.0 else None


def _rvol(v: list[float], n: int) -> float | None:
    """Annualised realised vol of log-returns over n days."""
    if len(v) <= n:
        return None
    window = v[-(n + 1):]
    rets: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0.0:
            return None
        rets.append(log(window[i] / prev))
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


def _breadth(ms: dict[str, list[dict[str, Any]]]) -> float:
    """Fraction of 11 sector ETFs above their 50-day SMA."""
    above = total = 0
    for t in _BREADTH_UNIVERSE:
        cs = _closes(ms.get(t))
        if len(cs) < 50:
            continue
        s50 = _sma(cs, 50)
        if s50 is None:
            continue
        total += 1
        if cs[-1] > s50:
            above += 1
    return above / total if total > 0 else 0.5

# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def _positions(ps: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for raw in (ps.get("positions") or []):
        t = str(raw.get("ticker", "")).upper()
        if not t:
            continue
        try:
            qty = float(raw.get("quantity", 0.0))
            ac  = float(raw.get("avg_cost",  0.0))
        except (TypeError, ValueError):
            continue
        if qty <= 0.0:
            continue
        entry = result.setdefault(t, {"quantity": 0.0, "avg_cost": ac})
        entry["quantity"] += qty
        entry["avg_cost"]  = ac or entry["avg_cost"]
    return result


def _equity(ps: dict[str, Any], cash: float) -> float:
    try:
        total = float(ps.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    lp = ps.get("last_prices") or {}
    for t, pos in _positions(ps).items():
        try:
            price = float(lp.get(t, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


def _prices(ms: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    return {
        t.upper(): float(bars[-1]["close"])
        for t, bars in ms.items()
        if bars and bars[-1].get("close", 0) > 0
    }


def _bar_date(ms: dict[str, list[dict[str, Any]]]) -> str | None:
    for anchor in ("SPY", "QQQ", "IWM"):
        bars = ms.get(anchor) or []
        if bars:
            ts = bars[-1].get("ts")
            return str(ts)[:10] if ts is not None else str(len(bars))
    return None


def _days_since(ms: dict[str, list[dict[str, Any]]]) -> int | None:
    if _last_date is None:
        return None
    for anchor in ("SPY", "QQQ"):
        bars = ms.get(anchor) or []
        if not bars:
            continue
        dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
        if _last_date in dates:
            return len(dates) - dates.index(_last_date) - 1
    return None


def _drifted(ps: dict[str, Any], eq: float) -> bool:
    if eq <= 0.0:
        return False
    lp = ps.get("last_prices") or {}
    for t, pos in _positions(ps).items():
        try:
            price = float(lp.get(t, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0.0 and (pos["quantity"] * price / eq) > DRIFT_LIMIT:
            return True
    return False

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def _regime(ms: dict[str, list[dict[str, Any]]], drawdown: float) -> str:
    """Return FULL_RISK, HALF_RISK, or DEFENSIVE."""
    # Circuit breaker: drawdown takes absolute priority
    if drawdown >= _DD_FULL:
        return "DEFENSIVE"
    if drawdown >= _DD_HALF:
        return "HALF_RISK"

    spy = _closes(ms.get("SPY"))
    qqq = _closes(ms.get("QQQ"))
    if len(spy) < 50 or len(qqq) < 50:
        return "DEFENSIVE"   # Not enough data → safe default

    spy50  = _sma(spy, 50)
    qqq50  = _sma(qqq, 50)
    spy200 = _sma(spy, 200)

    vol20  = _rvol(qqq, 20) or 1.0
    vol60  = _rvol(qqq, 60) or 1.0
    brd    = _breadth(ms)

    # Six independent signals (True = risk-on)
    sigs = [
        bool(spy50  and spy[-1] > spy50),         # SPY above 50d
        bool(qqq50  and qqq[-1] > qqq50),         # QQQ above 50d
        bool(spy200 and spy[-1] > spy200),         # SPY above 200d
        vol20 < 0.28,                              # Calm short-vol
        vol60 < 0.25,                              # Calm long-vol
        brd   > 0.60,                              # Broad participation
    ]
    score = sum(sigs)

    if score >= 5:
        return "FULL_RISK"
    if score >= 3:
        return "HALF_RISK"
    return "DEFENSIVE"

# ---------------------------------------------------------------------------
# Asset scoring
# ---------------------------------------------------------------------------

def _score(t: str, ms: dict[str, list[dict[str, Any]]], spy_cs: list[float]) -> float | None:
    """Composite quality-momentum score, or None if not rankable."""
    cs = _closes(ms.get(t))
    if len(cs) < max(_MOM_LONG + 1, 51):
        return None

    m60  = _mom(cs, _MOM_LONG)
    m20  = _mom(cs, _MOM_SHORT)
    s50  = _sma(cs, 50)
    v20  = _rvol(cs, _VOL_WINDOW)

    if None in (m60, m20, s50, v20):
        return None

    # Exclude assets in a clear downtrend
    if m60 < -0.03:
        return None

    trend_gap = cs[-1] / s50 - 1.0

    # Relative strength vs SPY (20-day)
    spy_m20 = _mom(spy_cs, _MOM_SHORT) if spy_cs else None
    rs = (m20 - spy_m20) if spy_m20 is not None else 0.0

    raw = 0.50 * m60 + 0.25 * m20 + 0.15 * trend_gap + 0.10 * rs
    # Divide by vol to reward Sharpe-efficient assets
    return raw / max(float(v20), 0.05)

# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------

def _defensive_weights(ms: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    raw = {t: w for t, w in _DEFENSIVE_BOOK if _closes(ms.get(t))}
    if not raw:
        # Absolute fallback: 50% SPY + 50% GLD if nothing else works
        fallback: dict[str, float] = {}
        if _closes(ms.get("SPY")):
            fallback["SPY"] = 0.50
        if _closes(ms.get("GLD")):
            fallback["GLD"] = 0.50
        if not fallback and _closes(ms.get("SPY")):
            fallback["SPY"] = 1.0
        return fallback
    total = sum(raw.values())
    return {t: w / total for t, w in raw.items()}


def _inv_vol_weights(
    ranked: list[tuple[float, str]],
    ms: dict[str, list[dict[str, Any]]],
    n: int,
    budget: float,
) -> dict[str, float]:
    """Inverse-vol weights across top-n assets."""
    winners = [t for _, t in ranked[:n]]
    if not winners:
        return {}

    inv: dict[str, float] = {}
    for t in winners:
        cs = _closes(ms.get(t))
        v  = _rvol(cs, _VOL_WINDOW) if cs else None
        inv[t] = 1.0 / max(float(v or 0.20), 0.05)

    total = sum(inv.values())
    if total <= 0.0:
        w = budget / len(winners)
        return {t: w for t in winners}
    return {t: iv / total * budget for t, iv in inv.items()}


def _port_vol_scale(weights: dict[str, float], ms: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Scale down if estimated portfolio vol > target."""
    pv = 0.0
    for t, w in weights.items():
        cs = _closes(ms.get(t))
        v  = _rvol(cs, _VOL_WINDOW) if cs else None
        pv += w * float(v or 0.20)
    if pv > _VOL_TARGET * 1.15:
        s = _VOL_TARGET / pv
        weights = {t: w * s for t, w in weights.items()}
    return weights


def _cap(weights: dict[str, float]) -> dict[str, float]:
    """Apply per-ticker cap and beta-adjusted gross-exposure cap."""
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    bg = sum(w * _BETA.get(t, 1.0) for t, w in capped.items())
    if bg > MAX_BETA_GROSS:
        s = MAX_BETA_GROSS / bg
        capped = {t: w * s for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w >= 0.001}

# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

def _targets(ms: dict[str, list[dict[str, Any]]], reg: str) -> dict[str, float]:

    if reg == "DEFENSIVE":
        return _cap(_defensive_weights(ms))

    spy_cs = _closes(ms.get("SPY"))
    scored: list[tuple[float, str]] = []
    for t in _RISK_ON_UNIVERSE:
        s = _score(t, ms, spy_cs)
        if s is not None:
            scored.append((s, t))
    scored.sort(reverse=True)

    if reg == "HALF_RISK":
        # 50% top-4 momentum, 50% defensive
        mom_w = _inv_vol_weights(scored, ms, 4, 0.50)
        def_w = _defensive_weights(ms)
        # Blend: defensive part scaled to 50% budget
        total_def = sum(def_w.values()) or 1.0
        for t, w in def_w.items():
            mom_w[t] = mom_w.get(t, 0.0) + w / total_def * 0.50
        return _cap(_port_vol_scale(mom_w, ms))

    # FULL_RISK
    qqq = _closes(ms.get("QQQ")) or []
    qqq_v20  = _rvol(qqq, 20) or 1.0
    qqq_s50  = _sma(qqq, 50)
    qqq_s20  = _sma(qqq, 20)
    qqq_m20  = _mom(qqq, 20)

    use_2x = bool(
        qqq_s50 and qqq_s20 and qqq_m20 is not None
        and qqq and qqq[-1] > qqq_s50
        and qqq_s20 > qqq_s50
        and qqq_m20 > 0.02
        and qqq_v20 < 0.20               # Only in very calm uptrends
        and _closes(ms.get("QLD"))
        and _closes(ms.get("SSO"))
    )

    n_picks    = 6 if use_2x else 7
    base_bud   = 0.80 if use_2x else 0.95
    weights    = _inv_vol_weights(scored, ms, n_picks, base_bud)

    # Fallback: if no scored assets, use SPY + QQQ
    if not weights:
        spyq: dict[str, float] = {}
        if _closes(ms.get("SPY")):
            spyq["SPY"] = 0.50
        if _closes(ms.get("QQQ")):
            spyq["QQQ"] = 0.45
        weights = spyq

    if use_2x:
        weights["QLD"] = 0.12
        weights["SSO"] = 0.08

    return _cap(_port_vol_scale(weights, ms))

# ---------------------------------------------------------------------------
# Order generation
# ---------------------------------------------------------------------------

def _orders(
    targets: dict[str, float],
    pos: dict[str, dict[str, float]],
    eq: float,
    px: dict[str, float],
    cash: float,
) -> list[dict[str, object]]:
    if eq <= 0.0:
        return []
    min_v = eq * MIN_TRADE_PCT
    ords: list[dict[str, object]] = []
    proceeds = 0.0

    # Sells first
    for t, p in pos.items():
        price = px.get(t)
        if not price or price <= 0.0:
            continue
        qty    = p["quantity"]
        cur_v  = qty * price
        tgt_v  = eq * targets.get(t, 0.0)
        delta  = tgt_v - cur_v
        if t not in targets:
            s = int(qty)
            if s > 0 and cur_v >= min_v:
                ords.append({"ticker": t, "side": "sell", "quantity": s})
                proceeds += s * price
        elif delta < -min_v:
            s = min(int(abs(delta) / price), int(qty))
            if s > 0:
                ords.append({"ticker": t, "side": "sell", "quantity": s})
                proceeds += s * price

    spendable = max(float(cash), 0.0) + proceeds * 0.98

    # Buys second
    for t, weight in sorted(targets.items()):
        price = px.get(t)
        if not price or price <= 0.0:
            continue
        cur_q  = pos.get(t, {}).get("quantity", 0.0)
        cur_v  = cur_q * price
        tgt_v  = eq * weight
        delta  = tgt_v - cur_v
        if delta < min_v:
            continue
        buy_v = min(delta, spendable)
        buy_q = int(buy_v / price)
        if buy_q > 0:
            ords.append({"ticker": t, "side": "buy", "quantity": buy_q})
            spendable -= buy_q * price

    return ords[:45]   # Stay under 50 trades/day cap

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    """Return a list of long-only buy/sell orders.

    Called once per decision interval (daily) by the builderr engine.
    """
    global _peak_equity, _last_date, _last_regime

    if not market_state:
        return []

    today = _bar_date(market_state)
    if today is None:
        return []

    eq = _equity(portfolio_state, cash)
    if eq <= 0.0:
        return []

    # Update high-water mark
    if eq > _peak_equity:
        _peak_equity = eq

    dd  = (_peak_equity - eq) / _peak_equity if _peak_equity > 0.0 else 0.0
    reg = _regime(market_state, dd)

    # Decide whether to act
    days      = _days_since(market_state)
    drift     = _drifted(portfolio_state, eq)
    reg_chg   = (reg != _last_regime)

    should = (
        _last_date is None
        or days is None
        or days >= REBALANCE_EVERY
        or drift
        or reg_chg
    )

    if not should:
        return []

    tgts = _targets(market_state, reg)

    # Edge case: no targets → liquidate to cash
    if not tgts:
        pos = _positions(portfolio_state)
        px  = _prices(market_state)
        liq = [
            {"ticker": t, "side": "sell", "quantity": int(p["quantity"])}
            for t, p in pos.items()
            if px.get(t, 0.0) > 0.0 and int(p["quantity"]) > 0
        ]
        _last_date   = today
        _last_regime = reg
        return liq[:45]

    px   = _prices(market_state)
    pos  = _positions(portfolio_state)
    ords = _orders(tgts, pos, eq, px, cash)

    if ords:
        _last_date   = today
        _last_regime = reg

    return ords
