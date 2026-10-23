"""Data download, technical indicators and the buy/sell recommendation.

Plain pandas only (no Streamlit), so it can be tested on its own.
"""
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:  # older yfinance versions
    class YFRateLimitError(Exception):
        pass


# How each timeframe is downloaded and how much of it is shown.
# Extra history is fetched so RSI/MACD are already warmed up when the window starts.
TIMEFRAMES = {
    "Day":     {"period": "5d",  "interval": "5m",  "sessions": 1},
    "Week":    {"period": "1mo", "interval": "30m", "sessions": 5},
    "Month":   {"period": "1y",  "interval": "1d",  "days": 30},
    "60 days": {"period": "1y",  "interval": "1d",  "days": 60},
    "90 days": {"period": "1y",  "interval": "1d",  "days": 90},
}


class DataError(Exception):
    """A problem the user should see as a friendly message."""


RATE_LIMIT_MSG = "Yahoo is rate-limiting requests right now. Please try again in a few minutes."


def fetch_history(symbol, period, interval):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
    except YFRateLimitError:
        raise DataError(RATE_LIMIT_MSG)
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            raise DataError(RATE_LIMIT_MSG)
        raise DataError(f"Could not download data for '{symbol}': {e}")
    if df is None or df.empty:
        raise DataError(f"No data found for '{symbol}'. Check that the symbol is correct.")
    return df.dropna(subset=["Close"])


def trim_window(df, timeframe):
    """Keep only the bars inside the chosen timeframe."""
    tf = TIMEFRAMES[timeframe]
    if "sessions" in tf:
        keep = sorted(set(df.index.date))[-tf["sessions"]:]
        return df[np.isin(df.index.date, keep)]
    cutoff = df.index[-1] - pd.Timedelta(days=tf["days"])
    return df[df.index > cutoff]


# ---------------------------------------------------------------- indicators

def rsi(close, period=14):
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(close, fast=12, slow=26, signal=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def support_resistance(daily, order=5, tolerance=0.015):
    """Support/resistance levels from 1 year of daily bars.

    Swing highs/lows (extremes of the surrounding +/- `order` bars) are grouped
    when within `tolerance` of each other; touches = number of swings in a group.
    Both lists are sorted nearest-to-price first and are never empty.
    """
    price = float(daily["Close"].iloc[-1])
    win = 2 * order + 1
    highs, lows = daily["High"], daily["Low"]
    swing_highs = highs[highs == highs.rolling(win, center=True).max()]
    swing_lows = lows[lows == lows.rolling(win, center=True).min()]
    points = np.sort(np.concatenate([swing_highs.values, swing_lows.values]))

    clusters = []
    for p in points:
        if clusters and p <= np.mean(clusters[-1]) * (1 + tolerance):
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = [{"price": float(np.mean(c)), "touches": len(c), "note": ""} for c in clusters]

    supports = sorted((l for l in levels if l["price"] < price), key=lambda l: price - l["price"])
    resistances = sorted((l for l in levels if l["price"] > price), key=lambda l: l["price"] - price)

    low_52, high_52 = float(lows.min()), float(highs.max())
    if not supports:
        supports = [{"price": low_52, "touches": 1, "note": "52-week low"} if low_52 < price
                    else {"price": price * 0.95, "touches": 1, "note": "No support found: 5% below price"}]
    if not resistances:
        resistances = [{"price": high_52, "touches": 1, "note": "52-week high"} if high_52 > price
                       else {"price": price * 1.05, "touches": 1, "note": "At 52-week high: +5% target"}]
    return supports, resistances


def volume_stats(window, daily):
    sessions = len(set(window.index.date))
    avg_window = float(daily["Volume"].tail(sessions).mean())
    avg_50d = float(daily["Volume"].tail(50).mean())
    return {
        "sessions": sessions,
        "avg_window": avg_window,
        "avg_50d": avg_50d,
        "avg_per_bar": float(window["Volume"].mean()),
        "ratio": avg_window / avg_50d if avg_50d else 1.0,
    }


# ------------------------------------------------------------ recommendation

def _key_level(levels):
    """Nearest level touched at least twice, else simply the nearest one."""
    strong = [l for l in levels if l["touches"] >= 2]
    return (strong or levels)[0]


def recommend(price, supports, resistances, rsi_series, macd_df, vol):
    score = 0
    reasons = []

    rsi_now = float(rsi_series.dropna().iloc[-1])
    if rsi_now < 30:
        score += 1
        reasons.append(f"RSI is {rsi_now:.0f}: oversold (below 30), selling may be exhausted.")
    elif rsi_now > 70:
        score -= 1
        reasons.append(f"RSI is {rsi_now:.0f}: overbought (above 70), a pullback is more likely.")
    else:
        reasons.append(f"RSI is {rsi_now:.0f}: neutral (between 30 and 70).")

    hist = macd_df["hist"].dropna()
    macd_dir = 1 if hist.iloc[-1] > 0 else -1
    score += macd_dir
    reasons.append("MACD is above its signal line (upward momentum)." if macd_dir > 0
                   else "MACD is below its signal line (downward momentum).")
    flips = (hist.iloc[-4:] > 0).astype(int).diff().dropna()
    flips = flips[flips != 0]
    if not flips.empty:
        cross = int(flips.iloc[-1])
        score += cross
        reasons.append("MACD just crossed ABOVE its signal line (fresh bullish crossover)." if cross > 0
                       else "MACD just crossed BELOW its signal line (fresh bearish crossover).")

    ratio = vol["ratio"]
    if ratio >= 1.2:
        score += macd_dir
        reasons.append(f"Volume is {ratio:.1f}x the 50-day average, which confirms the MACD direction.")
    elif ratio <= 0.8:
        reasons.append(f"Volume is only {ratio:.1f}x the 50-day average, so the move has weak conviction.")
    else:
        reasons.append(f"Volume is close to normal ({ratio:.1f}x the 50-day average).")

    signal = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"

    support = _key_level(supports)["price"]
    if signal == "Bullish":
        buy = (support + price) / 2
        action = f"Buy near ${buy:,.2f}"
        reasons.append("Signals are bullish, so the entry is set halfway between support and the current price.")
    elif signal == "Neutral":
        buy = support
        action = f"Buy on a pullback to support near ${buy:,.2f}"
        reasons.append("Signals are mixed, so the entry is at the nearest key support.")
    else:
        lower = [s for s in supports if s["price"] < support]
        if lower:
            support = _key_level(lower)["price"]
        buy = support
        action = f"Wait. Only consider buying near ${buy:,.2f}"
        reasons.append("Signals are bearish, so the entry is pushed down to the next support level.")

    # A resistance right on top of the price is not a useful target.
    targets = [r for r in resistances if r["price"] >= price * 1.02]
    if targets:
        sell = _key_level(targets)["price"]
    else:
        sell = price * 1.05
        reasons.append("No resistance at least 2% above the price, so the target is +5%.")
    stop = support * 0.98
    risk = buy - stop
    return {
        "signal": signal,
        "score": score,
        "action": action,
        "buy": buy,
        "sell": sell,
        "stop": stop,
        "reward_risk": (sell - buy) / risk if risk > 0 else None,
        "rsi": rsi_now,
        "reasons": reasons,
    }
