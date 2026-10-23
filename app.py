"""Stock Analyzer: run with `streamlit run app.py`."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import analysis as an

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")

GREEN, RED, BLUE, ORANGE, GREY = "#16a34a", "#dc2626", "#2563eb", "#f59e0b", "#64748b"

THEMES = {
    "Black": dict(bg="#000000", panel="#111111", text="#fafafa", muted="#a3a3a3", template="plotly_dark"),
    "White": dict(bg="#ffffff", panel="#f0f2f6", text="#31333f", muted="#6b7280", template="plotly_white"),
}

# ------------------------------------------------------------------- theme
_, theme_col = st.columns([5, 1])
THEME = THEMES[theme_col.radio("Theme", list(THEMES), horizontal=True)]
st.markdown(f"""<style>
.stApp, [data-testid="stHeader"] {{ background-color: {THEME['bg']}; color: {THEME['text']}; }}
[data-testid="stSidebar"] {{ background-color: {THEME['panel']}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {THEME['text']} !important; }}
[data-testid="stCaptionContainer"] {{ color: {THEME['muted']} !important; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def load(symbol, period, interval):
    return an.fetch_history(symbol, period, interval)


@st.cache_data(ttl=86400, show_spinner=False)
def company_name(symbol):
    try:
        return yf.Ticker(symbol).info.get("shortName") or symbol
    except Exception:
        return symbol


def fmt_vol(n):
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.2f}{suffix}"
    return f"{n:.0f}"


def style(fig, intraday, height, title):
    fig.update_layout(height=height, title=title, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, showlegend=False, template=THEME["template"],
                      paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"])
    if intraday:
        fig.update_xaxes(type="category", nticks=8)
    else:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def add_level(fig, lvl, color, tag):
    fig.add_hline(y=lvl["price"], line_color=color, opacity=0.7,
                  line_width=1 + min(lvl["touches"], 5) * 0.4,
                  annotation_text=f"{tag} {lvl['price']:,.2f} ({lvl['touches']}x)",
                  annotation_position="top left", annotation_font_color=color)


def candles(df, x):
    return go.Candlestick(x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                          increasing_line_color=GREEN, decreasing_line_color=RED)


# ------------------------------------------------------------------ inputs
with st.sidebar.form("query"):
    symbol = st.text_input("Stock symbol", value="AAPL").strip().upper()
    timeframe = st.radio("Timeframe", list(an.TIMEFRAMES), index=2)
    st.form_submit_button("Analyze")
st.sidebar.caption("Data: Yahoo Finance (cached 5 min). Support/resistance always uses 1 year of daily bars.")

if not symbol:
    st.info("Enter a stock symbol in the sidebar.")
    st.stop()

tf = an.TIMEFRAMES[timeframe]
intraday = tf["interval"] != "1d"
try:
    with st.spinner(f"Downloading {symbol}..."):
        daily = load(symbol, "1y", "1d")
        bars = load(symbol, tf["period"], tf["interval"])
except an.DataError as e:
    st.error(str(e))
    st.stop()

if len(daily) < 60:
    st.error(f"'{symbol}' has less than 60 days of history, which is not enough to analyze.")
    st.stop()

# ---------------------------------------------------------------- analysis
bars = bars.join(an.macd(bars["Close"]))
bars["RSI"] = an.rsi(bars["Close"])
window = an.trim_window(bars, timeframe)

price = float(daily["Close"].iloc[-1])
prev = float(daily["Close"].iloc[-2])
supports, resistances = an.support_resistance(daily)
vol = an.volume_stats(window, daily)
rec = an.recommend(price, supports, resistances, window["RSI"], window[["macd", "signal", "hist"]], vol)

# ------------------------------------------------------------------ header
st.title(f"{company_name(symbol)} ({symbol})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"${price:,.2f}", f"{price - prev:+.2f} ({(price / prev - 1) * 100:+.2f}%)")
c2.metric("52-week low", f"${daily['Low'].min():,.2f}")
c3.metric("52-week high", f"${daily['High'].max():,.2f}")
c4.metric("Timeframe", timeframe, f"{tf['interval']} bars", delta_color="off")

# ---------------------------------------------------------- recommendation
color = {"Bullish": GREEN, "Neutral": ORANGE, "Bearish": RED}[rec["signal"]]
with st.container(border=True):
    st.markdown(f"### Recommendation: <span style='color:{color}'>{rec['signal']}</span>",
                unsafe_allow_html=True)
    st.markdown(f"**{rec['action']}**")
    a, b, c, d = st.columns(4)
    a.metric("Buy price", f"${rec['buy']:,.2f}", f"{(rec['buy'] / price - 1) * 100:+.1f}% vs now", delta_color="off")
    b.metric("Sell price (target)", f"${rec['sell']:,.2f}", f"{(rec['sell'] / price - 1) * 100:+.1f}% vs now", delta_color="off")
    c.metric("Stop-loss", f"${rec['stop']:,.2f}", "2% below support", delta_color="off")
    d.metric("Reward : risk", f"{rec['reward_risk']:.1f} : 1" if rec["reward_risk"] else "n/a")
    with st.expander("Why this recommendation?", expanded=True):
        for r in rec["reasons"]:
            st.markdown(f"- {r}")
    st.caption("For learning only. This is not financial advice. Do your own research before trading.")

# ------------------------------------------------------------ price charts
tab_year, tab_window = st.tabs(["1 year: support & resistance", f"Selected window: {timeframe}"])
with tab_year:
    fig = go.Figure(candles(daily, daily.index))
    for lvl in supports[:5]:
        add_level(fig, lvl, GREEN, "S")
    for lvl in resistances[:5]:
        add_level(fig, lvl, RED, "R")
    fig.add_hline(y=rec["buy"], line_dash="dot", line_color=BLUE, annotation_text="Buy",
                  annotation_position="bottom right", annotation_font_color=BLUE)
    if not intraday:
        fig.add_vrect(x0=window.index[0], x1=window.index[-1], fillcolor=BLUE, opacity=0.07, line_width=0)
    st.plotly_chart(style(fig, False, 560, "Daily price, past year (shaded = selected window)"),
                    use_container_width=True, theme=None)

x = window.index.strftime("%b %d %H:%M") if intraday else window.index
with tab_window:
    fig = go.Figure(candles(window, x))
    lo, hi = window["Low"].min() * 0.97, window["High"].max() * 1.03
    for lvl in supports[:3]:
        if lvl["price"] >= lo:
            add_level(fig, lvl, GREEN, "S")
    for lvl in resistances[:3]:
        if lvl["price"] <= hi:
            add_level(fig, lvl, RED, "R")
    st.plotly_chart(style(fig, intraday, 480, f"Price, {timeframe} ({tf['interval']} bars)"),
                    use_container_width=True, theme=None)

# -------------------------------------------------------------- indicators
left, right = st.columns(2)
with left:
    fig = go.Figure(go.Scatter(x=x, y=window["RSI"], line_color=BLUE))
    fig.add_hline(y=70, line_dash="dash", line_color=RED, annotation_text="Overbought 70")
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, annotation_text="Oversold 30")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(style(fig, intraday, 320, f"RSI (14): now {rec['rsi']:.1f}"), use_container_width=True, theme=None)
with right:
    fig = go.Figure([
        go.Bar(x=x, y=window["hist"], marker_color=[GREEN if v >= 0 else RED for v in window["hist"]], name="Histogram"),
        go.Scatter(x=x, y=window["macd"], line_color=BLUE, name="MACD"),
        go.Scatter(x=x, y=window["signal"], line_color=ORANGE, name="Signal"),
    ])
    style(fig, intraday, 320, "MACD (12, 26, 9): blue = MACD, orange = signal")
    st.plotly_chart(fig, use_container_width=True, theme=None)

# ------------------------------------------------------------------ volume
st.subheader("Volume")
v1, v2, v3 = st.columns(3)
v1.metric(f"Avg daily volume ({vol['sessions']} sessions)", fmt_vol(vol["avg_window"]))
v2.metric("50-day avg daily volume", fmt_vol(vol["avg_50d"]))
v3.metric("Window vs 50-day", f"{vol['ratio']:.2f}x")
fig = go.Figure(go.Bar(x=x, y=window["Volume"], marker_color=GREY))
avg_line = vol["avg_per_bar"] if intraday else vol["avg_50d"]
fig.add_hline(y=avg_line, line_dash="dash", line_color=ORANGE,
              annotation_text="Avg per bar" if intraday else "50-day avg")
st.plotly_chart(style(fig, intraday, 280, f"Volume, {timeframe}"), use_container_width=True, theme=None)
if intraday:
    st.caption("Today's volume is partial while the market is open.")

# ------------------------------------------------------------- level table
st.subheader("All support & resistance levels (past year)")
rows = [{"Type": "Resistance", **l} for l in resistances][::-1] + [{"Type": "Support", **l} for l in supports]
table = pd.DataFrame(rows)
table["Distance"] = (table["price"] / price - 1) * 100
table = table.rename(columns={"price": "Price", "touches": "Touches", "note": "Note"})
st.dataframe(table[["Type", "Price", "Distance", "Touches", "Note"]], hide_index=True, use_container_width=True,
             column_config={"Price": st.column_config.NumberColumn(format="$%.2f"),
                            "Distance": st.column_config.NumberColumn("Distance from price", format="%+.1f%%")})
