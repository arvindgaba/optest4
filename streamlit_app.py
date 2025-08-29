# combined_nifty_atm0909_vwap.py
# NIFTY ΔOI Imbalance + TradingView VWAP alert
# - ATM: TV 09:09 → NSE underlying (provisional)
# - TV loop immediately upgrades ATM when 09:09 appears
# - Manual ATM override in sidebar
# - OC loop reloads ATM store every cycle
# - Weekday neighbors: Fri/Sat/Sun ±5, Mon ±4, Tue ±3, Wed ±2, Thu ±1
# - VWAP 15m session from TV 1m candles
# - Full logging + CSV/text outputs
# - ENHANCED: Multi-factor scoring (OI, Volume, IV), Dynamic Thresholds, RSI/ADX indicators
# - INTEGRATED: Telegram alerts, Signal History, Dual Timezone UI
# - ZERODHA INTEGRATION: Fetch live option price on signal trigger

import os, json, time, base64, datetime as dt, pathlib, threading, warnings, logging, sys, math, random
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
import certifi
import requests, urllib3
from kiteconnect import KiteConnect

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= USER SETTINGS =================
APP_VERSION = "2.2.4" # Card-based UI and Imbalance Metric
SYMBOL               = "NIFTY"
FETCH_EVERY_SECONDS  = 60          # option-chain poll (1 min)
TV_FETCH_SECONDS     = 60           # TradingView poll (1 min)
AUTOREFRESH_MS       = 10_000

OUT_DIR              = pathlib.Path.home() / "Documents" / "NSE_output"
CSV_PATH             = OUT_DIR / "nifty_currweek_change_oi_atm_dynamic.csv"
ATM_STORE_PATH       = OUT_DIR / "nifty_atm_store.json"
LOG_PATH             = OUT_DIR / "nifty_app.log"
ACCESS_TOKEN_PATH    = OUT_DIR / "zerodha_access_token.txt"


MAX_NEIGHBORS_LIMIT  = 20
IMBALANCE_TRIGGER    = 30.0         # % - This will be adjusted dynamically
VWAP_TOLERANCE_PTS   = 10.0          # alert when |spot - vwap| <= tolerance
RSI_PERIOD           = 14
ADX_PERIOD           = 14
NIFTY_LOT_SIZE       = 50

# ---- HARD-CODED TradingView credentials (REPLACE THESE) ----
TV_USERNAME          = "dileep.marchetty@gmail.com"
TV_PASSWORD          = "1dE6Land@123"
# ============================================================

# ---- TELEGRAM SETTINGS ----
TELEGRAM_BOT_TOKEN   = "1849589360:AAFW_O3pxt6NZJvoV-NeUMfqu90wIyP8bSA"
TELEGRAM_CHAT_ID     = "1887957750"
TELEGRAM_API_URL     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
# ===========================

API_URL  = "https://www.nseindia.com/api/option-chain-indices"
IST      = dt.timezone(dt.timedelta(hours=5, minutes=30))
UAE      = dt.timezone(dt.timedelta(hours=4))

BASE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"https://www.nseindia.com/option-chain?symbol={SYMBOL}",
}

# ensure certifi is used by libs that honor SSL_CERT_FILE
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# --- tiny 0.1s beep WAV (base64) ---
BEEP_WAV_B64 = (
    "UklGRmYAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAABhY2NkZGdn"
    "aGhoaWlpamptbW1tbm5ub29wcHBxcXFycnJzc3N0dHR1dXV2dnZ3d3d4eHh5"
    "eXl6enp7e3t8fHx9fX1+fn5/f3+AgICAgoKCg4ODhISEhYWFhoaGiIiIkJCQ"
    "kZGRkpKSlJSUlZWVmZmZmpqamsrKy8vLzMzMzc3Nzs7O0NDQ0dHR0lJSU1NT"
    "U9PT1NTU1dXV1paWmZmZmpqam5ubnBwcHJycnR0dHZ2dnd3d3h4eXl5enp6f"
    "Hx8fX19fn5+f39/gICA"
)

# ---------------- Logging ----------------
def setup_logger():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nifty_app")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); ch.setLevel(logging.INFO)

    logger.addHandler(fh); logger.addHandler(ch)
    logger.info("Logger initialized. Log file: %s", LOG_PATH)
    return logger

log = setup_logger()

# ---------------- Telegram Alert Function ----------------
def send_telegram_alert(message: str) -> bool:
    """Send alert message to Telegram. Returns True if successful."""
    try:
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(TELEGRAM_API_URL, data=payload, timeout=10)
        if response.status_code == 200:
            log.info("Telegram alert sent successfully")
            return True
        else:
            log.error("Telegram alert failed: HTTP %s - %s", response.status_code, response.text)
            return False
    except Exception as e:
        log.error("Telegram alert exception: %s", e)
        return False

def format_vwap_alert_message(alert: str, spot: float, vwap: float, suggestion: str,
                             atm_strike: int, expiry: str, final_score: float,
                             rsi: float, adx: float, timestamp: str, option_price: float = None) -> str:
    """Format a comprehensive VWAP alert message for Telegram."""
    emoji = "🚨" if "BUY" in suggestion else "📊"
    direction_emoji = "📈" if "CALL" in suggestion else "📉" if "PUT" in suggestion else "⚪"
    
    price_info = ""
    if option_price is not None:
        lot_value = option_price * NIFTY_LOT_SIZE
        price_info = f"""
✅ <b>Option Price (LTP):</b> ₹{option_price:,.2f}
📦 <b>Lot Value:</b> ₹{lot_value:,.2f}
"""

    message = f"""
{emoji} <b>NFS LIVE v{APP_VERSION} - VWAP ALERT</b> {emoji}

{direction_emoji} <b>Signal:</b> {suggestion}
📍 <b>Alert:</b> {alert}
{price_info}
💰 <b>Market Data:</b>
• Spot Price: ₹{spot:,.2f}
• VWAP (15m): ₹{vwap:,.2f}
• Difference: ₹{abs(spot - vwap):,.2f}

⚖️ <b>Options & Technicals:</b>
• ATM Strike: {atm_strike}
• Expiry: {expiry}
• Final Score: {final_score*100:+.2f}%
• RSI: {rsi:.2f}
• ADX: {adx:.2f}

🕐 <b>Time (IST):</b> {timestamp}
🇦🇪 <b>Time (UAE):</b> {dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST).astimezone(UAE).strftime("%Y-%m-%d %H:%M:%S")}

<i>Trade at your own risk. This is for educational purposes only.</i>
    """.strip()
    
    return message

def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)

def now_uae() -> dt.datetime:
    return dt.datetime.now(UAE)

def format_datetime_compact(dt_ist: dt.datetime) -> str:
    """Format datetime compactly for Streamlit display - single line format."""
    if dt_ist is None:
        return "—"
    dt_uae = dt_ist.astimezone(UAE)
    ist_str = dt_ist.strftime("%m-%d %H:%M:%S")
    uae_str = dt_uae.strftime("%H:%M")
    return f"{ist_str} IST / {uae_str} UAE"

def today_str() -> str:
    return now_ist().strftime("%Y%m%d")

def load_atm_store() -> dict:
    if ATM_STORE_PATH.exists():
        try:
            return json.loads(ATM_STORE_PATH.read_text())
        except Exception as e:
            log.error("Failed to read ATM store: %s", e)
    return {}

def save_atm_store(store: dict):
    try:
        ATM_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ATM_STORE_PATH.write_text(json.dumps(store, indent=2))
    except Exception as e:
        log.error("Failed to write ATM store: %s", e)

def update_store_atm(atm: int, base_value: float, status: str):
    """Atomic update of the ATM store (used by both loops)."""
    store = load_atm_store()
    store.update({
        "date": today_str(),
        "atm_strike": int(atm),
        "base_value": float(base_value),
        "atm_status": status
    })
    save_atm_store(store)
    log.info("Store ATM updated -> %s (%s, base=%.2f)", atm, status, base_value)

# ---------------- NSE OPTION-CHAIN ----------------
def new_session():
    try:
        import cloudscraper
        warnings.filterwarnings("ignore", category=UserWarning, module="cloudscraper")
        s = cloudscraper.create_scraper(delay=8, browser={'browser':'chrome','platform':'windows'})
        log.info("Created cloudscraper session")
    except ModuleNotFoundError:
        import requests as _rq
        s = _rq.Session()
        log.warning("cloudscraper not installed; using requests.Session")
    s.headers.update(BASE_HEADERS)
    try:
        s.get(f"https://www.nseindia.com/option-chain?symbol={SYMBOL}", timeout=8)
    except Exception as e:
        log.warning("Handshake to NSE failed (continuing): %s", e)
    return s

def pick_current_week_expiry(raw: dict) -> str | None:
    """
    Selects the nearest weekly (Thursday) expiry. Falls back to the soonest
    available future date if no Thursday is found (for monthlys/holidays).
    """
    today = now_ist().date()
    parsed = []
    expiry_dates = raw.get("records", {}).get("expiryDates", [])
    if not expiry_dates:
        log.error("No expiryDates in JSON from NSE.")
        return None

    for s in expiry_dates:
        try:
            parsed.append((s, dt.datetime.strptime(s, "%d-%b-%Y").date()))
        except Exception:
            pass
    
    if not parsed:
        log.error("Could not parse any expiry dates.")
        return None

    future_expiries = [p for p in parsed if p[1] >= today]
    if not future_expiries:
        log.warning("No future expiry dates found. Using the latest available past expiry.")
        return parsed[-1][0] if parsed else None

    thursday_expiries = [p for p in future_expiries if p[1].weekday() == 3]

    if thursday_expiries:
        chosen = min(thursday_expiries, key=lambda x: x[1])
        log.info(f"Selected Thursday expiry: {chosen[0]}")
        return chosen[0]
    else:
        chosen = min(future_expiries, key=lambda x: x[1])
        log.warning(f"No Thursday expiry found. Using soonest available future date: {chosen[0]}")
        return chosen[0]


def round_to_50(x: float) -> int:
    return int(round(x / 50.0) * 50)

@st.cache_resource
def oc_session_cached():
    return new_session()

def fetch_raw_option_chain():
    s = oc_session_cached()
    last_err = None
    for i in range(6):
        try:
            r = s.get(API_URL, params={"symbol": SYMBOL}, timeout=10)
            if r.status_code == 200:
                try:
                    raw = r.json()
                    if "records" in raw and "data" in raw["records"]:
                        log.info("OC fetch OK on attempt %d", i+1)
                        return raw
                    log.warning("OC JSON missing keys on attempt %d", i+1)
                except json.JSONDecodeError as e:
                    log.warning("OC JSON decode error on attempt %d: %s", i+1, e)
            else:
                log.warning("OC HTTP %s on attempt %d", r.status_code, i+1)
        except Exception as e:
            last_err = e
            log.warning("OC fetch exception on attempt %d: %s", i+1, e)
        time.sleep(2)
        if i == 2:
            try:
                oc_session_cached.clear_cache()
                s = oc_session_cached()
                log.info("Recreated NSE session")
            except Exception as ee:
                log.warning("Failed to recreate NSE session: %s", ee)
    log.error("OC fetch failed after retries: %s", last_err)
    return None

# ---------------- TradingView helpers ----------------
@st.cache_resource
def tv_login():
    from tvDatafeed import TvDatafeed
    try:
        tv = TvDatafeed(username=TV_USERNAME, password=TV_PASSWORD)
        log.info("Logged in to TradingView as %s", TV_USERNAME)
        return tv
    except Exception as e:
        log.error("TradingView login failed: %s", e)
        raise

def fetch_tv_1m_session(n_bars: int = 500):
    try:
        from tvDatafeed import Interval
    except Exception as e:
        log.error("tvDatafeed import failed: %s", e)
        return None
    last_err = None
    for i in range(1, 4):
        try:
            tv = tv_login()
            df = tv.get_hist(symbol="NIFTY", exchange="NSE", interval=Interval.in_1_minute, n_bars=n_bars)
            if df is not None and not df.empty:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
                else:
                    df.index = df.index.tz_convert("Asia/Kolkata")
                if "symbol" in df.columns:
                    df = df.drop(columns=["symbol"])
                log.info("TV 1m bars fetched. Last: %s", df.index.max())
                return df
            last_err = "empty dataframe"
            log.warning("TV 1m fetch attempt %d: empty dataframe", i)
        except Exception as e:
            last_err = e
            log.warning("TV 1m fetch attempt %d failed: %s", i, e)
        time.sleep(2 * i)
    log.error("TV 1m fetch failed after retries: %s", last_err)
    return None

def fetch_tv_15m_session(n_bars: int = 64):
    try:
        from tvDatafeed import Interval
    except Exception as e:
        log.error("tvDatafeed import failed (15m): %s", e)
        return None
    try:
        tv = tv_login()
        df = tv.get_hist(symbol="NIFTY", exchange="NSE", interval=Interval.in_15_minute, n_bars=n_bars)
        if df is None or df.empty:
            log.warning("TV 15m fetch returned empty dataframe")
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])
        log.info("TV 15m bars fetched. Last: %s", df.index.max())
        return df
    except Exception as e:
        log.error("TV 15m fetch failed: %s", e)
        return None

def compute_tv_vwap(df: pd.DataFrame, period_len: int = 14) -> float | None:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        log.error("compute_tv_vwap: invalid DataFrame")
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    win = df.tail(period_len)
    if win.shape[0] == 0: return None
    if win.shape[0] < period_len:
        log.warning("compute_tv_vwap: only %d of %d bars available", win.shape[0], period_len)
    tp = (win["high"] + win["low"] + win["close"]) / 3.0
    vol = win["volume"].fillna(0).astype(float)
    tpV = tp * vol
    sum_tpV = tpV.sum()
    sum_vol = vol.sum()
    if sum_vol == 0: return float(tp.mean())
    return float(sum_tpV / sum_vol)

def price_at_0909(df_1m: pd.DataFrame) -> float | None:
    if df_1m is None or df_1m.empty: return None
    latest_date = df_1m.index.max().date()
    t909 = dt.datetime.combine(latest_date, dt.time(9, 9), tzinfo=IST)
    try:
        if t909 in df_1m.index: return float(df_1m.loc[t909, "close"])
        win = df_1m.between_time("09:05", "09:14")
        if not win.empty and win.index.date.max() == latest_date:
            idx = min(win.index, key=lambda t: abs((t - t909).total_seconds()))
            return float(win.loc[idx, "close"])
        t915 = dt.datetime.combine(latest_date, dt.time(9, 15), tzinfo=IST)
        if t915 in df_1m.index: return float(df_1m.loc[t915, "open"])
    except Exception as e:
        log.error("price_at_0909 error: %s", e)
    return None

# ---------------- Technical Indicators ----------------
def calculate_rsi(data: pd.Series, period: int) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # Return 50 (neutral) if calculation fails

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean().replace(0, 1e-10)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)) * 100
    adx = dx.ewm(alpha=1/period).mean()
    return adx.fillna(20) # Return 20 (neutral) if calculation fails

# ---------------- Weekday neighbors mapping ----------------
def neighbors_by_weekday(d: dt.date) -> int:
    wd = d.weekday()
    mapping = {0: 4, 1: 3, 2: 2, 3: 1, 4: 5, 5: 5, 6: 5}
    return mapping.get(wd, 3)

def nearest_strike_block(strikes_sorted: list[int], atm: int, neighbors_each: int) -> list[int]:
    if not strikes_sorted: return []
    if atm not in strikes_sorted:
        atm = min(strikes_sorted, key=lambda x: abs(x - atm))
    idx = strikes_sorted.index(atm)
    lo = max(0, idx - neighbors_each)
    hi = min(len(strikes_sorted) - 1, idx + neighbors_each)
    return strikes_sorted[lo:hi+1]

# ---------------- Build OC DF with imbalance + ATM logic ----------------
def build_df_with_imbalance(raw: dict, store: dict):
    store = load_atm_store()
    if not raw: return pd.DataFrame(), None
    expiry = pick_current_week_expiry(raw)
    if not expiry: return pd.DataFrame(), None
    records = raw["records"]
    rows = [x for x in records["data"] if x.get("expiryDate") == expiry]
    if not rows:
        log.error("No rows for chosen expiry %s", expiry)
        return pd.DataFrame(), None
    df_all = pd.json_normalize(rows)
    if "strikePrice" not in df_all.columns:
        log.error("strikePrice missing")
        return pd.DataFrame(), None
    strikes_all = sorted({int(v) for v in df_all["strikePrice"].dropna().astype(int)})
    underlying = float(records.get("underlyingValue", 0.0))
    today_date  = now_ist().date()
    today_key   = today_str()

    def capture_today_atm_tv_0909():
        df1 = fetch_tv_1m_session()
        px909 = price_at_0909(df1) if df1 is not None else None
        if px909 and px909 > 0:
            base_val = float(px909)
            guess = round_to_50(base_val)
            atm_local = guess if guess in strikes_all else min(strikes_all, key=lambda x: abs(x - base_val))
            log.info("ATM capture(09:09 TV): base=%.2f atm=%s", base_val, atm_local)
            return int(atm_local), base_val, "captured-0909"
        return None, None, "capture-failed"

    def capture_today_atm_underlying():
        base_val = underlying
        guess = round_to_50(base_val)
        atm_local = guess if guess in strikes_all else min(strikes_all, key=lambda x: abs(x - base_val))
        log.warning("ATM provisional from underlying: base=%.2f atm=%s", base_val, atm_local)
        return int(atm_local), float(base_val), "provisional"

    stored_date   = store.get("date")
    stored_atm    = store.get("atm_strike")
    stored_status = store.get("atm_status", "unknown")
    need_fresh = (stored_date != today_key)

    if need_fresh:
        atm_strike, base_val, atm_status = None, None, "capture-failed"
        for capt in (capture_today_atm_tv_0909, capture_today_atm_underlying):
            a,b,s = capt()
            if a is not None:
                atm_strike, base_val, atm_status = a,b,s
                break
        if atm_strike is not None:
            update_store_atm(atm_strike, base_val, atm_status)
    else:
        atm_strike = int(stored_atm)
        atm_status = stored_status
        base_val   = store.get("base_value", 0.0)
        log.info("Using ATM: %s (%s)", atm_strike, atm_status)

    neighbors_each = neighbors_by_weekday(today_date)
    neighbors_each = min(neighbors_each, MAX_NEIGHBORS_LIMIT)
    wanted = set(nearest_strike_block(strikes_all, atm_strike, neighbors_each))
    log.info("Neighbors: weekday=%s ±%s, wanted_count=%s", today_date.weekday(), neighbors_each, len(wanted))

    required_cols = ["CE.changeinOpenInterest", "PE.changeinOpenInterest", "CE.totalTradedVolume", "PE.totalTradedVolume", "CE.impliedVolatility", "PE.impliedVolatility"]
    for c in required_cols:
        if c not in df_all.columns: df_all[c] = 0

    df = df_all[["strikePrice"] + required_cols].rename(columns={
        "strikePrice": "Strike", "CE.changeinOpenInterest": "Call Chg OI", "PE.changeinOpenInterest": "Put Chg OI",
        "CE.totalTradedVolume": "Call Volume", "PE.totalTradedVolume": "Put Volume",
        "CE.impliedVolatility": "Call IV", "PE.impliedVolatility": "Put IV",
    })
    df = df[df["Strike"].isin(wanted)].copy()
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce").astype("Int64")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Strike").reset_index(drop=True)

    # Multi-factor scoring & simple imbalance
    oi_call_sum = float(df["Call Chg OI"].sum(skipna=True))
    oi_put_sum  = float(df["Put Chg OI"].sum(skipna=True))
    vol_call_sum = float(df["Call Volume"].sum(skipna=True))
    vol_put_sum  = float(df["Put Volume"].sum(skipna=True))
    iv_call_avg = float(df["Call IV"].mean(skipna=True))
    iv_put_avg  = float(df["Put IV"].mean(skipna=True))

    oi_denominator = oi_put_sum + oi_call_sum
    imbalance_pct = ((oi_put_sum - oi_call_sum) / oi_denominator) * 100 if oi_denominator != 0 else 0

    oi_imbalance = (oi_put_sum - oi_call_sum) / oi_denominator if oi_denominator != 0 else 0
    vol_imbalance = (vol_put_sum - vol_call_sum) / (vol_put_sum + vol_call_sum) if (vol_put_sum + vol_call_sum) != 0 else 0
    iv_imbalance = (iv_put_avg - iv_call_avg) / (iv_put_avg + iv_call_avg) if (iv_put_avg + iv_call_avg) != 0 else 0
    
    w_oi, w_vol, w_iv = 0.5, 0.3, 0.2
    final_score = (w_oi * oi_imbalance) + (w_vol * vol_imbalance) + (w_iv * iv_imbalance)
    
    market_volatility = df['Call IV'].mean()
    dynamic_trigger = IMBALANCE_TRIGGER * (1 + market_volatility / 100) if pd.notna(market_volatility) else IMBALANCE_TRIGGER
    
    suggestion = "NO SIGNAL"
    if abs(final_score * 100) > dynamic_trigger:
        suggestion = "BUY PUT" if final_score < 0 else "BUY CALL"

    updated_str = now_ist().strftime("%Y-%m-%d %H:%M:%S")
    df.insert(0, "ATM", atm_strike)
    df.insert(0, "Expiry", expiry)
    df.insert(0, "Updated", updated_str)
    df["Final Score"]   = round(final_score * 100, 2)
    df["Suggestion"]    = suggestion

    latest_store = load_atm_store()
    atm_status_disp = latest_store.get("atm_status", "unknown")
    base_value_disp = latest_store.get("base_value", None)

    meta = {
        "neighbors_each": neighbors_each, "underlying": underlying,
        "final_score": final_score, "suggestion": suggestion, "expiry": expiry,
        "atm": atm_strike, "updated": updated_str, "atm_status": atm_status_disp,
        "base_value": base_value_disp, "dynamic_trigger": dynamic_trigger,
        "oi_call_sum": oi_call_sum, "oi_put_sum": oi_put_sum,
        "imbalance_pct": imbalance_pct # Add simple imbalance for graphing
    }
    log.info("Score: final=%.2f%% sugg=%s; ATM=%s (%s)", final_score * 100, suggestion, atm_strike, atm_status_disp)
    return df, meta

# ---------------- Signal History & Memory Store ----------------
class SignalHistory:
    def __init__(self):
        self.lock = threading.Lock()
        self.signals: list[dict] = []
        self.history_file = OUT_DIR / "signal_history.json"
        self.load_history()
    
    def load_history(self):
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    today = now_ist().date().isoformat()
                    self.signals = [s for s in data if s.get('date') == today]
        except Exception as e:
            log.error("Failed to load signal history: %s", e)
            self.signals = []
    
    def save_history(self):
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.signals, f, indent=2)
        except Exception as e:
            log.error("Failed to save signal history: %s", e)
    
    def add_signal(self, **kwargs):
        timestamp_ist = now_ist()
        signal_data = {
            'id': len(self.signals) + 1,
            'date': timestamp_ist.date().isoformat(),
            'timestamp_ist': timestamp_ist.strftime("%Y-%m-%d %H:%M:%S"),
            'timestamp_uae': timestamp_ist.astimezone(UAE).strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs
        }
        with self.lock:
            self.signals.append(signal_data)
            self.save_history()
        log.info("Signal added to history: %s at %s IST", kwargs.get('suggestion'), signal_data['timestamp_ist'])
    
    def to_dataframe(self) -> pd.DataFrame:
        with self.lock:
            return pd.DataFrame(self.signals) if self.signals else pd.DataFrame()

class IntradayImbalanceSeries:
    def __init__(self, max_points: int = 480):
        self.lock = threading.Lock()
        self.points: list[tuple[dt.datetime, float]] = []
        self.max_points = max_points
    def add_point(self, ts: dt.datetime, imbalance: float):
        if not (dt.time(9, 0) <= ts.time() <= dt.time(16, 0)): return
        today = now_ist().date()
        with self.lock:
            self.points = [(t, v) for t, v in self.points if t.date() == today]
            self.points.append((ts, imbalance))
            if len(self.points) > self.max_points:
                self.points = self.points[-self.max_points:]
    def to_dataframe(self) -> pd.DataFrame:
        with self.lock:
            if not self.points: return pd.DataFrame()
            df = pd.DataFrame(self.points, columns=["ts", "imbalance_pct"])
            df.set_index("ts", inplace=True)
            return df

class StoreMem:
    def __init__(self):
        self.lock = threading.Lock()
        self.df_opt: pd.DataFrame | None = None
        self.meta_opt: dict = {}
        self.last_opt: dt.datetime | None = None
        self.vwap_latest: float | None = None
        self.last_tv: dt.datetime | None = None
        self.rsi: float | None = None
        self.adx: float | None = None
        self.vwap_alert: str = "NO ALERT"
        self.last_alert_key: str = ""
        self.last_telegram_alert: str = ""
        self.intraday = IntradayImbalanceSeries()
        self.signal_history = SignalHistory()
        self.kite: KiteConnect | None = None

# ---------------- Worker Loops ----------------
def option_chain_loop(mem: StoreMem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            raw = fetch_raw_option_chain()
            if raw:
                df, meta = build_df_with_imbalance(raw, {})
                if not df.empty and meta:
                    imbalance_pct = meta.get("imbalance_pct")
                    if imbalance_pct is not None:
                        mem.intraday.add_point(now_ist(), float(imbalance_pct))
                    with mem.lock:
                        mem.df_opt, mem.meta_opt, mem.last_opt = df, dict(meta), now_ist()
                    try:
                        df.to_csv(CSV_PATH, index=False)
                    except Exception as e:
                        log.error("Write CSV failed: %s", e)
                    log.info("[OC] wrote %d rows", len(df))
                else:
                    log.warning("[OC] empty dataframe this cycle")
        except Exception as e:
            log.exception("OptionChain loop error: %s", e)
        time.sleep(FETCH_EVERY_SECONDS)

def get_option_ltp(kite: KiteConnect, atm_strike: int, expiry: str, option_type: str) -> float | None:
    """Constructs the trading symbol and fetches the LTP from Zerodha."""
    if not kite: return None
    try:
        expiry_dt = dt.datetime.strptime(expiry, "%d-%b-%Y")
        year = expiry_dt.strftime("%y")
        month = expiry_dt.month
        day = expiry_dt.strftime("%d")

        month_char_map = {10: 'O', 11: 'N', 12: 'D'}
        month_char = month_char_map.get(month, str(month))
        
        expiry_zerodha = f"{year}{month_char}{day}"
        
        instrument = f"NIFTY{expiry_zerodha}{atm_strike}{option_type}"
        trading_symbol = f"NFO:{instrument}"
        
        quote = kite.ltp(trading_symbol)
        if quote and trading_symbol in quote:
            ltp = quote[trading_symbol]['last_price']
            log.info(f"Fetched LTP for {trading_symbol}: {ltp}")
            return ltp
        log.warning(f"LTP not found for {trading_symbol}")
        return None
    except Exception as e:
        log.error(f"Error fetching LTP for expiry {expiry}: {e}")
        return None

def tradingview_loop(mem: StoreMem):
    while True:
        try:
            df_1m, df_15m = fetch_tv_1m_session(), fetch_tv_15m_session()
            if df_1m is None or df_1m.empty:
                log.warning("[TV] No 1m data, skipping cycle.")
                time.sleep(TV_FETCH_SECONDS)
                continue
            
            px909 = price_at_0909(df_1m)
            if px909:
                base_val, atm_guess = float(px909), round_to_50(px909)
                store = load_atm_store()
                if (store.get("date") != today_str() or store.get("atm_status") != "captured-0909" or int(store.get("atm_strike", 0)) != atm_guess):
                    update_store_atm(atm_guess, base_val, "captured-0909")
                    log.info("ATM upgraded to %s (base %.2f) by TV-loop", atm_guess, base_val)
                    raw_now = fetch_raw_option_chain()
                    if raw_now:
                        df_now, meta_now = build_df_with_imbalance(raw_now, {})
                        if not df_now.empty:
                            with mem.lock:
                                mem.df_opt, mem.meta_opt, mem.last_opt = df_now.copy(), dict(meta_now), now_ist()
                            try:
                                df_now.to_csv(CSV_PATH, index=False)
                            except Exception as e:
                                log.error("CSV write failed (TV-trigger): %s", e)

            vwap_latest = compute_tv_vwap(df_15m, period_len=14) if df_15m is not None else None
            rsi = calculate_rsi(df_1m['close'], RSI_PERIOD).iloc[-1]
            adx = calculate_adx(df_1m['high'], df_1m['low'], df_1m['close'], ADX_PERIOD).iloc[-1]

            with mem.lock:
                mem.last_tv, mem.vwap_latest, mem.rsi, mem.adx = now_ist(), vwap_latest, rsi, adx
                meta = mem.meta_opt or {}
                spot, sugg = meta.get("underlying"), meta.get("suggestion", "NO SIGNAL")
            
            alert = "NO ALERT"
            if (vwap_latest is not None and spot is not None and sugg in ("BUY CALL", "BUY PUT") and
                abs(float(spot) - float(vwap_latest)) <= VWAP_TOLERANCE_PTS and
                ((sugg == "BUY CALL" and rsi > 50 and adx > 20) or (sugg == "BUY PUT" and rsi < 50 and adx > 20))):
                alert = f"{sugg} (spot near VWAP ±{VWAP_TOLERANCE_PTS})"
            
            with mem.lock: mem.vwap_alert = alert
            
            if alert != "NO ALERT" and alert != mem.last_telegram_alert:
                try:
                    option_price = None
                    with mem.lock:
                        if mem.kite:
                            option_type = "CE" if "CALL" in sugg else "PE"
                            option_price = get_option_ltp(mem.kite, meta.get("atm"), meta.get("expiry"), option_type)

                    timestamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
                    telegram_message = format_vwap_alert_message(
                        alert=alert, spot=float(spot), vwap=float(vwap_latest), suggestion=sugg,
                        atm_strike=int(meta.get("atm", 0)), expiry=meta.get("expiry", "N/A"),
                        final_score=float(meta.get("final_score", 0.0)), rsi=float(rsi), adx=float(adx),
                        timestamp=timestamp, option_price=option_price
                    )
                    telegram_success = send_telegram_alert(telegram_message)
                    if telegram_success:
                        with mem.lock: mem.last_telegram_alert = alert
                    
                    mem.signal_history.add_signal(
                        alert=alert, suggestion=sugg, spot_price=round(float(spot), 2),
                        vwap=round(float(vwap_latest), 2), difference=round(abs(float(spot) - float(vwap_latest)), 2),
                        atm_strike=int(meta.get("atm", 0)), expiry=meta.get("expiry", "N/A"),
                        final_score=round(float(meta.get("final_score", 0.0)) * 100, 2),
                        rsi=round(float(rsi), 2), adx=round(float(adx), 2),
                        telegram_sent=telegram_success,
                        option_price=option_price,
                        lot_value=option_price * NIFTY_LOT_SIZE if option_price else None
                    )
                except Exception as e:
                    log.error("Error preparing/sending Telegram alert: %s", e)

            stamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
            log.info("[TV] vwap=%s alert=%s", f"{vwap_latest:.2f}" if vwap_latest is not None else "None", alert)
        except Exception as e:
            with mem.lock: mem.last_tv, mem.vwap_latest = now_ist(), None
            log.exception("TradingView loop error: %s", e)
        time.sleep(TV_FETCH_SECONDS)

@st.cache_resource
def start_background(app_version: str) -> StoreMem:
    log.info(f"Starting background threads for app version {app_version}")
    mem = StoreMem()
    threading.Thread(target=option_chain_loop, args=(mem,), daemon=True, name="OC-Loop").start()
    threading.Thread(target=tradingview_loop, args=(mem,), daemon=True, name="TV-Loop").start()
    return mem

# ---------------- UI helpers ----------------
def play_beep_once_on_new_alert(mem: StoreMem, alert_text: str):
    key = f"{today_str()}|{alert_text}"
    if alert_text != "NO ALERT" and key != mem.last_alert_key:
        st.markdown(f'<audio autoplay><source src="data:audio/wav;base64,{BEEP_WAV_B64}" type="audio/wav"></audio>', unsafe_allow_html=True)
        mem.last_alert_key = key

def create_signal_gauge(score: float, trigger: float) -> go.Figure:
    """Creates a gauge chart for the Final Score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Signal Strength", 'font': {'size': 20}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [-100, 100]},
            'bar': {'color': "rgba(0,0,0,0.3)"},
            'steps' : [
                {'range': [-100, -trigger], 'color': "rgba(255, 75, 75, 0.7)"},
                {'range': [trigger, 100], 'color': "rgba(75, 255, 75, 0.7)"}],
            'threshold' : {'line': {'color': "black", 'width': 3}, 'thickness': 1, 'value': trigger}}))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=30))
    return fig

def create_vwap_gauge(diff: float, tolerance: float) -> go.Figure:
    """Creates a horizontal bullet gauge for the VWAP difference."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = diff,
        title = {'text': "VWAP Difference", 'font': {'size': 16}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'shape': "bullet",
            'axis' : {'range': [-100, 100]},
            'threshold' : {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': diff},
            'steps' : [
                {'range': [-tolerance, tolerance], 'color': "rgba(75, 255, 75, 0.7)"},
            ],
            'bar': {'color': 'rgba(0,0,0,0.3)'}}))
    fig.update_layout(height=100, margin=dict(l=30, r=30, t=40, b=20))
    return fig

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title=f"NFS LIVE v{APP_VERSION}", layout="wide")
st_autorefresh(interval=AUTOREFRESH_MS, key="nifty_autorefresh")
mem = start_background(APP_VERSION)

with mem.lock:
    df_live, meta, last_opt, vwap_latest, last_tv, vwap_alert, rsi, adx = \
        mem.df_opt, mem.meta_opt, mem.last_opt, mem.vwap_latest, mem.last_tv, mem.vwap_alert, mem.rsi, mem.adx

# Extract data from meta dictionary for UI
final_score = meta.get("final_score", 0.0)
suggestion = meta.get("suggestion", "NO SIGNAL")
spot = meta.get("underlying")
dynamic_trigger = meta.get("dynamic_trigger", IMBALANCE_TRIGGER)
expiry = meta.get("expiry", "N/A")
updated_str = meta.get("updated", "")
atm_strike = meta.get("atm", 0)
atm_status = meta.get("atm_status", "unknown")
base_value = meta.get("base_value")
neighbors_each = meta.get("neighbors_each", 0)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Zerodha Connection")
    try:
        api_key = st.secrets["zerodha"]["api_key"]
        api_secret = st.secrets["zerodha"]["api_secret"]
        kite = KiteConnect(api_key=api_key)

        if ACCESS_TOKEN_PATH.exists():
            access_token = ACCESS_TOKEN_PATH.read_text()
            try:
                kite.set_access_token(access_token)
                profile = kite.profile()
                st.success(f"Connected as {profile['user_name']}")
                with mem.lock:
                    mem.kite = kite
            except Exception as e:
                st.warning("Access token expired or invalid. Please re-login.")
                if ACCESS_TOKEN_PATH.exists(): ACCESS_TOKEN_PATH.unlink()
        else:
            st.info("Please login to Zerodha to fetch live prices.")
            login_url = kite.login_url()
            st.markdown(f"[Click here to generate request token]({login_url})", unsafe_allow_html=True)
            
            request_token = st.text_input("Paste Request Token here")
            if st.button("Generate Access Token"):
                try:
                    data = kite.generate_session(request_token, api_secret=api_secret)
                    access_token = data["access_token"]
                    ACCESS_TOKEN_PATH.write_text(access_token)
                    st.success("Access token generated successfully! Refreshing...")
                    time.sleep(2)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Authentication failed: {e}")

    except (KeyError, FileNotFoundError):
        st.error("Zerodha API credentials not found in secrets.toml")
        log.error("Zerodha secrets error")

    st.divider()
    st.header("Settings")
    VWAP_tol = st.number_input("VWAP tolerance (pts)", value=float(VWAP_TOLERANCE_PTS), step=1.0)
    IMB_thr  = st.number_input("Imbalance trigger (%)", value=float(IMBALANCE_TRIGGER), step=1.0)
    st.caption(f"Logs: `{LOG_PATH}`")
    st.divider()
    st.subheader("Manual LTP Test")
    test_strike = st.number_input("Strike Price", min_value=20000, max_value=30000, value=atm_strike, step=50)
    test_type = st.selectbox("Option Type", ["CE", "PE"])
    if st.button("Fetch Test LTP"):
        with mem.lock:
            kite_conn = mem.kite
        
        if kite_conn and expiry != "N/A":
            st.write(f"Fetching LTP for NIFTY {test_strike} {test_type} (Expiry: {expiry})...")
            ltp = get_option_ltp(kite_conn, test_strike, expiry, test_type)
            if ltp is not None:
                st.success(f"Last Traded Price: ₹{ltp:,.2f}")
            else:
                st.error("Failed to fetch LTP. Check logs or if the instrument is valid.")
        else:
            st.warning("Zerodha not connected or expiry not available.")

    st.divider()
    st.subheader("Telegram Alert Test")
    if st.button("Send Test Alert"):
        with mem.lock:
            meta = mem.meta_opt or {}
            vwap = mem.vwap_latest or 24485.0
            rsi = mem.rsi or 55.0
            adx = mem.adx or 25.0
        test_msg = format_vwap_alert_message(
            alert="BUY CALL (TEST ALERT)", spot=meta.get("underlying", 24500.0), vwap=vwap,
            suggestion="BUY CALL", atm_strike=meta.get("atm", 24500), expiry=meta.get("expiry", "09-Jan-2025"),
            final_score=0.355, rsi=rsi, adx=adx, timestamp=now_ist().strftime("%Y-%m-%d %H:%M:%S")
        )
        if send_telegram_alert(test_msg): st.success("✅ Test alert sent!")
        else: st.error("❌ Failed to send test alert.")
    
# --- MAIN PAGE ---
st.title(f"NFS LIVE v{APP_VERSION} - Multi-Factor NIFTY Analysis")

# Status row
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1:
    st.metric("Last OC", last_opt.strftime('%H:%M:%S') if last_opt else "—")
    if last_opt: st.caption(f"{last_opt.astimezone(UAE).strftime('%H:%M:%S')} UAE")
with c2:
    st.metric("Last TV", last_tv.strftime('%H:%M:%S') if last_tv else "—")
    if last_tv: st.caption(f"{last_tv.astimezone(UAE).strftime('%H:%M:%S')} UAE")
c3.metric("Spot", f"{spot:,.2f}" if spot else "—")
c4.metric("VWAP", f"{vwap_latest:,.2f}" if vwap_latest else "—")
c5.metric("RSI", f"{rsi:,.2f}" if rsi else "—")
c6.metric("ADX", f"{adx:,.2f}" if adx else "—")
c7.metric("ATM Strike", f"{atm_strike}" if atm_strike else "—")


if df_live is None or df_live.empty:
    st.warning("Waiting for first successful option-chain fetch…")
    st.stop()

# Alert Logic
imbalance_ok = abs(final_score * 100) > IMB_thr
vwap_ok = (vwap_latest is not None and spot is not None and abs(float(spot) - float(vwap_latest)) <= VWAP_tol)
combined_alert = "NO ALERT"
if suggestion in ("BUY CALL", "BUY PUT") and imbalance_ok and vwap_ok:
    combined_alert = f"{suggestion} (spot near VWAP ±{VWAP_tol})"

if combined_alert != "NO ALERT":
    st.success(f"VWAP ALERT: **{combined_alert}**", icon="✅")
    play_beep_once_on_new_alert(mem, combined_alert)
else:
    st.info("No active alert. Waiting for conditions to align.", icon="ℹ️")

# Detailed Context Caption
try:
    updated_dt = dt.datetime.strptime(updated_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
    updated_display = format_datetime_compact(updated_dt)
except (ValueError, TypeError):
    updated_display = f"{updated_str} IST"
base_disp = f"{base_value:,.2f}" if isinstance(base_value, (int, float)) else "—"

st.subheader(f"Expiry: {expiry}")
st.caption(
    f"Updated: **{updated_display}** • ATM: **{atm_strike}** (**{atm_status}**, base={base_disp}) • "
    f"Neighbors each side: **{neighbors_each}**"
)

# Key Metrics & Signal Gauge
k1, k2 = st.columns(2)
with k1:
    st.metric("Final Score", f"{final_score*100:,.2f}%")
    st.metric("Dynamic Trigger", f"{dynamic_trigger:,.2f}%")
with k2:
    # Clamp score for display in gauge to handle off-market anomalies
    display_score = np.clip(final_score * 100, -100, 100)
    st.plotly_chart(create_signal_gauge(display_score, dynamic_trigger), use_container_width=True)


# VWAP vs Spot Difference Gauge
if vwap_latest is not None and spot is not None:
    vwap_diff = spot - vwap_latest
    st.plotly_chart(create_vwap_gauge(vwap_diff, VWAP_tol), use_container_width=True)
    st.caption(f"VWAP: **{vwap_latest:,.2f}** •  Spot: **{spot:,.2f}** •  Diff: **{vwap_diff:+.2f}**")
else:
    st.caption("VWAP or Spot not available yet.")

# Main Data Table
st.dataframe(df_live, use_container_width=True, hide_index=True)

# Intraday Imbalance Trend Chart
df_trend = mem.intraday.to_dataframe()
if not df_trend.empty:
    df_trend = df_trend.between_time("09:15", "16:00")
    if not df_trend.empty:
        df_plot = df_trend.reset_index().rename(columns={"ts": "Time", "imbalance_pct": "Imbalance %"})
        fig_trend = px.line(df_plot, x="Time", y="Imbalance %", title="Intraday Imbalance % (live)", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

# Signal History Table
st.subheader("📊 Buy Signal History (Today)")
signal_df = mem.signal_history.to_dataframe()
if signal_df.empty:
    st.info("No buy signals recorded today.")
else:
    # Define columns to display in the UI
    display_cols = {
        'id': 'ID', 'timestamp_ist': 'Time (IST)', 'suggestion': 'Signal', 
        'spot_price': 'Spot', 'vwap': 'VWAP', 'final_score': 'Score %', 
        'rsi': 'RSI', 'adx': 'ADX', 'telegram_sent': 'Sent',
        'option_price': 'Option Price', 'lot_value': 'Lot Value'
    }
    display_df = signal_df[list(display_cols.keys())].copy()
    display_df.rename(columns=display_cols, inplace=True)
    st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)

st.divider()
st.caption(f"App Version: {APP_VERSION} | Logic: Multi-Factor Score (OI, Vol, IV) + RSI/ADX Confirmation")
