# combined_nifty_atm0909_vwap.py
# NIFTY ΔOI Imbalance + TradingView VWAP alert
# - ATM: TV 09:09 → Yahoo daily open (robust, no-verify) → NSE underlying (provisional)
# - TV loop immediately upgrades ATM when 09:09 appears
# - Manual ATM override in sidebar
# - OC loop reloads ATM store every cycle
# - Weekday neighbors: Fri/Sat/Sun ±5, Mon ±4, Tue ±3, Wed ±2, Thu ±1
# - VWAP 15m session from TV 1m candles
# - Full logging + CSV/text outputs

import os, json, time, base64, datetime as dt, pathlib, threading, warnings, logging, sys, math, random
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
#import yfinance as yf
import certifi
import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= USER SETTINGS =================
SYMBOL               = "NIFTY"
FETCH_EVERY_SECONDS  = 60          # option-chain poll (1 min)
TV_FETCH_SECONDS     = 60           # TradingView poll (1 min)
AUTOREFRESH_MS       = 10_000

OUT_DIR              = pathlib.Path.home() / "Documents" / "NSE_output"
CSV_PATH             = OUT_DIR / "nifty_currweek_change_oi_atm_dynamic.csv"
ATM_STORE_PATH       = OUT_DIR / "nifty_atm_store.json"
LOG_PATH             = OUT_DIR / "nifty_app.log"
VWAP_NOW_TXT         = OUT_DIR / "nifty_vwap_now.txt"
VWAP_LOG_CSV         = OUT_DIR / "nifty_vwap_log.csv"

MAX_NEIGHBORS_LIMIT  = 20
IMBALANCE_TRIGGER    = 30.0         # %
VWAP_TOLERANCE_PTS   = 10.0          # alert when |spot - vwap| <= tolerance

# ---- HARD-CODED TradingView credentials (REPLACE THESE) ----
TV_USERNAME          = "dileep.marchetty@gmail.com"
TV_PASSWORD          = "1dE6Land@123"
# ============================================================

# ---- TELEGRAM SETTINGS ----
TELEGRAM_BOT_TOKEN   = "1849589360:AAFW_O3pxt6NZJvoV-NeUMfqu90wIyP8bSA"
TELEGRAM_CHAT_IDS    = ["1887957750", "5045651468"]  # Multiple chat IDs for alerts
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
    """Send alert message to multiple Telegram chat IDs. Returns True if at least one succeeds."""
    success_count = 0
    total_chats = len(TELEGRAM_CHAT_IDS)
    
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(TELEGRAM_API_URL, data=payload, timeout=10)
            if response.status_code == 200:
                log.info("Telegram alert sent successfully to chat_id: %s", chat_id)
                success_count += 1
            else:
                log.error("Telegram alert failed for chat_id %s: HTTP %s - %s", 
                         chat_id, response.status_code, response.text)
        except Exception as e:
            log.error("Telegram alert exception for chat_id %s: %s", chat_id, e)
    
    if success_count > 0:
        log.info("Telegram alert sent to %d/%d chat IDs successfully", success_count, total_chats)
        return True
    else:
        log.error("Telegram alert failed for all %d chat IDs", total_chats)
        return False

def format_vwap_alert_message(alert: str, spot: float, vwap: float, suggestion: str, 
                             atm_strike: int, expiry: str, imbalance_pct: float, 
                             timestamp: str) -> str:
    """Format a comprehensive VWAP alert message for Telegram."""
    
    # Determine alert emoji
    emoji = "🚨" if "BUY" in suggestion else "📊"
    direction_emoji = "📈" if "CALL" in suggestion else "📉" if "PUT" in suggestion else "⚪"
    
    message = f"""
{emoji} <b>NFS LIVE v1.0 - VWAP ALERT</b> {emoji}

{direction_emoji} <b>Signal:</b> {suggestion}
📍 <b>Alert:</b> {alert}

💰 <b>Market Data:</b>
• Spot Price: ₹{spot:,.2f}
• VWAP (15m): ₹{vwap:,.2f}
• Difference: ₹{abs(spot - vwap):,.2f}

⚖️ <b>Options Data:</b>
• ATM Strike: {atm_strike}
• Expiry: {expiry}
• OI Imbalance: {imbalance_pct:+.2f}%

🕐 <b>Time (IST):</b> {timestamp}
🇦🇪 <b>Time (UAE):</b> {dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST).astimezone(UAE).strftime("%Y-%m-%d %H:%M:%S")}

<i>Trade at your own risk. This is for educational purposes only.</i>
    """.strip()
    
    return message

def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)

def now_uae() -> dt.datetime:
    return dt.datetime.now(UAE)

def format_time_ist_uae(dt_ist: dt.datetime) -> str:
    """Format datetime to show both IST and UAE time for HTML contexts."""
    if dt_ist is None:
        return "—"
    dt_uae = dt_ist.astimezone(UAE)
    ist_str = dt_ist.strftime("%H:%M:%S")
    uae_str = dt_uae.strftime("%H:%M:%S")
    return f"{ist_str} IST<br><small>{uae_str} UAE</small>"

def format_datetime_ist_uae(dt_ist: dt.datetime) -> str:
    """Format full datetime to show both IST and UAE time for HTML contexts."""
    if dt_ist is None:
        return "—"
    dt_uae = dt_ist.astimezone(UAE)
    ist_str = dt_ist.strftime("%Y-%m-%d %H:%M:%S")
    uae_str = dt_uae.strftime("%Y-%m-%d %H:%M:%S")
    return f"{ist_str} IST<br><small>{uae_str} UAE</small>"

def format_time_compact(dt_ist: dt.datetime) -> str:
    """Format time for Streamlit metrics - IST only to avoid truncation."""
    if dt_ist is None:
        return "—"
    return dt_ist.strftime("%H:%M:%S IST")

def format_datetime_compact(dt_ist: dt.datetime) -> str:
    """Format datetime compactly for Streamlit display - single line format."""
    if dt_ist is None:
        return "—"
    dt_uae = dt_ist.astimezone(UAE)
    ist_str = dt_ist.strftime("%m-%d %H:%M:%S")  # Shorter date format
    uae_str = dt_uae.strftime("%H:%M")  # Remove seconds for UAE to save space
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
    today = now_ist().date()
    parsed = []
    for s in raw.get("records", {}).get("expiryDates", []):
        try:
            parsed.append((s, dt.datetime.strptime(s, "%d-%b-%Y").date()))
        except Exception:
            pass
    if not parsed:
        log.error("No expiryDates in JSON.")
        return None
    future = [p for p in parsed if p[1] >= today]
    chosen = min(future, key=lambda x: x[1]) if future else min(parsed, key=lambda x: x[1])
    return chosen[0]

def round_to_50(x: float) -> int:
    return int(round(x / 50.0) * 50)

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

        # if we’re struggling, rebuild the session once
        if i == 2:
            try:
                oc_session_cached.cache_clear()
                s = oc_session_cached()
                log.info("Recreated NSE session")
            except Exception as ee:
                log.warning("Failed to recreate NSE session: %s", ee)

    log.error("OC fetch failed after retries: %s", last_err)
    return None


# ---------------- TradingView helpers ----------------
def tv_login():
    #from tvDatafeed import TvDatafeed
    from tvDatafeed import TvDatafeed, Interval 
    try:
        tv = TvDatafeed(username="dileep.marchetty@gmail.com", password="1dE6Land@123")
        log.info("Logged in to TradingView as %s", TV_USERNAME)
        return tv
    except Exception as e:
        log.error("TradingView login failed: %s", e)
        raise

from functools import lru_cache

@lru_cache(maxsize=1)
def tv_login_cached():
    return tv_login()

@lru_cache(maxsize=1)
def oc_session_cached():
    return new_session()

def fetch_tv_1m_session(n_bars: int = 500):   # was 2000
    try:
        from tvDatafeed import Interval
    except Exception as e:
        log.error("tvDatafeed import failed: %s", e)
        return None

    last_err = None
    for i in range(1, 4):
        try:
            tv = tv_login_cached()  # reuse
            df = tv.get_hist(symbol="NIFTY", exchange="NSE",
                             interval=Interval.in_1_minute, n_bars=n_bars)
            if df is not None and not df.empty:
                # timezone align
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
                else:
                    df.index = df.index.tz_convert("Asia/Kolkata")
                # drop unused col to reduce memory
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

# ------------------------------------------------------------------
# TradingView – 15-minute history (kept separate from the 1-minute feed)
# ------------------------------------------------------------------
def fetch_tv_15m_session(n_bars: int = 64):   # was 500
    try:
        from tvDatafeed import Interval
    except Exception as e:
        log.error("tvDatafeed import failed (15m): %s", e)
        return None

    try:
        tv = tv_login_cached()
        df = tv.get_hist(symbol="NIFTY", exchange="NSE",
                         interval=Interval.in_15_minute, n_bars=n_bars)
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


# --- TradingView “VWAP (Length = N)” on any timeframe ----------------------
def compute_tv_vwap(
        df: pd.DataFrame,
        period_len: int = 14
) -> float | None:
    """
    Implements the Pine-script snippet you posted *verbatim*:

        typicalPrice                = (high + low + close)/3
        tpVol                       = typicalPrice * volume
        cumulativeTPVol (sum) (N)   = Σ(tpVol, N)
        cumulativeVol   (sum) (N)   = Σ(volume, N)
        vwapValue                   = cumulativeTPVol / cumulativeVol

    • Works on 15-minute candles **or any timeframe** (supply the DataFrame).
    • Uses the *last* `period_len` rows (same as Pine’s `sum(x, N)`).
    • If every bar’s volume is 0, falls back to the mean of typical price.
    """

    # 1️⃣ Sanity checks ------------------------------------------------------
    if (
        df is None
        or df.empty
        or not isinstance(df.index, pd.DatetimeIndex)
    ):
        log.error("compute_tv_vwap: invalid DataFrame")
        return None

    # 2️⃣ Ensure the index is IST (TradingView shows IST for NSE) -----------
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    # 3️⃣ Keep only the most-recent N bars ----------------------------------
    win = df.tail(period_len)
    if win.shape[0] == 0:
        return None
    if win.shape[0] < period_len:
        log.warning(
            "compute_tv_vwap: only %d of %d bars available – "
            "calculating with the bars we have",
            win.shape[0], period_len
        )

    # 4️⃣ Direct translation of the Pine script -----------------------------
    tp   = (win["high"] + win["low"] + win["close"]) / 3.0
    vol  = win["volume"].fillna(0).astype(float)
    tpV  = tp * vol

    sum_tpV = tpV.sum()
    sum_vol = vol.sum()

    if sum_vol == 0:                      # TV’s fallback when vol == 0
        return float(tp.mean())

    return float(sum_tpV / sum_vol)


def price_at_0909(df_1m: pd.DataFrame) -> float | None:
    """Close at 09:09 IST of latest session; fallback nearest 09:05–09:14; else 09:15 open."""
    if df_1m is None or df_1m.empty:
        return None
    latest_date = df_1m.index.max().date()
    t909 = dt.datetime.combine(latest_date, dt.time(9, 9), tzinfo=IST)
    try:
        if t909 in df_1m.index:
            return float(df_1m.loc[t909, "close"])
        win = df_1m.between_time("09:05", "09:14")
        if not win.empty and win.index.date.max() == latest_date:
            idx = min(win.index, key=lambda t: abs((t - t909).total_seconds()))
            return float(win.loc[idx, "close"])
        t915 = dt.datetime.combine(latest_date, dt.time(9, 15), tzinfo=IST)
        if t915 in df_1m.index:
            return float(df_1m.loc[t915, "open"])
    except Exception as e:
        log.error("price_at_0909 error: %s", e)
    return None

# --- Rolling VWAP over the most-recent *period_len* 1-minute bars -----------
def compute_period_vwap(df_1m: pd.DataFrame, period_len: int = 14) -> float | None:
    """
    TradingView “VWAP (Length = N)” re-implemented in pandas.

    Pine source you quoted:

        typicalPrice       = (high + low + close) / 3
        tpVol              = typicalPrice * volume
        sumTpVol_N         = sum(tpVol,     N)
        sumVol_N           = sum(volume,    N)
        vwapValue          = sumTpVol_N / sumVol_N

    We reproduce that literally:

        • the *last* N bars ( inclusive )
        • no session reset; just a rolling window
        • if every bar’s volume == 0 → return the mean(typicalPrice)
    """
    # 1) guard clauses -------------------------------------------------------
    if df_1m is None or df_1m.empty or not isinstance(df_1m.index, pd.DatetimeIndex):
        log.error("compute_period_vwap: invalid df_1m")
        return None

    # 2) force index → Asia/Kolkata -----------------------------------------
    if df_1m.index.tz is None:
        df_1m.index = df_1m.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df_1m.index = df_1m.index.tz_convert("Asia/Kolkata")

    # 3) keep only the last *period_len* rows -------------------------------
    win = df_1m.tail(period_len)

    if win.shape[0] == 0:
        return None
    if win.shape[0] < period_len:
        log.warning("compute_period_vwap: only %d of %d bars available", win.shape[0], period_len)

    # 4) exact Pine arithmetic ----------------------------------------------
    tp   = (win["high"] + win["low"] + win["close"]) / 3.0
    vol  = win["volume"].fillna(0).astype(float)
    tpV  = tp * vol

    sum_tpV = tpV.sum()
    sum_vol = vol.sum()

    # 5) all-zero volume fallback (same behaviour as Pine) -------------------
    if sum_vol == 0:
        return float(tp.mean())

    return float(sum_tpV / sum_vol)


# ---------------- Weekday neighbors mapping ----------------
def neighbors_by_weekday(d: dt.date) -> int:
    # Fri/Sat/Sun -> ±5, Mon -> ±4, Tue -> ±3, Wed -> ±2, Thu -> ±1
    wd = d.weekday()  # Mon=0 .. Sun=6
    mapping = {0: 4, 1: 3, 2: 2, 3: 1, 4: 5, 5: 5, 6: 5}
    return mapping.get(wd, 3)

def nearest_strike_block(strikes_sorted: list[int], atm: int, neighbors_each: int) -> list[int]:
    if not strikes_sorted:
        return []
    if atm not in strikes_sorted:
        atm = min(strikes_sorted, key=lambda x: abs(x - atm))
    idx = strikes_sorted.index(atm)
    lo = max(0, idx - neighbors_each)
    hi = min(len(strikes_sorted) - 1, idx + neighbors_each)
    return strikes_sorted[lo:hi+1]

# ---------------- Build OC DF with imbalance + ATM logic ----------------
def build_df_with_imbalance(raw: dict, store: dict):
    # always refresh from disk to pick up TV-loop upgrades / manual override
    store = load_atm_store()

    if not raw:
        return pd.DataFrame(), None

    expiry = pick_current_week_expiry(raw)
    if not expiry:
        return pd.DataFrame(), None

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
        atm_strike = None; base_val = None; atm_status = "capture-failed"
        for capt in (capture_today_atm_tv_0909, capture_today_atm_underlying):
            a,b,s = capt()
            if a is not None:
                atm_strike, base_val, atm_status = a,b,s
                break
        update_store_atm(atm_strike, base_val, atm_status)
    else:
        atm_strike = int(stored_atm)
        atm_status = stored_status
        base_val   = store.get("base_value", 0.0)
        """
        if atm_status != "captured-0909" and atm_status != "manual-override":
            y_a, y_b, y_s = capture_today_atm_yahoo_open()
            if y_a is not None and atm_strike != y_a:
                log.info("Correcting ATM via Yahoo: %s → %s", atm_strike, y_a)
                atm_strike, base_val, atm_status = y_a, y_b, y_s
                update_store_atm(atm_strike, base_val, atm_status)
        """
        log.info("Using ATM: %s (%s)", atm_strike, atm_status)

    # neighbors by weekday rule
    neighbors_each = neighbors_by_weekday(today_date)
    neighbors_each = min(neighbors_each, MAX_NEIGHBORS_LIMIT)
    wanted = set(nearest_strike_block(strikes_all, atm_strike, neighbors_each))
    log.info("Neighbors: weekday=%s ±%s, wanted_count=%s", today_date.weekday(), neighbors_each, len(wanted))

    for c in ("CE.changeinOpenInterest", "PE.changeinOpenInterest"):
        if c not in df_all.columns:
            df_all[c] = None

    df = df_all[["strikePrice", "CE.changeinOpenInterest", "PE.changeinOpenInterest"]].rename(
        columns={
            "strikePrice": "Strike",
            "CE.changeinOpenInterest": "Call Chg OI",
            "PE.changeinOpenInterest": "Put Chg OI",
        }
    )
    df = df[df["Strike"].isin(wanted)].copy()
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce").astype("Int64")
    df["Call Chg OI"] = pd.to_numeric(df["Call Chg OI"], errors="coerce")
    df["Put Chg OI"]  = pd.to_numeric(df["Put Chg OI"],  errors="coerce")
    df = df.sort_values("Strike").reset_index(drop=True)

    call_sum = float(df["Call Chg OI"].sum(skipna=True))
    put_sum  = float(df["Put Chg OI"].sum(skipna=True))
    denom = call_sum + put_sum
    if denom == 0:
        puts_pct = calls_pct = imbalance_pct = 0.0
    else:
        puts_pct = (put_sum / denom) * 100.0
        calls_pct = (call_sum / denom) * 100.0
        imbalance_pct = puts_pct - calls_pct

    suggestion = "NO SIGNAL"
    if abs(imbalance_pct) > IMBALANCE_TRIGGER:
        suggestion = "BUY PUT" if imbalance_pct < 0 else "BUY CALL"

    updated_str = now_ist().strftime("%Y-%m-%d %H:%M:%S")

    df.insert(0, "ATM", atm_strike)
    df.insert(0, "Expiry", expiry)
    df.insert(0, "Updated", updated_str)
    df["Put Σ Chg OI"]  = put_sum
    df["Call Σ Chg OI"] = call_sum
    df["PUTS %"]        = round(puts_pct, 2)
    df["CALLS %"]       = round(calls_pct, 2)
    df["Imbalance %"]   = round(imbalance_pct, 2)
    df["Suggestion"]    = suggestion

    # re-read store to show latest status/base (in case TV loop upgraded mid-build)
    latest_store = load_atm_store()
    atm_status_disp = latest_store.get("atm_status", "unknown")
    base_value_disp = latest_store.get("base_value", None)

    meta = {
        "neighbors_each": neighbors_each,
        "underlying": float(records.get("underlyingValue", 0.0)),
        "call_sum": call_sum,
        "put_sum": put_sum,
        "puts_pct": puts_pct,
        "calls_pct": calls_pct,
        "imbalance_pct": imbalance_pct,
        "suggestion": suggestion,
        "expiry": expiry,
        "atm": atm_strike,
        "updated": updated_str,
        "atm_status": atm_status_disp,
        "base_value": base_value_disp,
    }
    log.info("Imbalance: put_sum=%.0f call_sum=%.0f imb=%.2f%% sugg=%s; ATM=%s (%s)",
             put_sum, call_sum, imbalance_pct, suggestion, atm_strike, atm_status_disp)
    return df, meta

# ---------------- Signal History Tracking ----------------
class SignalHistory:
    """Track buy signal history for the day."""
    def __init__(self):
        self.lock = threading.Lock()
        self.signals: list[dict] = []
        self.history_file = OUT_DIR / "signal_history.json"
        self.load_history()
    
    def load_history(self):
        """Load existing signal history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    # Only keep today's signals
                    today = now_ist().date().isoformat()
                    self.signals = [s for s in data if s.get('date') == today]
        except Exception as e:
            log.error("Failed to load signal history: %s", e)
            self.signals = []
    
    def save_history(self):
        """Save signal history to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.signals, f, indent=2)
        except Exception as e:
            log.error("Failed to save signal history: %s", e)
    
    def add_signal(self, alert: str, spot: float, vwap: float, suggestion: str, 
                   atm_strike: int, expiry: str, imbalance_pct: float, 
                   telegram_sent: bool = False):
        """Add a new buy signal to history."""
        timestamp_ist = now_ist()
        timestamp_uae = timestamp_ist.astimezone(UAE)
        
        signal_data = {
            'id': len(self.signals) + 1,
            'date': timestamp_ist.date().isoformat(),
            'timestamp_ist': timestamp_ist.strftime("%Y-%m-%d %H:%M:%S"),
            'timestamp_uae': timestamp_uae.strftime("%Y-%m-%d %H:%M:%S"),
            'alert': alert,
            'suggestion': suggestion,
            'spot_price': round(float(spot), 2),
            'vwap': round(float(vwap), 2),
            'difference': round(abs(float(spot) - float(vwap)), 2),
            'atm_strike': int(atm_strike),
            'expiry': expiry,
            'imbalance_pct': round(float(imbalance_pct), 2),
            'telegram_sent': telegram_sent
        }
        
        with self.lock:
            self.signals.append(signal_data)
            self.save_history()
        
        log.info("Signal added to history: %s at %s IST", suggestion, signal_data['timestamp_ist'])
    
    def get_today_signals(self) -> list[dict]:
        """Get all signals for today."""
        with self.lock:
            return self.signals.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert signals to DataFrame for display."""
        with self.lock:
            if not self.signals:
                return pd.DataFrame()
            return pd.DataFrame(self.signals)

# ---------------- Memory store & loops ----------------
class StoreMem:
    def __init__(self):
        self.lock = threading.Lock()
        self.df_opt: pd.DataFrame | None = None
        self.meta_opt: dict = {}
        self.last_opt: dt.datetime | None = None

        self.vwap_latest: float | None = None
        self.vwap_df15: pd.DataFrame | None = None
        self.last_tv: dt.datetime | None = None

        self.vwap_alert: str = "NO ALERT"
        self.last_alert_key: str = ""
        self.last_telegram_alert: str = ""  # Track last sent telegram alert to avoid spam
        self.intraday = IntradayImbSeries()
        self.signal_history = SignalHistory()  # Track buy signal history
# ------------------------------------------------------------------------
# LIGHTWEIGHT IN-MEMORY SERIES TO TRACK INTRADAY IMBALANCE  (09:00-16:00)
# ------------------------------------------------------------------------
class IntradayImbSeries:
    def __init__(self, max_points: int = 480):
        self.lock = threading.Lock()
        self.points: list[tuple[dt.datetime, float]] = []
        self.max_points = max_points

    def add_point(self, ts: dt.datetime, imb: float):
        if not (dt.time(9, 0) <= ts.time() <= dt.time(16, 0)):
            return
        today = now_ist().date()
        with self.lock:
            self.points = [(t, v) for t, v in self.points if t.date() == today]
            self.points.append((ts, imb))
            # bound memory
            if len(self.points) > self.max_points:
                self.points = self.points[-self.max_points:]

    def to_dataframe(self) -> pd.DataFrame:
        with self.lock:
            if not self.points:
                return pd.DataFrame()
            df = pd.DataFrame(self.points, columns=["ts", "imbalance_pct"])
            df.set_index("ts", inplace=True)
            return df


def option_chain_loop(mem: StoreMem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            raw = fetch_raw_option_chain()
            df, meta = build_df_with_imbalance(raw, {})
            if not df.empty:
                # ── NEW: record today’s Imbalance % point ──────────────────
                imb_pct = meta.get("imbalance_pct")          # <- lives in meta
                if imb_pct is not None:
                    mem.intraday.add_point(now_ist(), float(imb_pct))

                # ── existing code (shared-memory, CSV, logging) ────────────
                with mem.lock:
                    mem.df_opt  = df
                    mem.meta_opt = dict(meta)
                    mem.last_opt = now_ist()
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


def write_vwap_files(stamp: str, vwap_latest: float | None, spot: float | None, suggestion: str):
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        v = f"{vwap_latest:.2f}" if vwap_latest is not None else "NA"
        s = f"{float(spot):.2f}" if spot is not None else "NA"
        #VWAP_NOW_TXT.write_text(f"{stamp} IST | VWAP15m={v} | Spot={s} | Signal={suggestion}\n")
        VWAP_NOW_TXT.write_text(f"{stamp} IST | VWAP15period={v} | Spot={s} | Signal={suggestion}\n")
        header_needed = not VWAP_LOG_CSV.exists()
        with VWAP_LOG_CSV.open("a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp_ist,vwap15m,spot,signal\n")
            f.write(f"{stamp},{v},{s},{suggestion}\n")
    except Exception as e:
        log.error("VWAP file write failed: %s", e)

# ---------------------------------------------------------------------------
# Trading-View worker thread
# • still pulls the 1-minute feed for 09 : 09 ATM logic
# • pulls an *independent* 15-minute feed that is used **only** for VWAP-with-period
# ---------------------------------------------------------------------------
def tradingview_loop(mem: StoreMem):
    """
    1. Fetch BOTH 1-minute and 15-minute NIFTY candles every TV_FETCH_SECONDS.
    2. Use the 1-minute feed for the 09:09 ATM upgrade just as before.
    3. Calculate VWAP-with-period from the 15-minute feed (period_len = 1),
       so it matches TradingView’s “VWAP (Length = 1)” applied to a 15-min chart.
    4. Persist the result, evaluate alerts, log, and sleep.
    """
    while True:
        try:
            # ── 1) Pull latest data ──────────────────────────────────────────
            df_1m  = fetch_tv_1m_session()          # unchanged – for ATM logic
            df_15m = fetch_tv_15m_session()         # NEW – 15-minute bars

            # ── 2) Instant ATM upgrade from 1-minute feed -------------------
            px909 = price_at_0909(df_1m) if df_1m is not None else None
            if px909:
                base_val  = float(px909)
                atm_guess = round_to_50(base_val)

                store = load_atm_store()
                needs_upgrade = (
                    store.get("date")       != today_str() or
                    store.get("atm_status") != "captured-0909" or
                    int(store.get("atm_strike", 0)) != atm_guess
                )
                if needs_upgrade:
                    update_store_atm(atm_guess, base_val, "captured-0909")
                    log.info("ATM upgraded to %s (base %.2f) by TV-loop", atm_guess, base_val)

                    # refresh imbalance immediately
                    raw_now = fetch_raw_option_chain()
                    df_now, meta_now = build_df_with_imbalance(raw_now, {})
                    if not df_now.empty:
                        with mem.lock:
                            mem.df_opt   = df_now.copy()
                            mem.meta_opt = dict(meta_now)
                            mem.last_opt = now_ist()
                        try:
                            df_now.to_csv(CSV_PATH, index=False)
                        except Exception as e:
                            log.error("CSV write failed (TV-trigger): %s", e)

            # ── 3) VWAP-with-period (from 15-minute bars) -------------------
            vwap_latest = None
            if df_15m is not None:
                # we only want the most-recent 15-minute candle,
                # so use period_len = 1 against 15-minute data
                vwap_latest = compute_tv_vwap(df_15m, period_len=14)

                # OPTIONAL DIAGNOSTIC: show the bar we just used
                log.debug("15-min bar for VWAP:\n%s", df_15m.tail(1).to_string())

            # ── 4) Store in shared memory -----------------------------------
            with mem.lock:
                mem.last_tv     = now_ist()
                mem.vwap_latest = vwap_latest      # what the UI reads
                # keep the name you were already using for anything else:
                mem.latest_vwap_period15 = vwap_latest

            # ── 5) Alert logic ----------------------------------------------
            with mem.lock:
                meta = mem.meta_opt or {}
                spot = meta.get("underlying")
                sugg = meta.get("suggestion", "NO SIGNAL")

            alert = "NO ALERT"
            if (
                vwap_latest is not None and
                spot is not None and
                sugg in ("BUY CALL", "BUY PUT") and
                abs(float(spot) - float(vwap_latest)) <= VWAP_TOLERANCE_PTS
            ):
                alert = f"{sugg} (spot near VWAP ±{VWAP_TOLERANCE_PTS})"

            with mem.lock:
                mem.vwap_alert = alert

            # ── 6) Telegram Alert Logic ------------------------------------
            # Send Telegram alert if we have a valid alert and it's different from the last one
            if alert != "NO ALERT" and alert != mem.last_telegram_alert:
                try:
                    # Get additional data for comprehensive alert
                    atm_strike = meta.get("atm", 0)
                    expiry = meta.get("expiry", "Unknown")
                    imbalance_pct = meta.get("imbalance_pct", 0.0)
                    timestamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Format and send Telegram message
                    telegram_message = format_vwap_alert_message(
                        alert=alert,
                        spot=float(spot),
                        vwap=float(vwap_latest),
                        suggestion=sugg,
                        atm_strike=int(atm_strike),
                        expiry=expiry,
                        imbalance_pct=float(imbalance_pct),
                        timestamp=timestamp
                    )
                    
                    telegram_success = send_telegram_alert(telegram_message)
                    if telegram_success:
                        with mem.lock:
                            mem.last_telegram_alert = alert
                        log.info("Telegram alert sent for: %s", alert)
                    else:
                        log.error("Failed to send Telegram alert for: %s", alert)
                    
                    # Add signal to history
                    mem.signal_history.add_signal(
                        alert=alert,
                        spot=float(spot),
                        vwap=float(vwap_latest),
                        suggestion=sugg,
                        atm_strike=int(atm_strike),
                        expiry=expiry,
                        imbalance_pct=float(imbalance_pct),
                        telegram_sent=telegram_success
                    )
                        
                except Exception as e:
                    log.error("Error preparing/sending Telegram alert: %s", e)

            # ── 7) Persist snapshot & write one-line log --------------------
            stamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
            write_vwap_files(stamp, vwap_latest, spot, sugg)

            log.info("[TV] vwap=%s alert=%s",
                     f"{vwap_latest:.2f}" if vwap_latest is not None else "None",
                     alert)

        except Exception as e:
            with mem.lock:
                mem.last_tv     = now_ist()
                mem.vwap_latest = None
            log.exception("TradingView loop error: %s", e)

        time.sleep(TV_FETCH_SECONDS)



@st.cache_resource
def start_background() -> StoreMem:
    mem = StoreMem()
    threading.Thread(target=option_chain_loop, args=(mem,), daemon=True, name="OC-Loop").start()
    threading.Thread(target=tradingview_loop, args=(mem,), daemon=True, name="TV-Loop").start()
    return mem

# ---------------- UI helpers ----------------
def play_beep_once_on_new_alert(mem: StoreMem, alert_text: str):
    key = f"{today_str()}|{alert_text}"
    if alert_text != "NO ALERT" and key != mem.last_alert_key:
        st.markdown(
            f"""
            <audio autoplay>
              <source src="data:audio/wav;base64,{BEEP_WAV_B64}" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True
        )
        mem.last_alert_key = key

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NIFTY ΔOI Imbalance + TV VWAP Alert", layout="wide")
st_autorefresh(interval=AUTOREFRESH_MS, key="nifty_autorefresh")

# Start background processes
mem = start_background()

# Sidebar thresholds
with st.sidebar:
    st.header("Settings")
    VWAP_tol = st.number_input("VWAP tolerance (pts)", value=float(VWAP_TOLERANCE_PTS), step=1.0)
    IMB_thr  = st.number_input("Imbalance trigger (%)", value=float(IMBALANCE_TRIGGER), step=1.0)
    st.caption(f"Logs: `{LOG_PATH}`")
    st.caption(f"Latest VWAP: `{VWAP_NOW_TXT}`")

    st.divider()
    st.subheader("Manual ATM override")
    man_atm = st.number_input("Set ATM strike (multiple of 50)", min_value=0, step=50, value=0)
    if st.button("Apply ATM override"):
        if man_atm > 0:
            update_store_atm(int(man_atm), float(man_atm), "manual-override")
            st.success(f"ATM overridden to {int(man_atm)}")
        else:
            st.warning("Enter a positive strike.")

    st.divider()
    st.subheader("Telegram Alert Test")
    st.caption("Test Telegram functionality during off-market hours")
    
    if st.button("Send Test Alert"):
        try:
            # Get current data or use mock data if not available
            with mem.lock:
                current_meta = dict(mem.meta_opt) if mem.meta_opt else {}
                current_vwap = mem.vwap_latest
            
            # Use real data if available, otherwise mock data for testing
            spot = current_meta.get("underlying", 24500.0)  # Mock spot price
            vwap = current_vwap if current_vwap else 24485.0  # Mock VWAP
            atm_strike = current_meta.get("atm", 24500)  # Mock ATM
            expiry = current_meta.get("expiry", "09-Jan-2025")  # Mock expiry
            imbalance_pct = current_meta.get("imbalance_pct", -35.5)  # Mock imbalance
            suggestion = current_meta.get("suggestion", "BUY PUT")  # Mock suggestion
            
            # Create test alert message
            test_alert = f"{suggestion} (TEST ALERT - spot near VWAP ±{VWAP_TOLERANCE_PTS})"
            timestamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
            
            telegram_message = format_vwap_alert_message(
                alert=test_alert,
                spot=float(spot),
                vwap=float(vwap),
                suggestion=suggestion,
                atm_strike=int(atm_strike),
                expiry=expiry,
                imbalance_pct=float(imbalance_pct),
                timestamp=timestamp
            )
            
            if send_telegram_alert(telegram_message):
                st.success("✅ Test Telegram alert sent successfully!")
                log.info("Manual test Telegram alert sent")
            else:
                st.error("❌ Failed to send test Telegram alert. Check logs.")
                
        except Exception as e:
            st.error(f"❌ Error sending test alert: {e}")
            log.error("Manual test Telegram alert failed: %s", e)
    
    if st.button("Send Simple Test Message"):
        try:
            simple_message = f"""
🧪 <b>TELEGRAM TEST MESSAGE</b> 🧪

This is a simple test to verify Telegram connectivity.

🕐 <b>Time (IST):</b> {now_ist().strftime("%Y-%m-%d %H:%M:%S")}
🇦🇪 <b>Time (UAE):</b> {now_uae().strftime("%Y-%m-%d %H:%M:%S")}

<i>If you receive this message, Telegram integration is working correctly.</i>
            """.strip()
            
            if send_telegram_alert(simple_message):
                st.success("✅ Simple test message sent successfully!")
                log.info("Simple test Telegram message sent")
            else:
                st.error("❌ Failed to send simple test message. Check logs.")
                
        except Exception as e:
            st.error(f"❌ Error sending simple test: {e}")
            log.error("Simple test Telegram message failed: %s", e)

    st.divider()
    if st.button("Show last 80 log lines"):
        try:
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-80:]
            st.code("\n".join(lines))
        except Exception as e:
            st.error(f"Could not read log: {e}")

with mem.lock:
    df_live = None if mem.df_opt is None else mem.df_opt.copy()
    meta = dict(mem.meta_opt)
    last_opt = mem.last_opt
    vwap_latest = mem.vwap_latest
    last_tv = mem.last_tv
    vwap_alert = mem.vwap_alert

st.title("NIFTY Change in OI — Imbalance + VWAP Alert (TradingView)")

# Current Time Display
current_time = now_ist()
current_uae = current_time.astimezone(UAE)

# Create 5 columns for better layout
col_time1, col_time2, col_time3, col_time4, col_time5 = st.columns([1, 1, 1, 1, 1])

with col_time1:
    st.metric("🕐 Current IST", current_time.strftime("%H:%M:%S"))
with col_time2:
    st.metric("🇦🇪 Current UAE", current_uae.strftime("%H:%M:%S"))
with col_time3:
    last_oc_uae = last_opt.astimezone(UAE).strftime("%H:%M:%S") if last_opt else "—"
    st.metric("📊 Last OC (UAE)", last_oc_uae, help="Last Option Chain pull in UAE time")
with col_time4:
    last_tv_uae = last_tv.astimezone(UAE).strftime("%H:%M:%S") if last_tv else "—"
    st.metric("📈 Last TV (UAE)", last_tv_uae, help="Last TradingView pull in UAE time")
with col_time5:
    st.metric("📅 Date", current_time.strftime("%m-%d %A"))

st.divider()

# Status row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last OC pull", 
          format_time_compact(last_opt) if last_opt else "—", 
          help="Option Chain data fetch time")
c2.metric("Last TV pull", 
          format_time_compact(last_tv) if last_tv else "—",
          help="TradingView data fetch time")
c3.metric("Spot (underlying)", f"{meta.get('underlying', float('nan')):,.2f}" if meta else "—")
c4.metric("VWAP (15-min period)", f"{vwap_latest:,.2f}" if vwap_latest else "—")
c5.metric("VWAP tolerance", f"±{VWAP_tol:.0f} pts")

if df_live is None or df_live.empty:
    st.warning("Waiting for first successful option-chain fetch…")
    st.stop()

expiry = meta.get("expiry", str(df_live["Expiry"].iloc[0]))
atm_strike = meta.get("atm", int(df_live["ATM"].iloc[0]))
atm_status = meta.get("atm_status", "unknown")
base_value = meta.get("base_value", None)
updated_str = meta.get("updated", str(df_live["Updated"].iloc[0]))
imbalance_pct = meta.get("imbalance_pct", float(df_live.get("Imbalance %", pd.Series([0])).iloc[0]))
suggestion = meta.get("suggestion", str(df_live.get("Suggestion", pd.Series(["NO SIGNAL"])).iloc[0]))
neighbors_each = meta.get("neighbors_each", 1)
call_sum = meta.get("call_sum", float(df_live["Call Σ Chg OI"].iloc[0]))
put_sum  = meta.get("put_sum",  float(df_live["Put Σ Chg OI"].iloc[0]))
puts_pct = meta.get("puts_pct", float(df_live["PUTS %"].iloc[0]))
calls_pct= meta.get("calls_pct",float(df_live["CALLS %"].iloc[0]))
spot     = meta.get("underlying", None)

# Apply sidebar thresholds for display
imbalance_ok = abs(imbalance_pct) > IMB_thr
vwap_ok = (vwap_latest is not None and spot is not None and abs(float(spot) - float(vwap_latest)) <= VWAP_tol)
combined_alert = "NO ALERT"
if suggestion in ("BUY CALL", "BUY PUT") and imbalance_ok and vwap_ok:
    combined_alert = f"{suggestion} (spot near VWAP ±{VWAP_tol})"

# Banner + sound
if combined_alert != "NO ALERT":
    st.success(f"VWAP ALERT: **{combined_alert}**", icon="✅")
    with mem.lock:
        play_beep_once_on_new_alert(mem, combined_alert)
else:
    st.info("No VWAP alert yet. Needs active BUY signal and |Spot−VWAP| within tolerance.", icon="ℹ️")

st.subheader(f"Expiry: {expiry}")
base_disp = f"{base_value:,.2f}" if isinstance(base_value, (int, float)) else "—"

# Convert updated_str to datetime for dual timezone display
try:
    updated_dt = dt.datetime.strptime(updated_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
    updated_display = format_datetime_compact(updated_dt)
except:
    updated_display = f"{updated_str} IST"

st.caption(
    f"Updated: **{updated_display}** • ATM: **{atm_strike}** (**{atm_status}**, base={base_disp}) • "
    f"Neighbors each side (weekday rule): **{neighbors_each}**"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("PUT Σ Chg OI", f"{put_sum:,.0f}")
k2.metric("CALL Σ Chg OI", f"{call_sum:,.0f}")
k3.metric("PUTS %", f"{puts_pct:,.2f}%")
k4.metric("CALLS %", f"{calls_pct:,.2f}%")
k5.metric("Imbalance (PUTS − CALLS)", f"{imbalance_pct:,.2f}%")

# VWAP/Spot caption
if vwap_latest is not None and spot is not None:
    #st.caption(f"VWAP15m: **{vwap_latest:,.2f}**  •  Spot: **{spot:,.2f}**  •  Diff: **{spot - vwap_latest:+.2f}**")
    st.caption(f"VWAP15-period: **{vwap_latest:,.2f}**  •  Spot: **{spot:,.2f}**  •  Diff: **{spot - vwap_latest:+.2f}**")
else:
    st.caption("VWAP or Spot not available yet. Check logs if this persists.")

st.dataframe(
    df_live[["Updated","Expiry","ATM","Strike","Call Chg OI","Put Chg OI",
             "Put Σ Chg OI","Call Σ Chg OI","PUTS %","CALLS %","Imbalance %","Suggestion"]]
      .sort_values("Strike"),
    use_container_width=True
)

# ── Intraday Imbalance trend (09:15–16:00) ──────────────────────────────
df_trend = mem.intraday.to_dataframe()

if df_trend.empty:
    st.info("Imbalance trend will appear after 09:15 IST once data accumulates.")
else:
    # Keep only session window
    df_trend = df_trend.between_time("09:15", "16:00")

    if df_trend.empty:
        st.info("No points yet for today after 09:15 IST.")
    else:
        # Decide which y column we actually have
        ycol = "imbalance_pct" if "imbalance_pct" in df_trend.columns else "imb"

        # Reset index so Plotly gets a proper time column; make time tz-naive
        df_plot = (
            df_trend
            .reset_index()                       # 'ts' becomes a column
            .rename(columns={"ts": "Time", ycol: "Imbalance %"})
        )
        # Ensure Time is naive datetime for Plotly
        if pd.api.types.is_datetime64_any_dtype(df_plot["Time"]):
            if getattr(df_plot["Time"].dt, "tz", None) is not None:
                df_plot["Time"] = df_plot["Time"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

        # Build figure
        fig_trend = px.line(
            df_plot,
            x="Time",
            y="Imbalance %",
            title="Intraday Imbalance % (live)",
            markers=True,
        )

        # Symmetric Y range in steps of 10
        y_min = float(df_plot["Imbalance %"].min())
        y_max = float(df_plot["Imbalance %"].max())
        yabs = max(abs(y_min), abs(y_max))
        ycap = max(10, math.ceil(yabs / 10.0) * 10)  # at least ±10

        fig_trend.update_layout(
            xaxis_title="Time (IST)",
            yaxis_title="Imbalance %",
            yaxis=dict(
                range=[-ycap, ycap],
                dtick=10,
                tick0=0,
                ticksuffix="%",
                zeroline=True,
                zerolinewidth=1,
            ),
            margin=dict(t=60, r=20, l=20, b=40),
            height=350,
        )

        st.plotly_chart(fig_trend, use_container_width=True)

# ── Signal History Table ──────────────────────────────────────────────────
st.divider()
st.subheader("📊 Buy Signal History (Today)")

# Ensure signal_history exists (for backward compatibility with cached objects)
if not hasattr(mem, 'signal_history'):
    mem.signal_history = SignalHistory()

signal_df = mem.signal_history.to_dataframe()
if signal_df.empty:
    st.info("No buy signals recorded today. Signals will appear here when VWAP conditions are met.")
else:
    # Display the signals in reverse chronological order (latest first)
    display_df = signal_df.iloc[::-1].copy()
    
    # Format the display columns
    display_columns = [
        'ID', 'Time (IST)', 'Time (UAE)', 'Signal', 'Spot Price', 'VWAP', 
        'Difference', 'ATM Strike', 'Expiry', 'OI Imbalance %', 'Telegram Sent'
    ]
    
    display_df_formatted = pd.DataFrame({
        'ID': display_df['id'],
        'Time (IST)': display_df['timestamp_ist'].str.split(' ').str[1],  # Show only time part
        'Time (UAE)': display_df['timestamp_uae'].str.split(' ').str[1],  # Show only time part
        'Signal': display_df['suggestion'],
        'Spot Price': display_df['spot_price'].apply(lambda x: f"₹{x:,.2f}"),
        'VWAP': display_df['vwap'].apply(lambda x: f"₹{x:,.2f}"),
        'Difference': display_df['difference'].apply(lambda x: f"₹{x:.2f}"),
        'ATM Strike': display_df['atm_strike'],
        'Expiry': display_df['expiry'],
        'OI Imbalance %': display_df['imbalance_pct'].apply(lambda x: f"{x:+.2f}%"),
        'Telegram Sent': display_df['telegram_sent'].apply(lambda x: "✅" if x else "❌")
    })
    
    st.dataframe(display_df_formatted, use_container_width=True, hide_index=True)
    
    # Summary stats
    total_signals = len(display_df)
    telegram_success = display_df['telegram_sent'].sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals Today", total_signals)
    col2.metric("Telegram Sent", f"{telegram_success}/{total_signals}")
    col3.metric("Success Rate", f"{(telegram_success/total_signals*100):.1f}%" if total_signals > 0 else "0%")
    col4.metric("Last Signal", display_df.iloc[0]['timestamp_ist'].split(' ')[1] if total_signals > 0 else "None")

# Footer
st.divider()
col_footer1, col_footer2, col_footer3 = st.columns([2, 1, 1])

with col_footer1:
    st.caption("🚀 **NFS LIVE v0.5** - NIFTY Options Chain Analysis with VWAP Alerts & Telegram Integration")
    st.caption("⚠️ **Disclaimer**: This tool is for educational purposes only. Trade at your own risk.")

with col_footer2:
    if st.button("📝 View Changelog", help="View complete version history and changes"):
        # Read and display changelog content
        try:
            changelog_content = pathlib.Path("changelog.md").read_text(encoding="utf-8")
            st.text_area("📝 Changelog", changelog_content, height=400, help="Complete version history")
        except Exception as e:
            st.error(f"Could not load changelog: {e}")

with col_footer3:
    st.caption("**Quick Links:**")
    st.caption("• [Logs]({}) 📋".format(LOG_PATH))
    st.caption("• [VWAP Data]({}) 📊".format(VWAP_NOW_TXT))
    st.caption("• [CSV Output]({}) 📁".format(CSV_PATH))
