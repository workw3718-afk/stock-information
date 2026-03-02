import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# -----------------------------
# Helpers
# -----------------------------
def get_api_key() -> str:
    try:
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "") or ""


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def fmt_money(x: Optional[float], d: int = 2) -> str:
    if x is None:
        return ""
    return f"{x:.{d}f}"


def fmt_pct(x: Optional[float], d: int = 2) -> str:
    if x is None:
        return ""
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{d}f}%"


def clean_no_long_dashes(s: str) -> str:
    return (s or "").replace("—", "-")


def truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


# -----------------------------
# News (StockTitan RSS)
# -----------------------------
def stocktitan_rss_url(ticker: str) -> str:
    t = ticker.strip().upper()
    return f"https://www.stocktitan.net/rss/news/{t}"


def fetch_news_rss(ticker: str, limit: int = 12) -> List[Dict[str, str]]:
    url = stocktitan_rss_url(ticker)
    feed = feedparser.parse(url)
    items: List[Dict[str, str]] = []
    for e in getattr(feed, "entries", [])[:limit]:
        items.append(
            {
                "title": (e.get("title") or "").strip(),
                "link": (e.get("link") or "").strip(),
                "summary": (e.get("summary") or "").strip(),
            }
        )
    return items


# -----------------------------
# Market data (LIVE %)
# -----------------------------
def fetch_market_data(ticker: str) -> Dict[str, Any]:
    t = ticker.strip().upper()
    tk = yf.Ticker(t)

    company = ""
    price: Optional[float] = None
    chg_pct: Optional[float] = None

    vol: Optional[float] = None
    vol10: Optional[float] = None
    vol3m: Optional[float] = None

    # Company name
    try:
        info = tk.get_info() or {}
        company = str(info.get("longName") or info.get("shortName") or "")[:120]
    except Exception:
        company = ""

    # Prev close from daily history
    prev_close: Optional[float] = None
    try:
        d = tk.history(period="7d", interval="1d")
        if d is not None and not d.empty and len(d) >= 2:
            prev_close = float(d["Close"].iloc[-2])
    except Exception:
        prev_close = None

    # LIVE price from 1m candles (PM/AH when available)
    try:
        intraday = tk.history(period="1d", interval="1m", prepost=True)
        if intraday is not None and not intraday.empty:
            price = float(intraday["Close"].iloc[-1])
    except Exception:
        price = None

    # Fallback price
    if price is None:
        try:
            fi = getattr(tk, "fast_info", {}) or {}
            price = safe_float(fi.get("last_price"))
        except Exception:
            price = None

    if price is None:
        try:
            d2 = tk.history(period="5d", interval="1d")
            if d2 is not None and not d2.empty:
                price = float(d2["Close"].iloc[-1])
        except Exception:
            price = None

    # LIVE % (price vs prev close)
    if price is not None and prev_close not in (None, 0):
        chg_pct = (price / prev_close - 1.0) * 100.0

    # Volume stats (daily)
    try:
        hist10 = tk.history(period="10d", interval="1d")
        if hist10 is not None and not hist10.empty:
            vol = float(hist10["Volume"].iloc[-1])
            vol10 = float(hist10["Volume"].tail(10).mean())

        hist3 = tk.history(period="3mo", interval="1d")
        if hist3 is not None and not hist3.empty:
            vol3m = float(hist3["Volume"].mean())
    except Exception:
        pass

    return {
        "ticker": t,
        "company": company,
        "price": price,
        "chg_pct": chg_pct,
        "vol": vol,
        "vol10": vol10,
        "vol3m": vol3m,
    }


# -----------------------------
# GPT styles
# -----------------------------
@dataclass
class StylePreset:
    name: str
    description: str
    system_en: str
    system_ru: str


COMMON_RULES = """
Rules:
- No direct calls to action (no buy/sell), no price targets, no "financial advice".
- No emojis. No long dashes.
- Sound human and specific, not generic.
""".strip()

STYLES: List[StylePreset] = [
    StylePreset(
        name="Observant Analyst (Quora DD)",
        description="Dense, analytical, news-driven DD tone.",
        system_en=f"Write Quora posts as a human analyst. Output EXACTLY in this structure:\nTITLE: ...\n\nBODY: ...\n\n2-4 paragraphs. Focus on what the news changes (execution, credibility, scope). Use market data in one short sentence max.\n{COMMON_RULES}",
        system_ru=f"Пиши пост для Quora как аналитик. Формат СТРОГО:\nTITLE: ...\n\nBODY: ...\n\n2-4 абзаца. Фокус: что меняет новость (исполнение, доверие, масштаб). Рыночные данные одной короткой фразой максимум.\n{COMMON_RULES}",
    ),
    StylePreset(
        name="Dividend / Stability Lens",
        description="Conservative resilience framing.",
        system_en=f"Write like a conservative long-term investor. Output:\nTITLE: ...\n\nBODY: ...\n\nFocus on resilience, duration, visibility, risk controls. Avoid hype.\n{COMMON_RULES}",
        system_ru=f"Пиши как консервативный долгосрок. Формат:\nTITLE: ...\n\nBODY: ...\n\nФокус: устойчивость, длительность, видимость, контроль рисков. Без хайпа.\n{COMMON_RULES}",
    ),
    StylePreset(
        name="Trader Read (clean, non-hype)",
        description="Trader perspective without trade instructions.",
        system_en=f"Write like a trader: catalyst + attention + what to watch next. Output:\nTITLE: ...\n\nBODY: ...\n\nNo trade instructions.\n{COMMON_RULES}",
        system_ru=f"Пиши как трейдер: катализатор, внимание, что смотреть дальше. Формат:\nTITLE: ...\n\nBODY: ...\n\nБез инструкций по сделке.\n{COMMON_RULES}",
    ),
    StylePreset(
        name="Bots / Algo Alert (timestamped)",
        description="Robotic alert with fields.",
        system_en=f"Output an alert with fields:\nTIME (UTC):\nTICKER:\nTITLE:\nSUMMARY:\nWATCH:\nShort and technical.\n{COMMON_RULES}",
        system_ru=f"Выдай алерт:\nTIME (UTC):\nTICKER:\nTITLE:\nSUMMARY:\nWATCH:\nКоротко и технично.\n{COMMON_RULES}",
    ),
]


def call_gpt(api_key: str, model: str, system: str, user: str) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text={"format": {"type": "text"}},
    )
    return (resp.output_text or "").strip()


def translate_text(api_key: str, model: str, text: str, target_lang: str) -> str:
    client = OpenAI(api_key=api_key)
    sys = (
        "You are a translation engine. Translate ONLY. "
        "Do not rewrite, summarize, add, remove, or change meaning. "
        "Preserve structure, headings like TITLE/BODY, numbers, tickers, and punctuation."
    )
    user = f"Translate to {target_lang}. Text:\n\n{text}"
    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        text={"format": {"type": "text"}},
    )
    return (resp.output_text or "").strip()


def build_user_prompt(
    data: Dict[str, Any],
    headlines: List[Dict[str, str]],
    selected_idx: List[int],
    key_catalyst: str,
    extra_context: str,
    ticker: str,
) -> str:
    sel = [headlines[i] for i in selected_idx if 0 <= i < len(headlines)]
    lines = [f"Ticker: {ticker}"]

    if data.get("company"):
        lines.append(f"Company: {data.get('company')}")

    if data.get("price") is not None:
        lines.append(f"Current Price: ${data.get('price'):.4f}")

    if data.get("chg_pct") is not None:
        lines.append(f"Price Change (%): {data.get('chg_pct'):.2f}%")

    if (
        data.get("vol") is not None
        and data.get("vol10") is not None
        and data.get("vol3m") is not None
    ):
        lines.append(
            f"Volume: {int(data['vol'])} | 10d avg: {int(data['vol10'])} | 3mo avg: {int(data['vol3m'])}"
        )

    if key_catalyst.strip():
        lines.append(f"Key Catalyst (user): {key_catalyst.strip()}")

    if sel:
        lines.append("Selected headlines (use these as context):")
        for it in sel:
            lines.append(f"- {it['title']}")
            if it.get("summary"):
                lines.append(f"  Summary: {truncate(strip_html(it['summary']), 260)}")
            if it.get("link"):
                lines.append(f"  Link: {it['link']}")

    if extra_context.strip():
        lines.append(f"Additional context: {extra_context.strip()}")

    return "\n".join(lines)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Post Studio", page_icon="⚡", layout="wide")
st.title("⚡ Quora Post Studio (News + Price + Styles)")

with st.sidebar:
    st.header("OpenAI")
    api_key_server = get_api_key()
    api_key_ui = st.text_input(
        "API Key (optional if server has it)",
        value="" if api_key_server else "",
        type="password",
    )
    model = st.text_input("Model", value="gpt-4.1-mini")
    st.caption("Local: paste key above OR set OPENAI_API_KEY in environment.")
    st.divider()
    mode = st.radio("Generation mode", ["Quick Mode", "Pipeline Mode"], index=0)

st.subheader("Stock Information")
cA, cB = st.columns([1.4, 1.0])
with cA:
    ticker = st.text_input("Ticker Symbol", value="NXXT")
with cB:
    if st.button("Fetch Data", type="primary"):
        st.session_state["market"] = fetch_market_data(ticker)
        st.session_state["news"] = fetch_news_rss(ticker, limit=12)

market = st.session_state.get("market", fetch_market_data(ticker))
news = st.session_state.get("news", [])

c1, c2, c3, c4 = st.columns(4)

# Current Price
c1.metric(
    "Current Price",
    f"${fmt_money(market.get('price'))}" if market.get("price") is not None else "—",
)

# Price Change (%) with green/red
chg = market.get("chg_pct")
if chg is not None:
    c2.metric(
        label="Price Change (%)",
        value=fmt_pct(chg),
        delta=fmt_pct(chg),
        delta_color="normal",  # green if +, red if -
    )
else:
    c2.metric("Price Change (%)", "—")

# Volume
vol_text = "—"
if (
    market.get("vol") is not None
    and market.get("vol10") is not None
    and market.get("vol3m") is not None
):
    vol_text = (
        f"{market['vol']/1e6:.2f}M | 10d: {(market['vol10']/1e6):.2f}M | 3mo: {(market['vol3m']/1e6):.2f}M"
    )
c3.metric("Volume (vs avg)", vol_text)

# Company
c4.text_input("Company", value=market.get("company", "") or "", disabled=False)

key_catalyst = st.text_input(
    "⚡ Key Catalyst / News",
    value="",
    placeholder="e.g., contract, advisory appointment, results...",
)

with st.expander("Advanced options", expanded=False):
    extra_context = st.text_area("Additional Context", height=120)
    base_language = st.selectbox("Base language (generate original)", ["EN", "RU"], index=0)
    out_lang = st.selectbox("Output languages", ["EN + RU", "EN only", "RU only"], index=0)
    length_hint = st.slider("Length hint (characters)", 250, 2500, (900, 1400), step=10)
    n_variations = st.number_input("Variations", min_value=1, max_value=10, value=2, step=1)

st.subheader("News Context")
left, right = st.columns([1.25, 1.0])

with left:
    if not news:
        st.info("Click Fetch Data to pull latest headlines automatically (StockTitan RSS).")
    else:
        if "selected_news" not in st.session_state:
            st.session_state["selected_news"] = list(range(min(3, len(news))))

        b1, b2, b3 = st.columns([0.25, 0.25, 0.5])
        with b1:
            if st.button("Auto"):
                st.session_state["selected_news"] = list(range(min(3, len(news))))
        with b2:
            if st.button("None"):
                st.session_state["selected_news"] = []
        with b3:
            if st.button("Refresh RSS"):
                st.session_state["news"] = fetch_news_rss(ticker, limit=12)
                news = st.session_state["news"]

        selected = set(st.session_state.get("selected_news", []))
        new_selected = []
        for i, it in enumerate(news[:12]):
            label = f"{i+1}. {it['title']}"
            if st.checkbox(label, value=(i in selected), key=f"news_{i}"):
                new_selected.append(i)
        st.session_state["selected_news"] = new_selected

with right:
    st.subheader("Style")
    style_name = st.selectbox("Choose style", [s.name for s in STYLES], index=0)
    style = next(s for s in STYLES if s.name == style_name)
    st.caption(style.description)
    ads_safe = st.toggle("Ads-safe wording", value=True)

st.divider()
st.subheader("Generate")

api_key = api_key_server or api_key_ui.strip()

if st.button("Generate posts", type="primary"):
    if not api_key:
        st.error("No API key. Paste it in sidebar OR set OPENAI_API_KEY as environment variable.")
        st.stop()

    sel_idx = st.session_state.get("selected_news", [])[:3]
    user_prompt = build_user_prompt(
        market,
        news,
        sel_idx,
        key_catalyst,
        extra_context,
        ticker.strip().upper(),
    )

    min_len, max_len = length_hint
    pipeline_extra = ""
    if mode == "Pipeline Mode":
        pipeline_extra = "\nStructure strictly: TITLE line, then BODY paragraphs. Each paragraph adds new info.\n"

    outputs = []

    for v in range(int(n_variations)):
        var_note = f"\nVariation: change phrasing and sentence rhythm. Avoid repeating phrases. Seed hint: {v}\n"

        if out_lang == "EN only":
            base_language_run = "EN"
        elif out_lang == "RU only":
            base_language_run = "RU"
        else:
            base_language_run = base_language

        if base_language_run == "EN":
            sys_base = style.system_en + pipeline_extra
            user_base = user_prompt + f"\nLanguage: English. Character target: {min_len}-{max_len}.\n" + var_note
            base_text = clean_no_long_dashes(call_gpt(api_key, model, sys_base, user_base))
            if ads_safe:
                base_text = re.sub(r"\b(BUY|SELL|PRICE TARGET|PT)\b", "", base_text, flags=re.IGNORECASE)

            en_text = base_text
            ru_text = ""
            if out_lang == "EN + RU":
                ru_text = clean_no_long_dashes(translate_text(api_key, model, en_text, "Russian"))

        else:
            sys_base = style.system_ru + pipeline_extra
            user_base = user_prompt + f"\nLanguage: Russian. Character target: {min_len}-{max_len}.\n" + var_note
            base_text = clean_no_long_dashes(call_gpt(api_key, model, sys_base, user_base))
            if ads_safe:
                base_text = re.sub(r"\b(КУПИТЬ|ПРОДАТЬ|ТАРГЕТ|ЦЕЛЬ ПО ЦЕНЕ)\b", "", base_text, flags=re.IGNORECASE)

            ru_text = base_text
            en_text = ""
            if out_lang == "EN + RU":
                en_text = clean_no_long_dashes(translate_text(api_key, model, ru_text, "English"))

        if out_lang == "EN only":
            outputs.append({"variation": v + 1, "lang": "EN", "ticker": ticker.strip().upper(), "style": style_name, "text": en_text})
        elif out_lang == "RU only":
            outputs.append({"variation": v + 1, "lang": "RU", "ticker": ticker.strip().upper(), "style": style_name, "text": ru_text})
        else:
            outputs.append({"variation": v + 1, "lang": "EN", "ticker": ticker.strip().upper(), "style": style_name, "text": en_text})
            outputs.append({"variation": v + 1, "lang": "RU", "ticker": ticker.strip().upper(), "style": style_name, "text": ru_text})

    st.session_state["generated_rows"] = outputs
    st.success(f"Generated {len(outputs)} rows.")

rows = st.session_state.get("generated_rows", [])
if rows:
    df = pd.DataFrame(rows)
    variations = sorted(df["variation"].unique().tolist())

    for v in variations:
        sub = df[df["variation"] == v]
        st.markdown(f"### Variation {v}")

        if out_lang == "EN + RU":
            en = sub[sub["lang"] == "EN"]["text"].values[0]
            ru = sub[sub["lang"] == "RU"]["text"].values[0]
            colL, colR = st.columns(2)
            with colL:
                st.markdown("**English (original or translated)**")
                st.text_area(f"EN v{v}", value=en, height=420)
            with colR:
                st.markdown("**Русский (перевод)**")
                st.text_area(f"RU v{v}", value=ru, height=420)
        elif out_lang == "EN only":
            en = sub[sub["lang"] == "EN"]["text"].values[0]
            st.text_area(f"EN v{v}", value=en, height=320)
        else:
            ru = sub[sub["lang"] == "RU"]["text"].values[0]
            st.text_area(f"RU v{v}", value=ru, height=320)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="generated_posts.csv",
        mime="text/csv",
    )

