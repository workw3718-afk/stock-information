# app.py
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# =============================
# Helpers
# =============================
def get_api_key() -> str:
    # Safe secrets access (won't crash if secrets.toml missing)
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


def fmt_big_num(x: Optional[float]) -> str:
    if x is None:
        return "—"
    n = float(x)
    absn = abs(n)
    if absn >= 1e12:
        return f"${n/1e12:.2f}T"
    if absn >= 1e9:
        return f"${n/1e9:.2f}B"
    if absn >= 1e6:
        return f"${n/1e6:.2f}M"
    if absn >= 1e3:
        return f"${n/1e3:.2f}K"
    return f"${n:.0f}"


def clean_no_long_dashes(s: str) -> str:
    return (s or "").replace("—", "-")


def truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def normalize_whitespace(s: str) -> str:
    # keep paragraph breaks, but normalize internal spacing
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    # collapse >2 blank lines to exactly 2
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def enforce_title_body_3paras(text: str) -> str:
    """
    Ensure output looks like:
    TITLE: ...
    (blank line)
    BODY: paragraph1
    (blank line)
    paragraph2
    (blank line)
    paragraph3
    """
    t = normalize_whitespace(text)

    # If model outputs Заголовок/Текст etc, normalize labels
    t = re.sub(r"^(ЗАГОЛОВОК|Заголовок)\s*:\s*", "TITLE: ", t, flags=re.MULTILINE)
    t = re.sub(r"^(ТЕКСТ|Текст)\s*:\s*", "BODY: ", t, flags=re.MULTILINE)

    # If missing TITLE/BODY, wrap it crudely
    if "TITLE:" not in t:
        # try first line as title
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if lines:
            title = lines[0]
            rest = "\n".join(lines[1:]).strip()
            if not rest:
                rest = "BODY: " + title
            t = f"TITLE: {title}\n\nBODY: {rest}"
        else:
            return "TITLE: \n\nBODY: \n\n\n"

    # Ensure BODY exists
    if "BODY:" not in t:
        # put everything after TITLE as body
        parts = t.split("TITLE:", 1)[1].strip()
        # first line as title
        lines = parts.split("\n")
        title = lines[0].strip()
        rest = "\n".join(lines[1:]).strip()
        t = f"TITLE: {title}\n\nBODY: {rest}"

    # Extract title and body
    m = re.search(r"TITLE:\s*(.+?)\n+BODY:\s*(.+)$", t, flags=re.DOTALL)
    if not m:
        return t

    title = m.group(1).strip()
    body = m.group(2).strip()

    # Split body into paragraphs by blank lines
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    # If model stuffed everything into one paragraph, try sentence split into 3 chunks
    if len(paras) < 3:
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\n+", " ", body).strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 6:
            # 3 roughly even chunks
            k = len(sentences)
            p1 = " ".join(sentences[: k // 3]).strip()
            p2 = " ".join(sentences[k // 3 : 2 * k // 3]).strip()
            p3 = " ".join(sentences[2 * k // 3 :]).strip()
            paras = [p1, p2, p3]
        else:
            # just pad
            while len(paras) < 3:
                paras.append("")
    elif len(paras) > 3:
        # merge extras into 3rd
        paras = [paras[0], paras[1], " ".join(paras[2:]).strip()]

    out = f"TITLE: {title}\n\nBODY: {paras[0]}\n\n{paras[1]}\n\n{paras[2]}".strip()
    return out


def pct_badge_html(chg_pct: Optional[float]) -> str:
    if chg_pct is None:
        return '<span style="opacity:0.7;">—</span>'
    color = "#22c55e" if chg_pct >= 0 else "#ef4444"
    bg = "rgba(34,197,94,0.15)" if chg_pct >= 0 else "rgba(239,68,68,0.15)"
    txt = fmt_pct(chg_pct)
    return f"""
    <span style="
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        background:{bg};
        color:{color};
        font-weight:700;
        font-size:14px;
        border:1px solid rgba(255,255,255,0.08);
    ">{txt}</span>
    """


# =============================
# News (StockTitan RSS)
# =============================
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


# =============================
# Market data (LIVE % + Market Cap)
# =============================
def fetch_market_data(ticker: str) -> Dict[str, Any]:
    """
    LIVE-ish:
    - price: last 1m candle close with prepost=True
    - prev_close: previous close from daily history
    - chg_pct: (price / prev_close - 1) * 100
    - market_cap_live: price * shares_outstanding (preferred) else info.marketCap
    """
    t = ticker.strip().upper()
    tk = yf.Ticker(t)

    company = ""
    price: Optional[float] = None
    chg_pct: Optional[float] = None

    vol: Optional[float] = None
    vol10: Optional[float] = None
    vol3m: Optional[float] = None

    shares_out: Optional[float] = None
    market_cap_info: Optional[float] = None
    market_cap_live: Optional[float] = None

    # Company + shares + marketcap (info)
    try:
        info = tk.get_info() or {}
        company = str(info.get("longName") or info.get("shortName") or "")[:120]
        shares_out = safe_float(info.get("sharesOutstanding"))
        market_cap_info = safe_float(info.get("marketCap"))
    except Exception:
        company = ""

    # Prev close (daily)
    prev_close: Optional[float] = None
    try:
        d = tk.history(period="7d", interval="1d")
        if d is not None and not d.empty and len(d) >= 2:
            prev_close = float(d["Close"].iloc[-2])
    except Exception:
        prev_close = None

    # LIVE price (1m)
    try:
        intraday = tk.history(period="1d", interval="1m", prepost=True)
        if intraday is not None and not intraday.empty:
            price = float(intraday["Close"].iloc[-1])
    except Exception:
        price = None

    # Fallback price (fast_info)
    if price is None:
        try:
            fi = getattr(tk, "fast_info", {}) or {}
            price = safe_float(fi.get("last_price"))
            if shares_out is None:
                shares_out = safe_float(fi.get("shares_outstanding"))
            if market_cap_info is None:
                market_cap_info = safe_float(fi.get("market_cap"))
        except Exception:
            price = None

    # Fallback price (daily last close)
    if price is None:
        try:
            d2 = tk.history(period="5d", interval="1d")
            if d2 is not None and not d2.empty:
                price = float(d2["Close"].iloc[-1])
        except Exception:
            price = None

    # LIVE % change
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

    # LIVE Market Cap
    if price is not None and shares_out not in (None, 0):
        market_cap_live = float(price) * float(shares_out)
    else:
        market_cap_live = market_cap_info  # fallback

    return {
        "ticker": t,
        "company": company,
        "price": price,
        "chg_pct": chg_pct,
        "vol": vol,
        "vol10": vol10,
        "vol3m": vol3m,
        "shares_out": shares_out,
        "mcap_live": market_cap_live,
        "mcap_info": market_cap_info,
    }


# =============================
# GPT Styles (POSITIVE ONLY)
# =============================
@dataclass
class StylePreset:
    name: str
    description: str
    system_en: str
    system_ru: str


FORMAT_RULES = """
OUTPUT FORMAT (MUST MATCH EXACTLY):

TITLE: <one line title>

BODY: <paragraph 1>

<blank line>
<paragraph 2>

<blank line>
<paragraph 3>

BODY must be exactly 3 paragraphs separated by one blank line.
""".strip()

CONTENT_RULES = """
Content rules:
- Positive framing only (no warnings, no caution, no downside framing).
- About 15–25% of the text should include meaningful numbers when available (revenue growth, capacity, timelines, contract duration, TAM, scale metrics).
- Do NOT always mention daily price/volume. Use market metrics only if they add value and keep it to one short sentence max.
- No buy/sell, no price targets, no "financial advice".
- No emojis. No long dashes.
- Sound human and specific (avoid generic filler).
""".strip()

STYLES: List[StylePreset] = [
    StylePreset(
        name="Observant Analyst (Quora DD)",
        description="Analytical, news-driven, execution-focused (positive only).",
        system_en=f"""
You write Quora-style analytical posts in English. Focus on execution, credibility, scale, and strategic positioning.
{FORMAT_RULES}
{CONTENT_RULES}
""".strip(),
        system_ru=f"""
Ты пишешь аналитические посты для Quora на русском. Фокус: исполнение, доверие, масштаб, стратегическое позиционирование.
{FORMAT_RULES}
{CONTENT_RULES}
""".strip(),
    ),
    StylePreset(
        name="Growth / Catalysts Lens",
        description="Growth angle: runway, expansion, measurable scaling (positive only).",
        system_en=f"""
You write Quora posts in English with a growth + catalysts lens: runway, scaling, distribution, execution milestones.
{FORMAT_RULES}
{CONTENT_RULES}
""".strip(),
        system_ru=f"""
Ты пишешь посты для Quora на русском с фокусом на рост и катализаторы: масштабирование, расширение, этапы исполнения.
{FORMAT_RULES}
{CONTENT_RULES}
""".strip(),
    ),
    StylePreset(
        name="Institutional Tone (clean)",
        description="Clean institutional tone, but still readable (positive only).",
        system_en=f"""
You write Quora posts in English in a clean institutional tone. Concrete, measured, constructive.
{FORMAT_RULES}
{CONTENT_RULES}
""".strip(),
        system_ru=f"""
Ты пишешь посты для Quora на русском в строгом институциональном тоне. Конкретно, без воды, позитивно.
{FORMAT_RULES}
{CONTENT_RULES}
""".strip(),
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
    """
    Strict translation only. No rewriting, no additions.
    target_lang: "Russian" or "English"
    """
    client = OpenAI(api_key=api_key)
    sys = (
        "You are a translation engine. Translate ONLY. "
        "Do not rewrite, summarize, add, remove, or change meaning. "
        "Preserve structure, headings like TITLE/BODY, numbers, tickers, and punctuation. "
        "Preserve paragraph breaks exactly."
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

    # Provide market data as OPTIONAL context (model may or may not use it)
    if data.get("price") is not None:
        lines.append(f"Live Price (optional): ${data.get('price'):.4f}")
    if data.get("chg_pct") is not None:
        lines.append(f"Live Change % (optional): {data.get('chg_pct'):.2f}%")
    if data.get("mcap_live") is not None:
        lines.append(f"Live Market Cap (optional): {data.get('mcap_live'):.0f}")

    if (
        data.get("vol") is not None
        and data.get("vol10") is not None
        and data.get("vol3m") is not None
    ):
        lines.append(
            f"Volume (optional): {int(data['vol'])} | 10d avg: {int(data['vol10'])} | 3mo avg: {int(data['vol3m'])}"
        )

    if key_catalyst.strip():
        lines.append(f"Key Catalyst (user): {key_catalyst.strip()}")

    if sel:
        lines.append("Selected headlines (use as primary context):")
        for it in sel:
            lines.append(f"- {it['title']}")
            if it.get("summary"):
                lines.append(f"  Summary: {truncate(strip_html(it['summary']), 260)}")
            if it.get("link"):
                lines.append(f"  Link: {it['link']}")

    if extra_context.strip():
        lines.append(f"Additional context: {extra_context.strip()}")

    # Hard requirement: EXACT formatting
    lines.append(
        "Hard requirement: Output exactly in the TITLE/BODY format with exactly 3 BODY paragraphs separated by one blank line."
    )
    return "\n".join(lines)


# =============================
# UI
# =============================
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

# Metrics row (add LIVE Market Cap)
m1, m2, m3, m4, m5 = st.columns(5)

# Current Price
m1.metric(
    "Current Price (live)",
    f"${fmt_money(market.get('price'))}" if market.get("price") is not None else "—",
)

# Price Change (%) + badge
chg = market.get("chg_pct")
m2.metric("Price Change (%) (live)", fmt_pct(chg) if chg is not None else "—")
m2.markdown(pct_badge_html(chg), unsafe_allow_html=True)

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
m3.metric("Volume (vs avg)", vol_text)

# Live Market Cap
mcap_live = market.get("mcap_live")
m4.metric("Market Cap (live)", fmt_big_num(mcap_live) if mcap_live is not None else "—")

# Company
m5.text_input("Company", value=market.get("company", "") or "", disabled=False)

key_catalyst = st.text_input(
    "⚡ Key Catalyst / News",
    value="",
    placeholder="e.g., advisory appointment, contract, results...",
)

with st.expander("Advanced options", expanded=False):
    extra_context = st.text_area("Additional Context", height=120)

    base_language = st.selectbox("Base language (generate original)", ["EN", "RU"], index=0)
    out_lang = st.selectbox("Output languages", ["EN + RU (translation)", "EN only", "RU only"], index=0)

    length_hint = st.slider("Length hint (characters)", 300, 2500, (900, 1400), step=10)
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
        pipeline_extra = (
            "\nStrictly comply with the required TITLE/BODY format and EXACTLY 3 BODY paragraphs.\n"
        )

    outputs: List[Dict[str, Any]] = []

    for v in range(int(n_variations)):
        var_note = (
            f"\nVariation directive: change phrasing and sentence rhythm. Avoid repeating phrases. Seed hint: {v}\n"
        )

        # Decide what to generate (original)
        if out_lang == "EN only":
            base_language_run = "EN"
        elif out_lang == "RU only":
            base_language_run = "RU"
        else:
            base_language_run = base_language  # user-chosen base, then translate

        if base_language_run == "EN":
            sys_base = style.system_en + pipeline_extra
            user_base = (
                user_prompt
                + f"\nLanguage: English. Character target: {min_len}-{max_len}.\n"
                + var_note
            )
            base_text = clean_no_long_dashes(call_gpt(api_key, model, sys_base, user_base))
            if ads_safe:
                base_text = re.sub(r"\b(BUY|SELL|PRICE TARGET|PT)\b", "", base_text, flags=re.IGNORECASE)
            base_text = enforce_title_body_3paras(base_text)

            en_text = base_text
            ru_text = ""

            if out_lang == "EN + RU (translation)":
                ru_text = clean_no_long_dashes(translate_text(api_key, model, en_text, "Russian"))
                ru_text = enforce_title_body_3paras(ru_text)

        else:
            sys_base = style.system_ru + pipeline_extra
            user_base = (
                user_prompt
                + f"\nLanguage: Russian. Character target: {min_len}-{max_len}.\n"
                + var_note
            )
            base_text = clean_no_long_dashes(call_gpt(api_key, model, sys_base, user_base))
            if ads_safe:
                base_text = re.sub(r"\b(КУПИТЬ|ПРОДАТЬ|ТАРГЕТ|ЦЕЛЬ ПО ЦЕНЕ)\b", "", base_text, flags=re.IGNORECASE)
            base_text = enforce_title_body_3paras(base_text)

            ru_text = base_text
            en_text = ""

            if out_lang == "EN + RU (translation)":
                en_text = clean_no_long_dashes(translate_text(api_key, model, ru_text, "English"))
                en_text = enforce_title_body_3paras(en_text)

        # Store outputs
        if out_lang == "EN only":
            outputs.append(
                {"variation": v + 1, "lang": "EN", "ticker": ticker.strip().upper(), "style": style_name, "text": en_text}
            )
        elif out_lang == "RU only":
            outputs.append(
                {"variation": v + 1, "lang": "RU", "ticker": ticker.strip().upper(), "style": style_name, "text": ru_text}
            )
        else:
            # paired rows for csv
            outputs.append(
                {"variation": v + 1, "lang": "EN", "ticker": ticker.strip().upper(), "style": style_name, "text": en_text}
            )
            outputs.append(
                {"variation": v + 1, "lang": "RU", "ticker": ticker.strip().upper(), "style": style_name, "text": ru_text}
            )

    st.session_state["generated_rows"] = outputs
    st.success(f"Generated {len(outputs)} rows.")

rows = st.session_state.get("generated_rows", [])
if rows:
    df = pd.DataFrame(rows)
    variations = sorted(df["variation"].unique().tolist())

    for v in variations:
        sub = df[df["variation"] == v]
        st.markdown(f"### Variation {v}")

        if out_lang == "EN + RU (translation)":
            en = sub[sub["lang"] == "EN"]["text"].values[0]
            ru = sub[sub["lang"] == "RU"]["text"].values[0]
            colL, colR = st.columns(2)
            with colL:
                st.markdown("**English**")
                st.text_area(f"EN v{v}", value=en, height=520)
            with colR:
                st.markdown("**Русский (перевод)**")
                st.text_area(f"RU v{v}", value=ru, height=520)
        elif out_lang == "EN only":
            en = sub[sub["lang"] == "EN"]["text"].values[0]
            st.text_area(f"EN v{v}", value=en, height=420)
        else:
            ru = sub[sub["lang"] == "RU"]["text"].values[0]
            st.text_area(f"RU v{v}", value=ru, height=420)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="generated_posts.csv",
        mime="text/csv",
    )
