# app.py
import os
import re
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import feedparser
from openai import OpenAI


# =========================================================
# OpenAI key / client
# =========================================================
def resolve_api_key(ui_key: str = "") -> str:
    if ui_key and ui_key.strip():
        return ui_key.strip()
    try:
        if "OPENAI_API_KEY" in st.secrets:
            v = str(st.secrets["OPENAI_API_KEY"]).strip()
            if v:
                return v
    except Exception:
        pass
    return (os.getenv("OPENAI_API_KEY", "") or "").strip()


def get_client(api_key: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Add in Streamlit Secrets or paste in sidebar.")
    return OpenAI(api_key=api_key)


# =========================================================
# Helpers
# =========================================================
def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def fmt_money(x: Optional[float], d: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{d}f}"


def fmt_pct(x: Optional[float], d: int = 2) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{d}f}%"


def fmt_big_num(x: Optional[float]) -> str:
    if x is None:
        return "—"
    n = float(x)
    a = abs(n)
    if a >= 1e12:
        return f"${n/1e12:.2f}T"
    if a >= 1e9:
        return f"${n/1e9:.2f}B"
    if a >= 1e6:
        return f"${n/1e6:.2f}M"
    if a >= 1e3:
        return f"${n/1e3:.2f}K"
    return f"${n:.0f}"


def normalize_ws(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def clean_no_long_dashes(s: str) -> str:
    return (s or "").replace("—", "-").replace("–", "-")


def truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


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


def count_number_anchors(text: str) -> int:
    if not text:
        return 0
    hits = re.findall(r"(?<!\w)(?:\$?\d[\d,]*(?:\.\d+)?%?)(?!\w)", text)
    return len(hits)


def to_dollar_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return "$"
    return t if t.startswith("$") else f"${t}"


# =========================================================
# News: URL extractor + RSS (optional)
# =========================================================
def stocktitan_rss_url(ticker: str) -> str:
    return f"https://www.stocktitan.net/rss/news/{ticker.strip().upper()}"


@st.cache_data(ttl=120)
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


@st.cache_data(ttl=600)
def fetch_article(url: str, timeout: int = 12) -> Dict[str, str]:
    url = (url or "").strip()
    if not url:
        return {"url": "", "title": "", "text": ""}

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    ogt = soup.find("meta", property="og:title")
    if ogt and ogt.get("content"):
        title = ogt.get("content").strip() or title

    ps = soup.find_all("p")
    paras = []
    for p in ps:
        txt = p.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) >= 40:
            paras.append(txt)

    text = "\n\n".join(paras[:16]).strip()
    return {"url": url, "title": truncate(title, 160), "text": truncate(text, 4200)}


# =========================================================
# Market data + Technicals
# =========================================================
@st.cache_data(ttl=45)
def fetch_market_data(ticker: str) -> Dict[str, Any]:
    t = ticker.strip().upper()
    tk = yf.Ticker(t)

    company = ""
    price: Optional[float] = None
    chg_pct: Optional[float] = None
    shares_out: Optional[float] = None
    market_cap_info: Optional[float] = None

    prev_close: Optional[float] = None
    vol: Optional[float] = None
    vol10: Optional[float] = None
    vol3m: Optional[float] = None

    low_52w: Optional[float] = None
    high_52w: Optional[float] = None

    try:
        info = tk.get_info() or {}
        company = str(info.get("longName") or info.get("shortName") or "")[:120]
        shares_out = safe_float(info.get("sharesOutstanding"))
        market_cap_info = safe_float(info.get("marketCap"))
        low_52w = safe_float(info.get("fiftyTwoWeekLow"))
        high_52w = safe_float(info.get("fiftyTwoWeekHigh"))
    except Exception:
        pass

    try:
        d = tk.history(period="7d", interval="1d")
        if d is not None and not d.empty and len(d) >= 2:
            prev_close = float(d["Close"].iloc[-2])
    except Exception:
        prev_close = None

    try:
        intraday = tk.history(period="1d", interval="1m", prepost=True)
        if intraday is not None and not intraday.empty:
            price = float(intraday["Close"].iloc[-1])
    except Exception:
        price = None

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

    if price is None:
        try:
            d2 = tk.history(period="5d", interval="1d")
            if d2 is not None and not d2.empty:
                price = float(d2["Close"].iloc[-1])
        except Exception:
            price = None

    if price is not None and prev_close not in (None, 0):
        chg_pct = (price / prev_close - 1.0) * 100.0

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

    # fallback 52w from 1y history if missing
    if low_52w is None or high_52w is None:
        try:
            h1y = tk.history(period="1y", interval="1d")
            if h1y is not None and not h1y.empty:
                low_52w = float(h1y["Low"].min())
                high_52w = float(h1y["High"].max())
        except Exception:
            pass

    mcap_live: Optional[float] = None
    if price is not None and shares_out not in (None, 0):
        mcap_live = float(price) * float(shares_out)
    else:
        mcap_live = market_cap_info

    pct_off_low: Optional[float] = None
    if price is not None and low_52w not in (None, 0):
        pct_off_low = (price / low_52w - 1.0) * 100.0

    vol_mult10: Optional[float] = None
    if vol is not None and vol10 not in (None, 0):
        vol_mult10 = vol / vol10

    return {
        "ticker": t,
        "company": company,
        "price": price,
        "chg_pct": chg_pct,
        "shares_out": shares_out,
        "mcap_live": mcap_live,
        "mcap_info": market_cap_info,
        "vol": vol,
        "vol10": vol10,
        "vol3m": vol3m,
        "vol_mult10": vol_mult10,
        "low_52w": low_52w,
        "high_52w": high_52w,
        "pct_off_low": pct_off_low,
    }


def auto_levels(price: Optional[float]) -> Dict[str, Optional[float]]:
    if price is None or price <= 0:
        return {"support": None, "mid": None, "trigger": None, "exp1": None, "exp2": None}
    # heuristic ladder (keeps trader-usable)
    support = round(price * 0.95, 2)
    mid = round(price * 0.99, 2)
    trigger = round(price * 1.05, 2)
    exp1 = round(price * 1.12, 2)
    exp2 = round(price * 1.18, 2)
    return {"support": support, "mid": mid, "trigger": trigger, "exp1": exp1, "exp2": exp2}


# =========================================================
# Master Prompt Engine
# =========================================================
OUTPUT_FORMATS = [
    ("big_post", "1 big analytical post"),
    ("thread", "Thread (5–7 parts)"),
    ("five_posts", "5 medium posts"),
    ("memo", "Institutional memo"),
    ("tech", "Technical breakdown"),
]

PLATFORMS = [
    ("twitter", "Twitter / X"),
    ("quora", "Quora"),
    ("reddit", "Reddit"),
    ("discord", "Discord"),
]

MASTER_SYSTEM = """
You are an institutional analyst who writes like a technical trader with deep understanding of:
- microgrid architecture
- grid-forming inverters
- distributed generation
- BESS economics
- diesel displacement
- federal procurement
- defense energy systems
- BYOG (Bring Your Own Generation)
- capital cycle dynamics
- technical chart structure

Your mission: produce a strong bullish analysis for the provided ticker, but without hype and without direct investment recommendations.
Tone: professional, technical, structured, confident, no pink glasses, no accusations.
""".strip()

SECTOR_VALIDATION_NUMBERS = """
You MUST include concrete microgrid efficiency ranges (use these exact ranges, integrate naturally, do not list as a raw bullet dump):
- 20% to 60% diesel reduction (mining, remote industry)
- up to 96% fuel consumption reduction (oilfield hybrid case)
- 30% to 40% energy cost reduction
- 15% to 20% peak demand reduction
- ~40% reduction in generator runtime
""".strip()

MACRO_NARRATIVE = """
Macro / Energy Security Narrative MUST include:
- BYOG as structural trend ("bring your own power")
- data center load growth / power density shock
- defense electrification and hardened energy systems
- federal resilience funding / runway
- energy autonomy > grid dependence

Core framing:
Energy security = national security.
Distributed generation = strategic asset.
Unified control layer = margin capture point.
""".strip()

NXXT_SECTION = """
"NXXT section" MUST directly connect:
- exclusive 2-year federal agreement (as provided by user context if included)
- role as technology + execution partner
- microgrids, battery storage, distributed energy, AI-driven control
Explain clearly:
NXXT is not just a solar installer.
NXXT is a control + integration + resilience stack.
""".strip()

TECHNICAL_SECTION = """
Technical analysis MUST include numbers:
- current price
- % off 52-week low
- volume vs avg volume (e.g., vs 10d avg)
- structure language: higher lows / channel / base / compression
- key levels formatted as:
  Support: X
  Mid-structure: Y
  Trigger: Z
  Expansion: A–B
Short interest mechanics: allowed only as conditional logic ("if low float + volume expansion then squeeze mechanics can appear"), no hard claims.
Write like trader-to-traders: structure + triggers, no emotion.
""".strip()

INSTITUTIONAL_SIGNAL = """
Institutional signal MUST explain:
- why defense energy / hardened systems validate the sector
- why BYOG narrative confirms macro
- why federal resilience funding implies runway
- why capital flows are attracted to hardened energy systems
""".strip()

FINAL_NARRATIVE = """
Close with this logic (paraphrase, not verbatim):
Grid can't scale fast enough.
Energy shifts to distributed architecture.
Unified control becomes essential.
Defense proves it first.
Industry follows.
Show NXXT is placed at this part of the cycle.
""".strip()


def platform_rules(platform_key: str) -> str:
    if platform_key == "twitter":
        return """
Platform: Twitter/X
- Output ONLY the text.
- Each post <= 280 chars.
- Use trader slang lightly.
- Mention ticker exactly once per post in $TICKER format.
- No hashtags unless enabled.
""".strip()
    if platform_key == "quora":
        return """
Platform: Quora
- Output exactly:
TITLE: <one line>
BODY: <3-6 short paragraphs>
- No bullet lists. No extra headings.
- Title should be click-friendly but not hype.
""".strip()
    if platform_key == "reddit":
        return """
Platform: Reddit
- Output ONLY the post text.
- 6-12 sentences, 1-2 short paragraph breaks allowed.
- Conversational but analytical, no corporate press release tone.
""".strip()
    if platform_key == "discord":
        return """
Platform: Discord
- Output ONLY the message text.
- 4-10 lines, tight.
- No bullet lists.
""".strip()
    return ""


def output_format_rules(fmt_key: str) -> str:
    if fmt_key == "big_post":
        return """
Output format: 1 big analytical post.
- Single cohesive piece.
- Must contain the 6 sections logically (can be implicit via paragraphing).
""".strip()
    if fmt_key == "thread":
        return """
Output format: Thread of 5–7 parts.
- Label parts as 1/ , 2/ , 3/ ... (Twitter style) ONLY if platform is Twitter.
- If platform is not Twitter, just separate parts with blank lines and "Part X:".
- Each part must advance the logic, no repetition.
""".strip()
    if fmt_key == "five_posts":
        return """
Output format: 5 medium posts.
- Each post stands alone.
- Must collectively cover all 6 sections, distributed across the 5 posts.
""".strip()
    if fmt_key == "memo":
        return """
Output format: 1 institutional memo.
- Header: Executive Summary, Macro, Sector Validation, Company Positioning, Technical Setup, Risks (light, structural), Conclusion.
- No hype, no recommendations.
""".strip()
    if fmt_key == "tech":
        return """
Output format: 1 technical breakdown.
- Focus 60–80% on technicals + flow + triggers, but still include microgrid / macro link briefly.
- Include the level ladder exactly (Support / Mid-structure / Trigger / Expansion).
""".strip()
    return ""


def build_master_user_prompt(
    *,
    ticker: str,
    market: Dict[str, Any],
    levels: Dict[str, Optional[float]],
    platform_key: str,
    fmt_key: str,
    include_hashtag: bool,
    trader_slang: bool,
    user_context: str,
    news: Dict[str, str],
) -> str:
    t = (ticker or "").strip().upper()
    cash = to_dollar_ticker(t)

    price = market.get("price")
    pct_off_low = market.get("pct_off_low")
    low_52w = market.get("low_52w")
    high_52w = market.get("high_52w")
    vol = market.get("vol")
    vol10 = market.get("vol10")
    vol_mult10 = market.get("vol_mult10")
    mcap = market.get("mcap_live")

    s = levels.get("support")
    mid = levels.get("mid")
    trig = levels.get("trigger")
    exp1 = levels.get("exp1")
    exp2 = levels.get("exp2")

    news_block = ""
    if (news.get("title") or "").strip() or (news.get("text") or "").strip():
        news_block = f"""
NEWS CONTEXT (use as optional catalyst lens, do not drift into generic military talk):
- URL: {news.get("url","")}
- Title: {news.get("title","")}
- Extract: {truncate(news.get("text",""), 1200)}
""".strip()

    tech_metrics = f"""
TECH METRICS (use these exact numbers if present):
- Current price: {price if price is not None else "N/A"}
- 52w low: {low_52w if low_52w is not None else "N/A"}
- 52w high: {high_52w if high_52w is not None else "N/A"}
- % off 52w low: {pct_off_low if pct_off_low is not None else "N/A"}
- Volume (last day): {int(vol) if vol is not None else "N/A"}
- Volume vs 10d avg: {int(vol10) if vol10 is not None else "N/A"} (mult={vol_mult10:.2f}x if mult available)
- Market cap (live est): {mcap if mcap is not None else "N/A"}
""".strip()

    level_ladder = f"""
LEVEL LADDER (format exactly like this in output):
Support: {s if s is not None else "N/A"}
Mid-structure: {mid if mid is not None else "N/A"}
Trigger: {trig if trig is not None else "N/A"}
Expansion: {exp1 if exp1 is not None else "N/A"}–{exp2 if exp2 is not None else "N/A"}
""".strip()

    twitter_rules_extra = ""
    if platform_key == "twitter":
        twitter_rules_extra = f"""
Twitter enforcement:
- Each post must include the ticker exactly once as {cash}.
- Do NOT include the plain ticker without $.
- Keep each post <= 280 characters.
- Hashtags: {"allow max 1" if include_hashtag else "do not use"}.
- Trader slang: {"enabled" if trader_slang else "disabled"} (if disabled, keep it clean institutional).
""".strip()

    # fix mult formatting
    if vol_mult10 is None:
        tech_metrics = tech_metrics.replace("(mult={vol_mult10:.2f}x if mult available)", "")

    return "\n\n".join(
        x for x in [
            f"Ticker: {t} (Twitter format: {cash})",
            tech_metrics,
            level_ladder,
            news_block,
            ("USER CONTEXT:\n" + user_context.strip()) if user_context.strip() else "",
            twitter_rules_extra,
            "Hard requirement: Must include the sector validation numeric ranges exactly as described.",
        ]
        if x and x.strip()
    )


def call_gpt(api_key: str, model: str, system: str, user: str, temperature: float) -> str:
    client = get_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def translate_text(api_key: str, model: str, text: str, target_lang: str) -> str:
    client = get_client(api_key)
    sys = (
        "You are a translation engine. Translate ONLY. "
        "Do not rewrite or summarize. Preserve numbers, $TICKER formatting, and line breaks."
    )
    user = f"Translate to {target_lang}. Text:\n\n{text}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def postprocess(text: str, platform_key: str, ticker: str, include_hashtag: bool) -> str:
    s = clean_no_long_dashes(text).strip()

    # anti press-release filler (light)
    s = re.sub(r"\b(well[- ]positioned|strategic positioning|stands to benefit significantly)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[ \t]{2,}", " ", s)

    t = (ticker or "").strip().upper()
    cash = to_dollar_ticker(t)

    if platform_key == "twitter":
        # Ensure $TICKER appears exactly once per post in Twitter outputs.
        # For threads / 5 posts, we enforce per block split.
        blocks = split_twitter_blocks(s)

        fixed = []
        for b in blocks:
            bb = b.strip()
            if not bb:
                continue
            # remove duplicates
            bb = re.sub(re.escape(cash), "", bb, flags=re.IGNORECASE).strip()
            bb = re.sub(rf"\b{re.escape(t)}\b", "", bb, flags=re.IGNORECASE).strip()

            # hashtag policy
            if not include_hashtag:
                bb = re.sub(r"#\w+", "", bb).strip()

            bb = (bb + f" {cash}").strip()

            # hard trim
            if len(bb) > 280:
                bb = bb[:277] + "..."
            fixed.append(bb)

        return "\n\n".join(fixed).strip()

    # Non-twitter: no special enforcement
    if not include_hashtag:
        s = re.sub(r"#\w+", "", s).strip()
    return s


def split_twitter_blocks(s: str) -> List[str]:
    """
    Try to split into blocks for Twitter:
    - if model used "1/" lines or "Part"
    - else split on double newlines
    """
    s = s.strip()
    if not s:
        return []
    # If has "1/" style
    if re.search(r"(?m)^\s*\d\s*/", s):
        # split by occurrences of \n\n or line starts with digit/
        parts = re.split(r"\n{2,}", s)
        return [p.strip() for p in parts if p.strip()]
    # fallback
    parts = re.split(r"\n{2,}", s)
    return [p.strip() for p in parts if p.strip()]


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="NXXT Master Prompt Studio", page_icon="⚡", layout="wide")
st.title("⚡ NXXT Master Prompt Studio (Macro + Defense + BYOG + Technical)")

CARD_CSS = """
<style>
:root{
  --card-bg: rgba(255,255,255,0.05);
  --card-border: rgba(255,255,255,0.12);
  --card-border-active: rgba(34,197,94,0.70);
  --text-dim: rgba(255,255,255,0.70);
}
.card {
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  border-radius: 14px;
  padding: 14px 14px 12px 14px;
}
.smallhint{ color: rgba(255,255,255,0.65); font-size:12px; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("OpenAI")
    api_key_ui = st.text_input("API Key (optional if Secrets has it)", value="", type="password")
    api_key = resolve_api_key(api_key_ui)
    model = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Creativity", 0.40, 1.10, 0.85, 0.05)

    st.divider()
    st.header("Output")
    platform_key = st.selectbox("Platform", [k for k, _ in PLATFORMS], format_func=lambda x: dict(PLATFORMS)[x], index=0)
    fmt_key = st.selectbox("Format", [k for k, _ in OUTPUT_FORMATS], format_func=lambda x: dict(OUTPUT_FORMATS)[x], index=0)

    out_lang = st.selectbox("Language", ["EN only", "RU only", "EN + RU (translation)"], index=0)

    st.divider()
    st.header("Twitter options")
    twitter_slang = st.toggle("Trader slang (Twitter)", value=True)
    allow_hashtag = st.toggle("Allow 1 hashtag (Twitter)", value=False)

    st.divider()
    st.header("Variations")
    n_variations = st.number_input("Variations", min_value=1, max_value=10, value=2, step=1)

# Ticker + market
st.subheader("Ticker & Market Snapshot")
c1, c2 = st.columns([1.1, 1.2])
with c1:
    ticker = st.text_input("Ticker", value="NXXT")
    if st.button("Fetch Market", type="primary"):
        st.session_state["market"] = fetch_market_data(ticker)

market = st.session_state.get("market", fetch_market_data(ticker))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Price", f"${fmt_money(market.get('price'))}" if market.get("price") is not None else "—")
m2.metric("% off 52w low", fmt_pct(market.get("pct_off_low")) if market.get("pct_off_low") is not None else "—")
m3.metric("Vol vs 10d avg", f"{market['vol_mult10']:.2f}x" if market.get("vol_mult10") is not None else "—")
m4.metric("52w Low / High", f"{fmt_money(market.get('low_52w'),2)} / {fmt_money(market.get('high_52w'),2)}" if market.get("low_52w") else "—")
m5.metric("Mkt Cap", fmt_big_num(market.get("mcap_live")))

# Levels
st.subheader("Technical Levels")
auto = st.toggle("Auto-calc levels", value=True)

if "levels" not in st.session_state or auto:
    st.session_state["levels"] = auto_levels(market.get("price"))

lv = st.session_state["levels"]
l1, l2, l3, l4, l5 = st.columns(5)
with l1:
    support = st.number_input("Support", value=float(lv["support"] or 0.0), step=0.01, format="%.2f") if not auto else lv["support"]
with l2:
    mid = st.number_input("Mid-structure", value=float(lv["mid"] or 0.0), step=0.01, format="%.2f") if not auto else lv["mid"]
with l3:
    trigger = st.number_input("Trigger", value=float(lv["trigger"] or 0.0), step=0.01, format="%.2f") if not auto else lv["trigger"]
with l4:
    exp1 = st.number_input("Expansion A", value=float(lv["exp1"] or 0.0), step=0.01, format="%.2f") if not auto else lv["exp1"]
with l5:
    exp2 = st.number_input("Expansion B", value=float(lv["exp2"] or 0.0), step=0.01, format="%.2f") if not auto else lv["exp2"]

levels = {"support": support, "mid": mid, "trigger": trigger, "exp1": exp1, "exp2": exp2}

st.divider()

# News
st.subheader("Optional News Catalyst")
tab_url, tab_rss = st.tabs(["Paste URL", "RSS (optional)"])

if "article" not in st.session_state:
    st.session_state["article"] = {"url": "", "title": "", "text": ""}

with tab_url:
    url_in = st.text_input("News URL", value="")
    if st.button("Fetch Article", type="primary"):
        if not url_in.strip():
            st.warning("Paste a URL first.")
        else:
            try:
                st.session_state["article"] = fetch_article(url_in.strip())
                st.success("Fetched article text.")
            except Exception as e:
                st.error(f"Fetch error: {e}")

    art = st.session_state["article"]
    if art.get("title") or art.get("text"):
        st.text_input("Extracted title", value=art.get("title", ""), disabled=True)
        st.text_area("Extracted text", value=art.get("text", ""), height=180)

with tab_rss:
    if st.button("Fetch RSS"):
        st.session_state["rss"] = fetch_news_rss(ticker, limit=12)
    rss = st.session_state.get("rss", [])
    if rss:
        if "rss_sel" not in st.session_state:
            st.session_state["rss_sel"] = list(range(min(2, len(rss))))
        selected = set(st.session_state["rss_sel"])
        new_sel = []
        for i, it in enumerate(rss[:12]):
            if st.checkbox(f"{i+1}. {it['title']}", value=(i in selected), key=f"rss_{i}"):
                new_sel.append(i)
        st.session_state["rss_sel"] = new_sel
    else:
        st.info("RSS optional. URL works best.")

st.divider()

# User context (your “exclusive agreement” etc.)
st.subheader("Extra Context (optional, but useful)")
user_context = st.text_area(
    "Paste any specific NXXT facts you want enforced (exclusive agreement details, contracts, filings, etc.)",
    height=120,
    value="",
)

# =========================================================
# Generate
# =========================================================
def generate_one(lang: str, variation_seed: int) -> str:
    random.seed(variation_seed + 1337)

    platform_block = platform_rules(platform_key)
    format_block = output_format_rules(fmt_key)

    # Add slang rules only if Twitter + enabled
    slang_block = ""
    if platform_key == "twitter" and twitter_slang:
        slang_block = """
Trader slang allowed (use lightly, not spam):
runner, squeeze mechanics (conditional), breakout, reclaim, VWAP, trend, bid, dip-buyers, momentum, volume pop, liquidity compression, low float.
""".strip()

    # Hashtag instruction
    hashtag_block = ""
    if platform_key == "twitter":
        hashtag_block = "Hashtags: allow max 1." if allow_hashtag else "Hashtags: do not use hashtags."

    system = "\n\n".join(
        [
            MASTER_SYSTEM,
            MACRO_NARRATIVE,
            SECTOR_VALIDATION_NUMBERS,
            NXXT_SECTION,
            TECHNICAL_SECTION,
            INSTITUTIONAL_SIGNAL,
            FINAL_NARRATIVE,
            platform_block,
            format_block,
            slang_block,
            hashtag_block,
            f"Variation seed: {variation_seed}. Rewrite phrasing and rhythm. Avoid repeating sentence templates.",
        ]
    ).strip()

    user = build_master_user_prompt(
        ticker=ticker,
        market=market,
        levels=levels,
        platform_key=platform_key,
        fmt_key=fmt_key,
        include_hashtag=allow_hashtag,
        trader_slang=twitter_slang,
        user_context=user_context,
        news=st.session_state.get("article", {"url": "", "title": "", "text": ""}),
    )

    out = call_gpt(api_key, model, system, user, temperature=temperature)
    out = postprocess(out, platform_key=platform_key, ticker=ticker, include_hashtag=allow_hashtag)
    return out


if st.button("Generate", type="primary"):
    if not api_key:
        st.error("No API key. Add OPENAI_API_KEY in Streamlit Secrets or paste it in sidebar.")
        st.stop()

    outs = []
    for v in range(int(n_variations)):
        base_lang = "EN" if out_lang != "RU only" else "RU"
        txt = generate_one(base_lang, v)

        if out_lang == "EN only":
            outs.append({"variation": v + 1, "lang": "EN", "platform": dict(PLATFORMS)[platform_key], "format": dict(OUTPUT_FORMATS)[fmt_key], "text": txt})
        elif out_lang == "RU only":
            if base_lang != "RU":
                txt = translate_text(api_key, model, txt, "Russian")
                txt = postprocess(txt, platform_key=platform_key, ticker=ticker, include_hashtag=allow_hashtag)
            outs.append({"variation": v + 1, "lang": "RU", "platform": dict(PLATFORMS)[platform_key], "format": dict(OUTPUT_FORMATS)[fmt_key], "text": txt})
        else:
            en = txt if base_lang == "EN" else translate_text(api_key, model, txt, "English")
            ru = translate_text(api_key, model, en, "Russian")
            en = postprocess(en, platform_key=platform_key, ticker=ticker, include_hashtag=allow_hashtag)
            ru = postprocess(ru, platform_key=platform_key, ticker=ticker, include_hashtag=allow_hashtag)
            outs.append({"variation": v + 1, "lang": "EN", "platform": dict(PLATFORMS)[platform_key], "format": dict(OUTPUT_FORMATS)[fmt_key], "text": en})
            outs.append({"variation": v + 1, "lang": "RU", "platform": dict(PLATFORMS)[platform_key], "format": dict(OUTPUT_FORMATS)[fmt_key], "text": ru})

    st.session_state["generated"] = outs
    st.success(f"Generated {len(outs)} outputs.")

rows = st.session_state.get("generated", [])
if rows:
    df = pd.DataFrame(rows)
    for v in sorted(df["variation"].unique().tolist()):
        st.markdown(f"### Variation {v}")
        sub = df[df["variation"] == v]

        if out_lang == "EN + RU (translation)":
            en = sub[sub["lang"] == "EN"]["text"].values[0]
            ru = sub[sub["lang"] == "RU"]["text"].values[0]
            L, R = st.columns(2)
            with L:
                st.markdown("**EN**")
                st.text_area(f"EN v{v}", value=en, height=420)
            with R:
                st.markdown("**RU**")
                st.text_area(f"RU v{v}", value=ru, height=420)
        else:
            txt = sub["text"].values[0]
            st.text_area(f"v{v}", value=txt, height=420)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="nxxt_master_outputs.csv",
        mime="text/csv",
    )
