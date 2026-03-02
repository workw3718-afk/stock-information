# app.py
import os
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# =========================================================
# API key / client
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
        raise RuntimeError("OPENAI_API_KEY missing. Add to Streamlit Secrets or paste in sidebar.")
    return OpenAI(api_key=api_key)


# =========================================================
# Small helpers
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


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def normalize_ws(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def clean_no_long_dashes(s: str) -> str:
    return (s or "").replace("—", "-").replace("–", "-")


def truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def ensure_title_body(text: str) -> str:
    """
    Normalize output to:
    TITLE: ...
    BODY: ...
    No extra headings.
    """
    t = normalize_ws(text)

    # allow RU headings
    t = re.sub(r"^(ЗАГОЛОВОК|Заголовок)\s*:\s*", "TITLE: ", t, flags=re.MULTILINE)
    t = re.sub(r"^(ТЕКСТ|Текст)\s*:\s*", "BODY: ", t, flags=re.MULTILINE)

    # remove accidental "Variation" etc
    t = re.sub(r"^\s*Variation\s*\d+\s*$", "", t, flags=re.MULTILINE).strip()

    if "TITLE:" not in t and "BODY:" not in t:
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if not lines:
            return "TITLE: \n\nBODY: "
        title = lines[0]
        body = "\n".join(lines[1:]).strip() or lines[0]
        return f"TITLE: {title}\n\nBODY: {body}"

    if "TITLE:" in t and "BODY:" not in t:
        after = t.split("TITLE:", 1)[1].strip()
        lines = after.split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        return f"TITLE: {title}\n\nBODY: {body}"

    # if BODY exists but TITLE missing
    if "BODY:" in t and "TITLE:" not in t:
        body = t.split("BODY:", 1)[1].strip()
        title = truncate(body.split("\n")[0].strip(), 90)
        return f"TITLE: {title}\n\nBODY: {body}"

    # keep only first TITLE/BODY pair if model repeats
    m = re.search(r"(TITLE:\s*.+?\n\nBODY:\s*.+)", t, flags=re.DOTALL)
    return (m.group(1).strip() if m else t.strip())


def count_number_anchors(body: str) -> int:
    """
    Count numeric anchors in text: digits with optional commas/decimals/%/$
    """
    if not body:
        return 0
    hits = re.findall(r"(?<!\w)(?:\$?\d[\d,]*(?:\.\d+)?%?)(?!\w)", body)
    return len(hits)


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


# =========================================================
# RSS
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
            {"title": (e.get("title") or "").strip(),
             "link": (e.get("link") or "").strip(),
             "summary": (e.get("summary") or "").strip()}
        )
    return items


# =========================================================
# URL article fetch (any page)
# =========================================================
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

    text = "\n\n".join(paras[:14]).strip()

    return {
        "url": url,
        "title": truncate(title, 160),
        "text": truncate(text, 3800),
    }


# =========================================================
# Market data
# =========================================================
@st.cache_data(ttl=45)
def fetch_market_data(ticker: str) -> Dict[str, Any]:
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
    prev_close: Optional[float] = None

    try:
        info = tk.get_info() or {}
        company = str(info.get("longName") or info.get("shortName") or "")[:120]
        shares_out = safe_float(info.get("sharesOutstanding"))
        market_cap_info = safe_float(info.get("marketCap"))
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

    if price is not None and shares_out not in (None, 0):
        market_cap_live = float(price) * float(shares_out)
    else:
        market_cap_live = market_cap_info

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


# =========================================================
# Styles / system prompts (NO boring corporate tone)
# =========================================================
@dataclass
class StylePreset:
    name: str
    description: str
    system_en: str
    system_ru: str


BANNED_PHRASES = [
    "well-positioned",
    "positioned for growth",
    "strategic positioning",
    "stands to benefit significantly",
    "pivotal moment",
    "increasing need for",
    "robust",
    "game-changer",
]

CORE_RULES = """
You generate Quora-style posts. Constraints:
- Output MUST be exactly:
  TITLE: <one line>
  BODY: <text in 1-3 short paragraphs>
- No bullet lists. No numbering. No headings beyond TITLE/BODY.
- Sound human, direct, and specific. No press-release tone.
- No buy/sell, no price targets, no "not financial advice".
- No emojis. No long dashes.
- MUST connect: news -> why it matters -> why it matters for the ticker.
- Ticker MUST appear in BODY at least once.
""".strip()

TITLE_RULES = """
TITLE rules:
- Create a UNIQUE title each time.
- Avoid repeating the same template.
- 6-14 words. Punchy. Not corporate.
""".strip()

NUMBERS_RULES = """
Numbers rules:
- BODY must contain at least 3 numeric anchors (%, $, dates, counts).
- Prefer: market cap, defense budget size, % of oil flows, % move, volume, etc.
- If company numbers are missing, use macro numbers provided in input.
""".strip()

STYLE_EN_ANALYST = f"""
{CORE_RULES}
{TITLE_RULES}
{NUMBERS_RULES}
Write in English. Dense, clean, non-hype.
""".strip()

STYLE_EN_AGGRO = f"""
{CORE_RULES}
{TITLE_RULES}
{NUMBERS_RULES}
Write in English. More aggressive, more market-aware. Short sentences. Hard claims only if supported by input.
""".strip()

STYLE_RU_ANALYST = f"""
{CORE_RULES}
{TITLE_RULES}
{NUMBERS_RULES}
Пиши по-русски. Плотно, конкретно, без канцелярита.
""".strip()

STYLE_RU_AGGRO = f"""
{CORE_RULES}
{TITLE_RULES}
{NUMBERS_RULES}
Пиши по-русски. Агрессивнее, ближе к трейдерскому мышлению, но без призывов купить/продать.
""".strip()

STYLES: List[StylePreset] = [
    StylePreset(
        name="Analyst (clean)",
        description="Плотно, конкретно, без корпоративщины.",
        system_en=STYLE_EN_ANALYST,
        system_ru=STYLE_RU_ANALYST,
    ),
    StylePreset(
        name="Aggressive (market-aware)",
        description="Жёстче, быстрее, больше давления/масштаба.",
        system_en=STYLE_EN_AGGRO,
        system_ru=STYLE_RU_AGGRO,
    ),
]


# =========================================================
# OpenAI calls
# =========================================================
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
        "Do not rewrite, summarize, add, remove, or change meaning. "
        "Preserve TITLE/BODY and paragraph breaks exactly. Preserve numbers and tickers."
    )
    user = f"Translate to {target_lang}. Text:\n\n{text}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# =========================================================
# Prompt builder (forces macro numbers + anti-boring)
# =========================================================
def build_prompt(
    ticker: str,
    company_name: str,
    company_one_liner: str,
    market: Dict[str, Any],
    rss_items: List[Dict[str, str]],
    selected_rss_idx: List[int],
    article_url: str,
    article_title: str,
    article_text: str,
    title_mode: str,
    tone_seed: int,
    chars_min: int,
    chars_max: int,
) -> str:
    t = ticker.strip().upper()

    # Title mode behavior
    if title_mode == "mixed":
        effective = random.choice(["exclude_company", "include_company"])
    else:
        effective = title_mode

    if effective == "exclude_company":
        title_directive = "In TITLE: do NOT mention the company name."
    else:
        title_directive = f"In TITLE: you MAY mention the company name ({company_name}) but don't force it."

    # Choose angle each time to avoid same rhythm
    angle = random.choice([
        "macro oil risk premium and energy volatility",
        "defense budget and infrastructure procurement",
        "grid resilience economics and payback math",
        "why markets reprice small-cap infrastructure fast",
        "what changes in capital allocation under escalation",
    ])

    # Build market anchors (these count as numeric anchors if used)
    anchors = []
    if market.get("mcap_live") is not None:
        anchors.append(f"Market cap (live): {int(market['mcap_live']):,}")
    if market.get("price") is not None:
        anchors.append(f"Last price: {market['price']:.4f}")
    if market.get("chg_pct") is not None:
        anchors.append(f"Day %: {market['chg_pct']:.2f}%")
    if market.get("vol") is not None and market.get("vol10") is not None:
        anchors.append(f"Volume: {int(market['vol']):,} vs 10d avg {int(market['vol10']):,}")

    # Macro anchors (fallback if market data is missing)
    # Keep these stable and safe — they exist to force numeric anchors in text.
    macro_anchors = [
        "Macro anchor: ~20% of global oil flows through the Strait of Hormuz.",
        "Macro anchor: U.S. defense spending is roughly $850B+ annually (order-of-magnitude context).",
        "Macro anchor: energy cost swings of 10-25% can change project payback decisions.",
    ]

    # RSS block
    rss_sel = [rss_items[i] for i in selected_rss_idx if 0 <= i < len(rss_items)]
    rss_block = []
    if rss_sel:
        rss_block.append("Selected RSS headlines (secondary context):")
        for it in rss_sel:
            rss_block.append(f"- {it['title']}")
            if it.get("summary"):
                rss_block.append(f"  {truncate(strip_html(it['summary']), 240)}")

    # Article block (primary context)
    art_block = []
    if article_url.strip():
        art_block.append(f"News URL: {article_url.strip()}")
    if article_title.strip():
        art_block.append(f"Extracted title: {article_title.strip()}")
    if article_text.strip():
        art_block.append("Extracted article text (primary):")
        art_block.append(truncate(article_text.strip(), 3600))

    # Company block
    comp_block = []
    if company_one_liner.strip():
        comp_block.append(f"Company context (use only if relevant): {company_one_liner.strip()}")
    else:
        comp_block.append("Company context: keep minimal. Do not invent products or contracts.")

    # Anti-boring explicit
    anti_boring = f"""
Banned phrases (do not use): {", ".join(BANNED_PHRASES)}.
Avoid generic filler (no "increasing need", "robust", "pivotal moment").
Write like a trader-analyst: pressure, scale, numbers, relevance.
""".strip()

    hard = f"""
Hard requirements:
- BODY must include at least 3 numeric anchors.
- Mention ticker {t} in BODY at least once.
- Title uniqueness required.
- No bullet points.
- Character target: {chars_min}-{chars_max}.
- Angle: {angle}.
Seed: {tone_seed}.
{title_directive}
""".strip()

    parts = [
        f"Ticker: {t}",
        f"Company name: {company_name}".strip(),
        "\n".join(comp_block),
        ("Data anchors (use 1-2 max):\n" + "\n".join(anchors)) if anchors else "",
        ("Macro anchors (use if needed):\n" + "\n".join(macro_anchors)),
        "\n".join(art_block),
        "\n".join(rss_block),
        anti_boring,
        hard,
    ]
    return "\n\n".join([p for p in parts if p and p.strip()])


def postprocess(text: str, ticker: str) -> str:
    """
    Enforce:
    - TITLE/BODY only
    - remove banned phrases
    - ensure numeric anchors count >= 3 (retry should handle, but we check)
    """
    t = ensure_title_body(clean_no_long_dashes(text))

    # remove banned phrases if model slips (soft)
    for ph in BANNED_PHRASES:
        t = re.sub(re.escape(ph), "", t, flags=re.IGNORECASE)

    # ensure ticker exists in BODY
    tick = ticker.strip().upper()
    if "BODY:" in t:
        title, body = t.split("BODY:", 1)
        body_clean = body.strip()
        if tick not in body_clean:
            body_clean = f"{body_clean}\n\n{tick}"
        t = title.strip() + "\n\nBODY: " + body_clean

        # ensure numeric anchors >=3; if not, append minimal numeric line
        if count_number_anchors(body_clean) < 3:
            body_clean += "\n\n20% oil flow. $850B defense spend. 10-25% power cost swings."
            t = title.strip() + "\n\nBODY: " + body_clean

    return t.strip()


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Post Studio", page_icon="⚡", layout="wide")
st.title("⚡ Quora Post Studio (Less boring, numbers forced, unique titles)")

with st.sidebar:
    st.header("OpenAI")
    api_key_ui = st.text_input("API Key (optional if Secrets has it)", value="", type="password")
    api_key = resolve_api_key(api_key_ui)

    model = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Creativity", 0.5, 1.2, 1.02, 0.05)

    st.divider()
    st.header("Controls")
    out_lang = st.selectbox("Output languages", ["EN + RU (translation)", "EN only", "RU only"], index=0)
    style_name = st.selectbox("Style", [s.name for s in STYLES], index=1)
    title_mode = st.selectbox("Title mode", ["mixed", "exclude_company", "include_company"], index=0)
    n_variations = st.number_input("Variations", min_value=1, max_value=20, value=3, step=1)
    chars_min, chars_max = st.slider("Length (characters)", 250, 2200, (650, 1100), step=10)

st.subheader("Ticker")
c1, c2 = st.columns([1.1, 1.2])
with c1:
    ticker = st.text_input("Ticker", value="NXXT")
    if st.button("Fetch Market + RSS", type="primary"):
        st.session_state["market"] = fetch_market_data(ticker)
        st.session_state["rss"] = fetch_news_rss(ticker, limit=12)

market = st.session_state.get("market", fetch_market_data(ticker))
rss = st.session_state.get("rss", [])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"${fmt_money(market.get('price'))}" if market.get("price") is not None else "—")
chg = market.get("chg_pct")
m2.metric("Chg %", fmt_pct(chg) if chg is not None else "—")
m2.markdown(pct_badge_html(chg), unsafe_allow_html=True)
m3.metric("Mkt Cap", fmt_big_num(market.get("mcap_live")))
m4.metric("Company", market.get("company") or "—")

with c2:
    company_name = st.text_input("Company name", value=(market.get("company") or "NextNRG").strip())
    company_one_liner = st.text_area(
        "Company one-liner (optional)",
        value="Distributed energy + microgrid / storage / resilience infrastructure layer.",
        height=80,
    )

st.divider()
st.subheader("News input (URL preferred)")

tab_url, tab_rss = st.tabs(["Paste URL", "RSS (optional)"])

article_url = ""
article_title = ""
article_text = ""

with tab_url:
    article_url = st.text_input("Paste news URL", value="")
    if st.button("Fetch article text"):
        if not article_url.strip():
            st.warning("Paste URL first.")
        else:
            try:
                art = fetch_article(article_url.strip())
                st.session_state["article"] = art
                st.success("Fetched.")
            except Exception as e:
                st.error(f"Fetch error: {e}")

    art = st.session_state.get("article", {"url": "", "title": "", "text": ""})
    article_title = art.get("title", "")
    article_text = art.get("text", "")

    if article_title or article_text:
        st.text_input("Extracted title", value=article_title, disabled=True)
        st.text_area("Extracted text", value=article_text, height=220)

with tab_rss:
    if not rss:
        st.info("Click 'Fetch Market + RSS' first, or use URL mode.")
    else:
        if "selected_rss" not in st.session_state:
            st.session_state["selected_rss"] = list(range(min(2, len(rss))))
        selected = set(st.session_state.get("selected_rss", []))
        new_selected = []
        for i, it in enumerate(rss[:12]):
            label = f"{i+1}. {it['title']}"
            if st.checkbox(label, value=(i in selected), key=f"rss_{i}"):
                new_selected.append(i)
        st.session_state["selected_rss"] = new_selected

st.divider()
st.subheader("Generate")

def generate_posts() -> List[Dict[str, Any]]:
    if not api_key:
        raise RuntimeError("No API key. Add OPENAI_API_KEY in Secrets or paste in sidebar.")

    style = next(s for s in STYLES if s.name == style_name)
    rows: List[Dict[str, Any]] = []
    selected_idx = st.session_state.get("selected_rss", [])

    # If user didn't fetch URL text, still allow RSS-only
    for v in range(int(n_variations)):
        prompt = build_prompt(
            ticker=ticker,
            company_name=company_name.strip() or "NextNRG",
            company_one_liner=company_one_liner,
            market=market,
            rss_items=rss,
            selected_rss_idx=selected_idx[:3] if rss else [],
            article_url=art.get("url", "") if "article" in st.session_state else article_url,
            article_title=article_title,
            article_text=article_text,
            title_mode=title_mode,
            tone_seed=v,
            chars_min=chars_min,
            chars_max=chars_max,
        )

        # Generate base in EN or RU depending on out_lang selection
        if out_lang == "RU only":
            sys = style.system_ru
            base = call_gpt(api_key, model, sys, prompt, temperature=temperature)
            base = postprocess(base, ticker)

            rows.append({"variation": v + 1, "lang": "RU", "ticker": ticker.strip().upper(), "style": style_name, "text": base})

        else:
            sys = style.system_en
            base = call_gpt(api_key, model, sys, prompt, temperature=temperature)
            base = postprocess(base, ticker)

            if out_lang == "EN only":
                rows.append({"variation": v + 1, "lang": "EN", "ticker": ticker.strip().upper(), "style": style_name, "text": base})
            else:
                ru = translate_text(api_key, model, base, "Russian")
                ru = postprocess(ru, ticker)
                rows.append({"variation": v + 1, "lang": "EN", "ticker": ticker.strip().upper(), "style": style_name, "text": base})
                rows.append({"variation": v + 1, "lang": "RU", "ticker": ticker.strip().upper(), "style": style_name, "text": ru})

    return rows


if st.button("Generate posts", type="primary"):
    try:
        rows = generate_posts()
        st.session_state["generated_rows"] = rows
        st.success(f"Generated {len(rows)} rows.")
    except Exception as e:
        st.error(str(e))

rows = st.session_state.get("generated_rows", [])
if rows:
    df = pd.DataFrame(rows)
    variations = sorted(df["variation"].unique().tolist())

    for v in variations:
        st.markdown(f"### Variation {v}")
        sub = df[df["variation"] == v]

        if out_lang == "EN + RU (translation)":
            en = sub[sub["lang"] == "EN"]["text"].values[0]
            ru = sub[sub["lang"] == "RU"]["text"].values[0]
            L, R = st.columns(2)
            with L:
                st.markdown("**English**")
                st.text_area(f"EN v{v}", value=en, height=420)
            with R:
                st.markdown("**Русский (перевод)**")
                st.text_area(f"RU v{v}", value=ru, height=420)
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
