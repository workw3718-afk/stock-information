# app.py
import os
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# =============================
# OpenAI key / client
# =============================
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


# =============================
# Helpers
# =============================
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
    Ensure output contains TITLE: and BODY:
    Keep model variability, do NOT force 3 paragraphs.
    """
    t = normalize_ws(text)
    t = re.sub(r"^(ЗАГОЛОВОК|Заголовок)\s*:\s*", "TITLE: ", t, flags=re.MULTILINE)
    t = re.sub(r"^(ТЕКСТ|Текст)\s*:\s*", "BODY: ", t, flags=re.MULTILINE)

    if "TITLE:" not in t and "BODY:" not in t:
        # assume first line is title
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if not lines:
            return "TITLE: \n\nBODY: "
        title = lines[0]
        body = "\n".join(lines[1:]).strip() or lines[0]
        return f"TITLE: {title}\n\nBODY: {body}"

    if "TITLE:" in t and "BODY:" not in t:
        # split title from rest
        after = t.split("TITLE:", 1)[1].strip()
        lines = after.split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        return f"TITLE: {title}\n\nBODY: {body}"

    return t.strip()


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
# News (RSS)
# =============================
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


# =============================
# Article fetch (any URL)
# =============================
@st.cache_data(ttl=600)
def fetch_article(url: str, timeout: int = 12) -> Dict[str, str]:
    """
    Lightweight HTML extraction for a news URL.
    """
    url = (url or "").strip()
    if not url:
        return {"url": "", "title": "", "text": ""}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # title
    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    # meta og:title
    ogt = soup.find("meta", property="og:title")
    if ogt and ogt.get("content"):
        title = ogt.get("content").strip() or title

    # main text: grab paragraphs
    # heuristic: take visible <p> blocks, filter too-short
    ps = soup.find_all("p")
    paras = []
    for p in ps:
        txt = p.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) >= 40:
            paras.append(txt)

    # reduce boilerplate: keep first ~12 paras max
    text = "\n\n".join(paras[:12]).strip()

    # final cleanup
    title = truncate(title, 160)
    text = truncate(text, 3500)

    return {"url": url, "title": title, "text": text}


# =============================
# Market data
# =============================
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


# =============================
# Style presets
# =============================
@dataclass
class StylePreset:
    name: str
    description: str
    system_en: str
    system_ru: str


BASE_RULES = """
Core rules:
- Write like a real human, not a template.
- Vary rhythm, sentence length, and structure each time.
- No emojis. No long dashes.
- No price targets. No buy/sell. No "financial advice".
- MUST connect: News -> Why it matters -> Why it matters for the ticker.
- Ticker must appear in BODY at least once (e.g., NXXT).
""".strip()

TITLE_VARIATION_RULES = """
Title rules:
- Generate a UNIQUE title each time.
- Avoid repeating patterns like "Positioned for Growth", "Strategic Positioning", etc.
- If instructed to exclude company name from title, do NOT mention the company name in TITLE.
- Title should be 6-14 words, punchy, not corporate.
""".strip()

PARA_RULES = """
Body structure:
- Use 1 to 3 paragraphs depending on what fits best.
- Keep it readable. Prefer dense but clear writing.
""".strip()

NUMBERS_RULES = """
Numbers:
- If numbers are available in the input (market cap, % move, etc.), use 1-2 of them max.
- Do NOT force numbers if they don't add value.
""".strip()

STYLES: List[StylePreset] = [
    StylePreset(
        name="Quora Analyst (clean)",
        description="Analytical, execution-focused, not hype.",
        system_en=f"You write Quora posts in English.\n{BASE_RULES}\n{TITLE_VARIATION_RULES}\n{PARA_RULES}\n{NUMBERS_RULES}",
        system_ru=f"Ты пишешь посты для Quora на русском.\n{BASE_RULES}\n{TITLE_VARIATION_RULES}\n{PARA_RULES}\n{NUMBERS_RULES}",
    ),
    StylePreset(
        name="Macro / Energy Angle",
        description="Macro -> energy -> infrastructure relevance.",
        system_en=f"You write Quora posts in English with a macro/energy framing.\n{BASE_RULES}\n{TITLE_VARIATION_RULES}\n{PARA_RULES}\n{NUMBERS_RULES}",
        system_ru=f"Ты пишешь посты для Quora на русском с макро/энергетическим углом.\n{BASE_RULES}\n{TITLE_VARIATION_RULES}\n{PARA_RULES}\n{NUMBERS_RULES}",
    ),
    StylePreset(
        name="Short & Punchy",
        description="Tighter, faster read, minimal fluff.",
        system_en=f"You write Quora posts in English.\n{BASE_RULES}\n{TITLE_VARIATION_RULES}\nBody should be 1-2 paragraphs only.\n{NUMBERS_RULES}",
        system_ru=f"Ты пишешь посты для Quora на русском.\n{BASE_RULES}\n{TITLE_VARIATION_RULES}\nТекст должен быть 1-2 абзаца.\n{NUMBERS_RULES}",
    ),
]


# =============================
# OpenAI call
# =============================
def call_gpt(api_key: str, model: str, system: str, user: str, temperature: float = 0.9) -> str:
    client = get_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def translate_text(api_key: str, model: str, text: str, target_lang: str) -> str:
    client = get_client(api_key)
    sys = (
        "You are a translation engine. Translate ONLY. "
        "Do not rewrite, summarize, add, remove, or change meaning. "
        "Preserve structure, headings like TITLE/BODY, numbers, tickers, and punctuation. "
        "Preserve paragraph breaks exactly."
    )
    user = f"Translate to {target_lang}. Text:\n\n{text}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# =============================
# Prompt builder
# =============================
ANGLE_BANK = [
    "execution + credibility angle",
    "macro + oil risk premium angle",
    "grid resilience + infrastructure angle",
    "defense procurement relevance angle",
    "why investors reprice risk angle",
    "why this changes narrative angle",
    "what this means for capex cycles angle",
]

TITLE_MODE_BANK = ["exclude_company", "include_company", "mixed"]

def build_user_prompt(
    ticker: str,
    company_name: str,
    company_one_liner: str,
    market: Dict[str, Any],
    rss_items: List[Dict[str, str]],
    selected_rss_idx: List[int],
    pasted_news_url: str,
    pasted_news_text: str,
    pasted_news_title: str,
    user_notes: str,
    include_market_sentence: bool,
    title_mode: str,
    positive_only: bool,
    chars_min: int,
    chars_max: int,
    variation_seed: int,
) -> str:
    t = ticker.strip().upper()
    angle = random.choice(ANGLE_BANK)

    # Title include/exclude directive
    if title_mode == "mixed":
        title_mode_effective = random.choice(["exclude_company", "include_company"])
    else:
        title_mode_effective = title_mode

    if title_mode_effective == "exclude_company":
        title_directive = "TITLE MUST NOT mention the company name. Use only the topic/implication."
    else:
        # allow company name sometimes, but not mandatory
        title_directive = f"TITLE may mention the company name ({company_name}) but do not force it."

    # sentiment directive
    if positive_only:
        sentiment = "Keep framing constructive/positive. Do not include risk warnings or bearish language."
    else:
        sentiment = "Balanced tone allowed. You may mention risks briefly, but keep it neutral and not alarmist."

    # market sentence optional
    market_lines = []
    if include_market_sentence:
        if market.get("price") is not None:
            market_lines.append(f"Last price (optional): ${market['price']:.4f}")
        if market.get("chg_pct") is not None:
            market_lines.append(f"Day change % (optional): {market['chg_pct']:.2f}%")
        if market.get("mcap_live") is not None:
            market_lines.append(f"Market cap (optional): {market['mcap_live']:.0f}")
        if market.get("vol") is not None and market.get("vol10") is not None:
            market_lines.append(f"Volume (optional): {int(market['vol'])} vs 10d avg {int(market['vol10'])}")

    # RSS selection
    rss_block = []
    sel = [rss_items[i] for i in selected_rss_idx if 0 <= i < len(rss_items)]
    if sel:
        rss_block.append("Selected RSS headlines:")
        for it in sel:
            rss_block.append(f"- {it.get('title','').strip()}")
            if it.get("summary"):
                rss_block.append(f"  Summary: {truncate(strip_html(it['summary']), 260)}")
            if it.get("link"):
                rss_block.append(f"  Link: {it['link']}")

    # pasted news block
    pasted_block = []
    if pasted_news_url.strip():
        pasted_block.append(f"News URL: {pasted_news_url.strip()}")
    if pasted_news_title.strip():
        pasted_block.append(f"Extracted page title: {pasted_news_title.strip()}")
    if pasted_news_text.strip():
        pasted_block.append("Extracted article text (primary context):")
        pasted_block.append(truncate(pasted_news_text.strip(), 3500))

    # company context
    company_block = []
    if company_one_liner.strip():
        company_block.append(f"Company one-liner (use only if relevant): {company_one_liner.strip()}")
    else:
        # fallback: allow model to use company name minimally
        company_block.append("Company context is not provided; keep the company description minimal and tied to the news.")

    # user notes
    notes_block = []
    if user_notes.strip():
        notes_block.append(f"User note / focus: {user_notes.strip()}")

    # hard output requirements
    hard = f"""
OUTPUT FORMAT (must match):
TITLE: <one line>
BODY: <text>
No other headers, no bullet lists.
Character target: {chars_min}-{chars_max}.
Ticker requirement: BODY must mention {t} at least once.
{title_directive}
Write with {angle}.
{sentiment}
Seed hint: {variation_seed}
""".strip()

    pieces = [
        f"Ticker: {t}",
        f"Company name: {company_name}".strip(),
        "\n".join(company_block).strip(),
        ("\n".join(market_lines)).strip(),
        ("\n".join(rss_block)).strip(),
        ("\n".join(pasted_block)).strip(),
        ("\n".join(notes_block)).strip(),
        hard,
    ]

    # remove empties
    return "\n\n".join([p for p in pieces if p and p.strip()])


# =============================
# UI
# =============================
st.set_page_config(page_title="Post Studio", page_icon="⚡", layout="wide")
st.title("⚡ Quora Post Studio (News -> Importance -> Ticker relevance)")

with st.sidebar:
    st.header("OpenAI")
    api_key_ui = st.text_input("API Key (optional if Secrets has it)", value="", type="password")
    api_key = resolve_api_key(api_key_ui)

    model = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Creativity (temperature)", 0.4, 1.2, 0.95, 0.05)

    st.divider()
    st.header("Output controls")
    title_mode = st.selectbox("Title mode", ["mixed", "exclude_company", "include_company"], index=0)
    positive_only = st.toggle("Positive-only framing", value=True)
    include_market_sentence = st.toggle("Allow 1 short market-data sentence", value=True)
    out_lang = st.selectbox("Output languages", ["EN + RU (translation)", "EN only", "RU only"], index=0)
    n_variations = st.number_input("Variations", min_value=1, max_value=20, value=3, step=1)
    chars_min, chars_max = st.slider("Length (characters)", 250, 2200, (700, 1200), step=10)

st.subheader("Ticker + Context")
c1, c2 = st.columns([1.1, 1.2])
with c1:
    ticker = st.text_input("Ticker", value="NXXT")
    market = fetch_market_data(ticker)

    a, b, c, d = st.columns(4)
    a.metric("Price", f"${fmt_money(market.get('price'))}" if market.get("price") is not None else "—")
    chg = market.get("chg_pct")
    b.metric("Chg %", fmt_pct(chg) if chg is not None else "—")
    b.markdown(pct_badge_html(chg), unsafe_allow_html=True)
    c.metric("Mkt Cap", fmt_big_num(market.get("mcap_live")))
    d.metric("Company", market.get("company") or "—")

with c2:
    company_name = st.text_input("Company name (for title/body rules)", value=(market.get("company") or "NextNRG").strip())
    company_one_liner = st.text_area(
        "Company one-liner (optional, helps relevance)",
        value="Distributed energy + microgrid / storage / resilience-oriented infrastructure.",
        height=80,
    )
    user_notes = st.text_area(
        "Your note (optional): what angle do you want this time?",
        value="Tie the news to energy volatility / defense infrastructure relevance and why this matters for the ticker.",
        height=80,
    )

st.divider()
st.subheader("News input (choose 1: URL OR RSS)")

tab_url, tab_rss = st.tabs(["Paste URL", "StockTitan RSS (optional)"])

pasted_news_url = ""
pasted_news_title = ""
pasted_news_text = ""

with tab_url:
    pasted_news_url = st.text_input("Paste a news URL", value="")
    if st.button("Fetch article from URL", type="primary"):
        if not pasted_news_url.strip():
            st.warning("Paste a URL first.")
        else:
            try:
                art = fetch_article(pasted_news_url.strip())
                st.session_state["article"] = art
                st.success("Fetched.")
            except Exception as e:
                st.error(f"Fetch error: {e}")

    art = st.session_state.get("article", {"url": "", "title": "", "text": ""})
    pasted_news_title = art.get("title", "")
    pasted_news_text = art.get("text", "")

    if pasted_news_title or pasted_news_text:
        st.markdown("**Extracted preview**")
        st.text_input("Page title", value=pasted_news_title, disabled=True)
        st.text_area("Article text", value=pasted_news_text, height=220)

with tab_rss:
    if st.button("Load RSS headlines"):
        st.session_state["rss"] = fetch_news_rss(ticker, limit=12)

    rss = st.session_state.get("rss", [])
    if not rss:
        st.info("Click 'Load RSS headlines' (optional). If you paste a URL, RSS is not needed.")
        selected_rss_idx = []
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
        selected_rss_idx = new_selected

st.divider()
st.subheader("Generate")

def run_generation() -> List[Dict[str, Any]]:
    if not api_key:
        st.error("No API key. Add OPENAI_API_KEY in Streamlit Secrets OR paste it in sidebar.")
        st.stop()

    rss_items = st.session_state.get("rss", [])
    selected_idx = st.session_state.get("selected_rss", []) if rss_items else []

    rows: List[Dict[str, Any]] = []

    for v in range(int(n_variations)):
        prompt = build_user_prompt(
            ticker=ticker,
            company_name=company_name.strip() or "NextNRG",
            company_one_liner=company_one_liner,
            market=market,
            rss_items=rss_items,
            selected_rss_idx=selected_idx[:3],
            pasted_news_url=pasted_news_url or "",
            pasted_news_text=pasted_news_text or "",
            pasted_news_title=pasted_news_title or "",
            user_notes=user_notes,
            include_market_sentence=include_market_sentence,
            title_mode=title_mode,
            positive_only=positive_only,
            chars_min=chars_min,
            chars_max=chars_max,
            variation_seed=v,
        )

        style_name = st.session_state.get("style_name", STYLES[0].name)
        style = next(s for s in STYLES if s.name == style_name)

        # choose base language generation
        if out_lang == "RU only":
            sys = style.system_ru
            lang = "RU"
        else:
            sys = style.system_en
            lang = "EN"

        base = call_gpt(api_key, model, sys, prompt, temperature=temperature)
        base = clean_no_long_dashes(ensure_title_body(base))

        # enforce ticker in BODY
        t = ticker.strip().upper()
        if f" {t}" not in base and f"{t}" not in base:
            # append minimal if missing
            base = base.strip() + f"\n\nBODY: {t}"

        if out_lang == "EN only":
            rows.append({"variation": v + 1, "lang": "EN", "ticker": ticker.strip().upper(), "text": base})

        elif out_lang == "RU only":
            rows.append({"variation": v + 1, "lang": "RU", "ticker": ticker.strip().upper(), "text": base})

        else:
            # EN base + RU translation
            en_text = base
            ru_text = clean_no_long_dashes(translate_text(api_key, model, en_text, "Russian"))
            ru_text = ensure_title_body(ru_text)

            rows.append({"variation": v + 1, "lang": "EN", "ticker": ticker.strip().upper(), "text": en_text})
            rows.append({"variation": v + 1, "lang": "RU", "ticker": ticker.strip().upper(), "text": ru_text})

    return rows

colA, colB = st.columns([1.1, 1.0])
with colA:
    style_name = st.selectbox("Style", [s.name for s in STYLES], index=0)
    st.session_state["style_name"] = style_name
    st.caption(next(s for s in STYLES if s.name == style_name).description)

with colB:
    generate = st.button("Generate posts", type="primary")

if generate:
    try:
        rows = run_generation()
        st.session_state["generated_rows"] = rows
        st.success(f"Generated {len(rows)} rows.")
    except Exception as e:
        st.error(f"Error: {e}")

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
