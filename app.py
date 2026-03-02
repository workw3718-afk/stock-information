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
    Ensure output is exactly:
    TITLE: ...
    BODY: ...
    """
    t = normalize_ws(text)
    t = re.sub(r"^(ЗАГОЛОВОК|Заголовок)\s*:\s*", "TITLE: ", t, flags=re.MULTILINE)
    t = re.sub(r"^(ТЕКСТ|Текст)\s*:\s*", "BODY: ", t, flags=re.MULTILINE)
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

    if "BODY:" in t and "TITLE:" not in t:
        body = t.split("BODY:", 1)[1].strip()
        title = truncate((body.split("\n")[0] if body else "").strip(), 90)
        return f"TITLE: {title}\n\nBODY: {body}"

    m = re.search(r"(TITLE:\s*.+?\n\nBODY:\s*.+)", t, flags=re.DOTALL)
    return (m.group(1).strip() if m else t.strip())


def count_number_anchors(text: str) -> int:
    if not text:
        return 0
    hits = re.findall(r"(?<!\w)(?:\$?\d[\d,]*(?:\.\d+)?%?)(?!\w)", text)
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
# News: RSS + URL extractor
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

    text = "\n\n".join(paras[:14]).strip()
    return {"url": url, "title": truncate(title, 160), "text": truncate(text, 3800)}


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
# Persona + Platform config
# =========================================================
@dataclass
class Persona:
    key: str
    title: str
    subtitle: str
    icon: str
    accent: str
    system_en: str
    system_ru: str


@dataclass
class Platform:
    key: str
    title: str
    hint: str
    rules_en: str
    rules_ru: str
    default_len: Tuple[int, int]


BANNED_PHRASES = [
    "well-positioned",
    "positioned for growth",
    "strategic positioning",
    "stands to benefit significantly",
    "pivotal moment",
    "increasing need for",
    "robust",
    "game-changer",
    "peace of mind",
]

CORE_RULES_EN = """
Global rules:
- Sound like a real trader/analyst, not a press release.
- No buy/sell calls, no price targets, no "not financial advice".
- No long dashes.
- Avoid these phrases: {banned}.
- Connect: news -> why it matters -> why it matters for the ticker (or theme if short).
- Use numbers when possible (2-4 anchors). If forced, ensure >=3.
""".strip()

CORE_RULES_RU = """
Глобальные правила:
- Пиши как трейдер/аналитик, без пресс-релиза.
- Без "buy/sell", без таргетов, без "не финсовет".
- Без длинных тире.
- Избегай фраз: {banned}.
- Связка: новость -> почему важно -> почему важно для тикера (или темы, если коротко).
- Цифры: по возможности 2-4 якоря, если форс - >=3.
""".strip()

MACRO_ANCHORS_EN = [
    "~20% of global oil flows through the Strait of Hormuz (order-of-magnitude).",
    "U.S. defense spending is ~$850B+ annually (order-of-magnitude).",
    "~10-25% energy cost swings can change project payback math.",
]
MACRO_ANCHORS_RU = [
    "Около ~20% мировых потоков нефти проходят через Ормузский пролив (порядок величины).",
    "Оборонные расходы США порядка ~$850B+ в год (порядок величины).",
    "Колебания энергии ~10-25% меняют окупаемость проектов.",
]

TWITTER_SLANG_GUIDE_EN = """
Twitter trader slang guide (use selectively, no spam):
- Terms: runner, squeeze, breakout, reclaim, VWAP, trend, bid, dip-buyers, momentum, volume pop, low float, liquidity.
- Keep it punchy. 1-3 sentences.
- Mention ticker once as $TICKER.
""".strip()

TWITTER_SLANG_GUIDE_RU = """
Гайд для Twitter (трейдерский сленг, без спама):
- Термины: раннер, сквиз, брейк, ре-клейм, VWAP, тренд, бид, дип-байеры, моментум, всплеск объема, лоуфлоат, ликвидность.
- Коротко: 1-3 предложения.
- Тикер 1 раз в формате $TICKER.
""".strip()

PERSONAS: List[Persona] = [
    Persona(
        key="price_action",
        title="Price Action",
        subtitle="Technical",
        icon="📈",
        accent="#60a5fa",
        system_en="You are a price-action trader. Talk in levels, reclaim, trend, VWAP, volume. No fluff.",
        system_ru="Ты трейдер по прайс-экшену. Уровни, реклейм, тренд, VWAP, объем. Без воды.",
    ),
    Persona(
        key="deep_tech",
        title="Deep Tech",
        subtitle="Technical",
        icon="🧠",
        accent="#93c5fd",
        system_en="You are a deep-tech analyst. Focus on systems, constraints, deployment reality, measurable specifics.",
        system_ru="Ты deep-tech аналитик. Системы, ограничения, реальность внедрения, измеримые детали.",
    ),
    Persona(
        key="macro_energy",
        title="Macro / Energy",
        subtitle="Macro",
        icon="📊",
        accent="#a7f3d0",
        system_en="You are a macro strategist. Connect geopolitics, oil, risk premium, power costs, infrastructure.",
        system_ru="Ты макро-стратег. Связываешь геополитику, нефть, риск-премию, стоимость энергии, инфраструктуру.",
    ),
    Persona(
        key="value",
        title="Value Investor",
        subtitle="Value",
        icon="💼",
        accent="#86efac",
        system_en="You are a value investor. Focus on valuation math, dilution risk, balance sheet constraints.",
        system_ru="Ты value-инвестор. Оценка, риск разводнения, баланс и ограничения.",
    ),
    Persona(
        key="algo",
        title="Algo-Trader",
        subtitle="Quant",
        icon="🤖",
        accent="#67e8f9",
        system_en=(
            "You are an Algo-Trader signal account.\n"
            "Tone: bullish, trader-native, tight.\n"
            "Write like tape/flow/levels/volume/VWAP. Short bursts.\n"
            "Use slang: runner, squeeze, breakout, reclaim, bid, dip-buyers, momentum.\n"
            "Avoid corporate phrasing."
        ),
        system_ru=(
            "Ты алго/сигнал аккаунт.\n"
            "Тон: буллиш, трейдерский, плотный.\n"
            "Пиши как лента/поток/уровни/объем/VWAP. Короткими фразами.\n"
            "Сленг: раннер, сквиз, брейк, реклейм, бид, дип-байеры, моментум.\n"
            "Без корпоративщины."
        ),
    ),
    Persona(
        key="fomo",
        title="FOMO Emotion",
        subtitle="FOMO",
        icon="🔥",
        accent="#fb7185",
        system_en="You write with bullish urgency and market psychology, but still factual and not promotional.",
        system_ru="Пишешь с буллиш-срочностью и психологиями толпы, но фактично и без агрессивной рекламы.",
    ),
    Persona(
        key="journalism",
        title="Financial Journalism",
        subtitle="News",
        icon="📰",
        accent="#c4b5fd",
        system_en="You are a financial journalist. Clear, contextual, neutral, why-it-matters.",
        system_ru="Ты финансовый журналист. Ясно, контекстно, нейтрально, почему важно.",
    ),
    Persona(
        key="sellside",
        title="Sell-Side Analyst",
        subtitle="Research",
        icon="🧾",
        accent="#fbbf24",
        system_en="You write sell-side style: structured catalysts, execution, measurable statements.",
        system_ru="Пишешь как sell-side ресерч: катализаторы, исполнение, измеримые тезисы.",
    ),
]

PLATFORMS: List[Platform] = [
    Platform(
        key="twitter",
        title="Twitter / X",
        hint="1-3 sentences. Punchy. Trader slang optional. $TICKER once.",
        rules_en=(
            "Platform rules: Twitter/X.\n"
            "- Output MUST be ONLY the post text (no TITLE/BODY labels).\n"
            "- 1–3 sentences, max ~260 characters.\n"
            "- No bullet lists.\n"
            "- Mention ticker once in $TICKER format.\n"
            "- No price targets. No buy/sell.\n"
        ),
        rules_ru=(
            "Правила: Twitter/X.\n"
            "- Вывод ТОЛЬКО текст поста (без TITLE/BODY).\n"
            "- 1–3 предложения, до ~260 символов.\n"
            "- Без списков.\n"
            "- Тикер 1 раз в формате $TICKER.\n"
            "- Без таргетов и призывов.\n"
        ),
        default_len=(160, 260),
    ),
    Platform(
        key="reddit",
        title="Reddit",
        hint="2-5 sentences. Slightly longer. 1 optional line break.",
        rules_en=(
            "Platform rules: Reddit.\n"
            "- Output MUST be ONLY the post text (no TITLE/BODY labels).\n"
            "- 2–5 sentences. 1 optional line break.\n"
            "- Conversational, not corporate.\n"
        ),
        rules_ru=(
            "Правила: Reddit.\n"
            "- Вывод ТОЛЬКО текст поста (без TITLE/BODY).\n"
            "- 2–5 предложений. 1 перенос строки можно.\n"
            "- Разговорный тон, без корпоративщины.\n"
        ),
        default_len=(320, 700),
    ),
    Platform(
        key="quora",
        title="Quora",
        hint="TITLE + BODY, 1–3 short paragraphs. No bullet lists.",
        rules_en=(
            "Platform rules: Quora.\n"
            "- Output MUST be exactly:\n"
            "  TITLE: <one line>\n"
            "  BODY: <1-3 short paragraphs>\n"
            "- No bullet lists. No extra headings.\n"
        ),
        rules_ru=(
            "Правила: Quora.\n"
            "- Вывод строго:\n"
            "  TITLE: <одна строка>\n"
            "  BODY: <1-3 коротких абзаца>\n"
            "- Без списков и лишних заголовков.\n"
        ),
        default_len=(650, 1200),
    ),
    Platform(
        key="discord",
        title="Discord",
        hint="1-4 lines. Fast chat message.",
        rules_en=(
            "Platform rules: Discord.\n"
            "- Output MUST be ONLY the message text (no TITLE/BODY labels).\n"
            "- 1–4 lines max.\n"
            "- No bullet lists.\n"
        ),
        rules_ru=(
            "Правила: Discord.\n"
            "- Вывод ТОЛЬКО текст сообщения (без TITLE/BODY).\n"
            "- 1–4 строки максимум.\n"
            "- Без списков.\n"
        ),
        default_len=(180, 420),
    ),
]


# =========================================================
# OpenAI
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
        "Preserve tickers, $TICKER formatting, numbers, and line breaks."
    )
    user = f"Translate to {target_lang}. Text:\n\n{text}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# =========================================================
# Prompt builder
# =========================================================
ANGLE_BANK = [
    "macro risk premium and energy volatility",
    "defense spending and infrastructure procurement pressure",
    "why this shifts payback math and capex approvals",
    "how markets reprice small caps in volatility",
    "execution relevance vs narrative noise",
    "commodity -> electricity -> distributed infrastructure chain",
]

def to_dollar_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return "$"
    return t if t.startswith("$") else f"${t}"


def build_prompt(
    *,
    lang: str,
    platform: Platform,
    persona: Persona,
    ticker: str,
    company_name: str,
    company_one_liner: str,
    market: Dict[str, Any],
    article: Dict[str, str],
    rss_items: List[Dict[str, str]],
    selected_rss_idx: List[int],
    title_mode: str,
    force_numbers: bool,
    force_ticker_mention: bool,
    allow_hashtags: bool,
    twitter_trader_slang: bool,
    char_min: int,
    char_max: int,
    variation_seed: int,
) -> str:
    t = ticker.strip().upper()
    cash_ticker = to_dollar_ticker(t)

    angle = random.choice(ANGLE_BANK)

    # Quora title mode only
    title_directive = ""
    if platform.key == "quora":
        eff = random.choice(["exclude_company", "include_company"]) if title_mode == "mixed" else title_mode
        if eff == "exclude_company":
            title_directive = "In TITLE: do NOT mention the company name."
        else:
            title_directive = f"In TITLE: you MAY mention the company name ({company_name}) but do not force it."

    # Market anchors
    anchors = []
    if market.get("mcap_live") is not None:
        anchors.append(f"Market cap (live): {int(market['mcap_live']):,}")
    if market.get("price") is not None:
        anchors.append(f"Last price: {market['price']:.4f}")
    if market.get("chg_pct") is not None:
        anchors.append(f"Day change: {market['chg_pct']:.2f}%")
    if market.get("vol") is not None and market.get("vol10") is not None:
        anchors.append(f"Volume: {int(market['vol']):,} vs 10d avg {int(market['vol10']):,}")

    # Primary article
    art_title = (article.get("title") or "").strip()
    art_text = (article.get("text") or "").strip()
    art_url = (article.get("url") or "").strip()

    # RSS secondary
    rss_sel = [rss_items[i] for i in selected_rss_idx if 0 <= i < len(rss_items)]
    rss_block = []
    if rss_sel:
        rss_block.append("Secondary headlines:")
        for it in rss_sel:
            rss_block.append(f"- {it['title']}")
            if it.get("summary"):
                rss_block.append(f"  {truncate(strip_html(it['summary']), 220)}")

    # Macro anchors fallback
    macro_block = "\n".join(MACRO_ANCHORS_EN if lang == "EN" else MACRO_ANCHORS_RU)

    # Platform extras
    hashtag_rule = ""
    if platform.key == "twitter":
        hashtag_rule = "Hashtags: allowed (max 1)." if allow_hashtags else "Hashtags: do NOT use hashtags."
    else:
        hashtag_rule = "Hashtags: do NOT use hashtags."

    # Ticker mention rule
    if platform.key == "twitter":
        ticker_rule = f"Mention ticker exactly once as {cash_ticker}."
    else:
        ticker_rule = f"Ticker must appear at least once: {t}." if force_ticker_mention else f"Prefer mentioning ticker {t} if it fits."

    # Numbers rule
    if force_numbers:
        numbers_rule = "Include at least 3 numeric anchors (%, $, dates, counts). Use macro anchors if needed."
    else:
        numbers_rule = "Use numbers if they add value."

    # Twitter slang add-on only if enabled
    slang_addon = ""
    if platform.key == "twitter" and twitter_trader_slang:
        slang_addon = TWITTER_SLANG_GUIDE_EN if lang == "EN" else TWITTER_SLANG_GUIDE_RU

    # Core rules per language
    if lang == "EN":
        core = CORE_RULES_EN.format(banned=", ".join(BANNED_PHRASES))
        platform_rules = platform.rules_en
        persona_sys = persona.system_en
    else:
        core = CORE_RULES_RU.format(banned=", ".join(BANNED_PHRASES))
        platform_rules = platform.rules_ru
        persona_sys = persona.system_ru

    # Input pack
    input_parts = [
        f"Ticker: {t} (Twitter format: {cash_ticker})",
        f"Company name: {company_name}",
        f"Company one-liner (use only if relevant): {company_one_liner}".strip(),
        ("Data anchors (use 1-2 max):\n" + "\n".join(anchors)) if anchors else "Data anchors: not available.",
        f"Macro anchors (use if needed):\n{macro_block}",
        f"Primary article URL: {art_url}" if art_url else "",
        f"Primary article title: {art_title}" if art_title else "",
        f"Primary article text:\n{truncate(art_text, 3400)}" if art_text else "",
        "\n".join(rss_block) if rss_block else "",
    ]

    hard = f"""
Hard constraints:
- Character target: {char_min}-{char_max}.
- Angle: {angle}.
- Seed: {variation_seed}.
- {ticker_rule}
- {numbers_rule}
- {hashtag_rule}
- {title_directive}
- No bullet lists. No numbering.
""".strip()

    return "\n\n".join(
        [x for x in [persona_sys, core, platform_rules, slang_addon, "\n".join([p for p in input_parts if p.strip()]), hard] if x and x.strip()]
    )


def postprocess_output(
    raw: str,
    *,
    platform: Platform,
    ticker: str,
    force_numbers: bool,
    force_ticker_mention: bool,
) -> str:
    s = clean_no_long_dashes(raw)

    # remove banned phrases softly
    for ph in BANNED_PHRASES:
        s = re.sub(re.escape(ph), "", s, flags=re.IGNORECASE)

    t = ticker.strip().upper()
    cash_ticker = to_dollar_ticker(t)

    if platform.key == "quora":
        s = ensure_title_body(s)

        # enforce ticker in BODY
        if "BODY:" in s:
            pre, body = s.split("BODY:", 1)
            body = body.strip()
            if force_ticker_mention and t not in body:
                body = f"{body}\n\n{t}"
            if force_numbers and count_number_anchors(body) < 3:
                body += "\n\n20% oil flow. $850B defense spend. 10-25% energy swings."
            # strip bullets if any
            body = re.sub(r"(?m)^\s*[-•]\s+", "", body)
            s = pre.strip() + "\n\nBODY: " + body

        return s.strip()

    # Non-quora: strip TITLE/BODY if model produced them
    s = re.sub(r"(?is)\bTITLE:\s*.*?\n\nBODY:\s*", "", s).strip()
    s = re.sub(r"(?im)^\s*(TITLE|BODY)\s*:\s*", "", s).strip()
    s = re.sub(r"(?m)^\s*[-•]\s+", "", s).strip()

    # enforce ticker mention
    if platform.key == "twitter":
        # ensure $TICKER exactly once
        s = re.sub(re.escape(cash_ticker), "", s, flags=re.IGNORECASE).strip()
        # remove raw ticker occurrences to reduce duplicates
        s = re.sub(rf"\b{re.escape(t)}\b", "", s, flags=re.IGNORECASE).strip()
        # append once
        s = (s + f" {cash_ticker}").strip()
    else:
        if force_ticker_mention and t not in s:
            s = (s + f" {t}").strip()

    # enforce numbers minimum
    if force_numbers and count_number_anchors(s) < 3:
        s += " (20% oil flow, $850B defense spend, 10-25% energy swings)"

    # hard trim for Twitter
    if platform.key == "twitter" and len(s) > 280:
        s = s[:277] + "..."

    return s.strip()


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Post Studio", page_icon="⚡", layout="wide")
st.title("⚡ Post Studio: Persona + Platform + Trader-Slang Twitter")

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
  min-height: 92px;
}
.card.active {
  border: 2px solid var(--card-border-active);
}
.card .top{
  display:flex;
  align-items:center;
  gap:10px;
  margin-bottom:6px;
}
.card .icon{
  width:34px;height:34px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  font-size:18px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
}
.card .title{ font-weight:800; font-size:14px; line-height:1.1;}
.card .sub{ color: var(--text-dim); font-size:12px; margin-top:2px;}
.smallhint{ color: rgba(255,255,255,0.65); font-size:12px; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("OpenAI")
    api_key_ui = st.text_input("API Key (optional if Secrets has it)", value="", type="password")
    api_key = resolve_api_key(api_key_ui)
    model = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Creativity", 0.55, 1.20, 1.00, 0.05)

    st.divider()
    st.header("Output")
    out_lang = st.selectbox("Output languages", ["EN + RU (translation)", "EN only", "RU only"], index=0)
    n_variations = st.number_input("Variations", min_value=1, max_value=20, value=3, step=1)

    st.divider()
    st.header("Hard constraints")
    force_ticker_mention = st.toggle("Ticker must be mentioned", value=True)
    force_numbers = st.toggle("Force numbers (>=3 anchors)", value=True)
    title_mode = st.selectbox("Title mode (Quora)", ["mixed", "exclude_company", "include_company"], index=0)

    st.divider()
    st.header("Twitter mode")
    twitter_trader_slang = st.toggle("Twitter: bullish trader slang", value=True)
    allow_hashtags = st.toggle("Allow 1 hashtag (Twitter only)", value=False)

# Persona selection
st.subheader("Select Persona")
if "persona_key" not in st.session_state:
    st.session_state["persona_key"] = "algo"

cols = st.columns(4)
for i, p in enumerate(PERSONAS):
    with cols[i % 4]:
        if st.button(f"{p.icon} {p.title}", key=f"persona_btn_{p.key}", use_container_width=True):
            st.session_state["persona_key"] = p.key
        active = (st.session_state["persona_key"] == p.key)
        card_class = "card active" if active else "card"
        st.markdown(
            f"""
            <div class="{card_class}">
              <div class="top">
                <div class="icon">{p.icon}</div>
                <div>
                  <div class="title">{p.title}</div>
                  <div class="sub">{p.subtitle}</div>
                </div>
              </div>
              <div class="smallhint">{p.system_en.splitlines()[0][:70]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

persona = next(x for x in PERSONAS if x.key == st.session_state["persona_key"])

st.divider()

# Platform selection
st.subheader("Select Platform")
platform_titles = [pl.title for pl in PLATFORMS]
platform_map = {pl.title: pl for pl in PLATFORMS}
chosen = st.selectbox("Platform", platform_titles, index=0)  # default Twitter
platform = platform_map[chosen]
st.caption(platform.hint)

# Length
if "len_override" not in st.session_state:
    st.session_state["len_override"] = False

cA, cB = st.columns([0.35, 0.65])
with cA:
    st.session_state["len_override"] = st.toggle("Manual length", value=st.session_state["len_override"])
with cB:
    if st.session_state["len_override"]:
        char_min, char_max = st.slider("Length (characters)", 120, 2200, (platform.default_len[0], platform.default_len[1]), step=10)
    else:
        char_min, char_max = platform.default_len
        st.markdown(f"Length: **{char_min}-{char_max} chars** (platform default)")

st.divider()

# Ticker / market / RSS
st.subheader("Ticker & Market")
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
        value="Distributed energy infrastructure: microgrids, storage, resilience, localized power optimization.",
        height=90,
    )

st.divider()

# News input
st.subheader("News Context")
tab_url, tab_rss = st.tabs(["Paste URL (recommended)", "RSS (optional)"])

if "article" not in st.session_state:
    st.session_state["article"] = {"url": "", "title": "", "text": ""}

with tab_url:
    url_in = st.text_input("Paste news URL", value="")
    if st.button("Fetch Article", type="primary"):
        if not url_in.strip():
            st.warning("Paste a URL first.")
        else:
            try:
                st.session_state["article"] = fetch_article(url_in.strip())
                st.success("Fetched.")
            except Exception as e:
                st.error(f"Fetch error: {e}")

    art = st.session_state["article"]
    if art.get("title") or art.get("text"):
        st.text_input("Extracted title", value=art.get("title", ""), disabled=True)
        st.text_area("Extracted text", value=art.get("text", ""), height=200)

with tab_rss:
    if not rss:
        st.info("Click 'Fetch Market + RSS' first or just use URL mode.")
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

# Generate
st.subheader("Generate")

def generate_posts() -> List[Dict[str, Any]]:
    if not api_key:
        raise RuntimeError("No API key. Add OPENAI_API_KEY in Secrets or paste in sidebar.")

    art = st.session_state.get("article", {"url": "", "title": "", "text": ""})
    selected_idx = st.session_state.get("selected_rss", [])

    rows: List[Dict[str, Any]] = []
    for v in range(int(n_variations)):
        base_lang = "RU" if out_lang == "RU only" else "EN"

        prompt = build_prompt(
            lang=base_lang,
            platform=platform,
            persona=persona,
            ticker=ticker,
            company_name=company_name.strip() or "NextNRG",
            company_one_liner=company_one_liner,
            market=market,
            article=art,
            rss_items=rss,
            selected_rss_idx=selected_idx[:3] if rss else [],
            title_mode=title_mode,
            force_numbers=force_numbers,
            force_ticker_mention=force_ticker_mention,
            allow_hashtags=allow_hashtags,
            twitter_trader_slang=twitter_trader_slang,
            char_min=char_min,
            char_max=char_max,
            variation_seed=v,
        )

        # Compose system message
        if base_lang == "EN":
            system = persona.system_en + "\n\n" + CORE_RULES_EN.format(banned=", ".join(BANNED_PHRASES)) + "\n\n" + platform.rules_en
            if platform.key == "twitter" and twitter_trader_slang:
                system += "\n\n" + TWITTER_SLANG_GUIDE_EN
        else:
            system = persona.system_ru + "\n\n" + CORE_RULES_RU.format(banned=", ".join(BANNED_PHRASES)) + "\n\n" + platform.rules_ru
            if platform.key == "twitter" and twitter_trader_slang:
                system += "\n\n" + TWITTER_SLANG_GUIDE_RU

        raw = call_gpt(api_key, model, system, prompt, temperature=temperature)
        out = postprocess_output(raw, platform=platform, ticker=ticker, force_numbers=force_numbers, force_ticker_mention=force_ticker_mention)

        if out_lang == "EN only":
            if base_lang != "EN":
                out = translate_text(api_key, model, out, "English")
                out = postprocess_output(out, platform=platform, ticker=ticker, force_numbers=force_numbers, force_ticker_mention=force_ticker_mention)
            rows.append({"variation": v + 1, "lang": "EN", "platform": platform.title, "persona": persona.title, "ticker": ticker.strip().upper(), "text": out})

        elif out_lang == "RU only":
            if base_lang != "RU":
                out = translate_text(api_key, model, out, "Russian")
                out = postprocess_output(out, platform=platform, ticker=ticker, force_numbers=force_numbers, force_ticker_mention=force_ticker_mention)
            rows.append({"variation": v + 1, "lang": "RU", "platform": platform.title, "persona": persona.title, "ticker": ticker.strip().upper(), "text": out})

        else:
            # EN + RU
            if base_lang != "EN":
                en_text = translate_text(api_key, model, out, "English")
            else:
                en_text = out
            ru_text = translate_text(api_key, model, en_text, "Russian")

            en_text = postprocess_output(en_text, platform=platform, ticker=ticker, force_numbers=force_numbers, force_ticker_mention=force_ticker_mention)
            ru_text = postprocess_output(ru_text, platform=platform, ticker=ticker, force_numbers=force_numbers, force_ticker_mention=force_ticker_mention)

            rows.append({"variation": v + 1, "lang": "EN", "platform": platform.title, "persona": persona.title, "ticker": ticker.strip().upper(), "text": en_text})
            rows.append({"variation": v + 1, "lang": "RU", "platform": platform.title, "persona": persona.title, "ticker": ticker.strip().upper(), "text": ru_text})

    return rows


if st.button("Generate Posts", type="primary"):
    try:
        rows = generate_posts()
        st.session_state["generated_rows"] = rows
        st.success(f"Generated {len(rows)} outputs.")
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
                st.text_area(f"EN v{v}", value=en, height=320)
            with R:
                st.markdown("**Русский (перевод)**")
                st.text_area(f"RU v{v}", value=ru, height=320)
        elif out_lang == "EN only":
            en = sub[sub["lang"] == "EN"]["text"].values[0]
            st.text_area(f"EN v{v}", value=en, height=260)
        else:
            ru = sub[sub["lang"] == "RU"]["text"].values[0]
            st.text_area(f"RU v{v}", value=ru, height=260)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="generated_posts.csv",
        mime="text/csv",
    )
