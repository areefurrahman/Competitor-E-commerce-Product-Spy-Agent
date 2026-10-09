"""
E-commerce Competitor & Product Spy Agent
-----------------------------------------
Single CrewAI agent + free DuckDuckGo search + Groq (openai/gpt-oss-120b) + Streamlit.
"""

import os

# Turn off CrewAI telemetry/tracing BEFORE crewai is imported.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import re
import threading
import time
from datetime import datetime

import streamlit as st
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from ddgs import DDGS

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="E-com Spy Agent", page_icon="🕵️", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding:1.3rem 1.5rem;border-radius:14px;margin-bottom:1.2rem;
           background:linear-gradient(135deg,#0f172a,#1e293b);}
    .hero h1 {margin:0 0 .35rem 0;font-size:1.9rem;color:#f8fafc;}
    .hero p {margin:0;color:#cbd5e1;font-size:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# CrewAI strips ONE leading "openai/" routing prefix when custom_openai=True,
# so the model name Groq actually receives is "openai/gpt-oss-120b".
CREWAI_MODEL_STRING = "openai/openai/gpt-oss-120b"

MAX_SEARCHES = 6          # hard cap per run (keeps Groq token usage low)
RESULTS_PER_SEARCH = 5
SNIPPET_CHARS = 420
TAVILY_DEPTH = "basic"  # "advanced" gives richer text but costs 2 credits per search
# If one search engine blocks us (common on cloud servers), try the next one.
SEARCH_BACKENDS = ["brave", "duckduckgo", "yahoo", "startpage", "mojeek", "google"]
# NOTE: do not use "auto" or the old "wt-wt" region: both fail in ddgs 9.x.

REGIONS = {
    "Global": {
        "code": "us-en",
        "market": "the global online market (Amazon, eBay, AliExpress, Shopify brand stores)",
        "currency": "USD (or the currency shown by the source)",
    },
    "US / Amazon": {
        "code": "us-en",
        "market": "the United States market, mainly Amazon.com, Walmart, Target and Shopify brand stores",
        "currency": "USD",
    },
    "Pakistan / Local E-commerce": {
        "code": "pk-en",
        "market": "the Pakistan market, mainly Daraz.pk, PriceOye, Telemart and local Shopify/WooCommerce stores",
        "currency": "PKR",
    },
}


# --------------------------------------------------------------------------- #
# Search tool (DuckDuckGo, no API key)
# --------------------------------------------------------------------------- #
class SearchSession:
    """Keeps track of what the agent searched. Thread-safe."""

    def __init__(self, region_code: str, tavily_key: str = ""):
        self.region_code = region_code
        self.tavily_key = tavily_key
        self.queries: list[str] = []
        self.urls: dict[str, str] = {}  # url -> title (keeps order)
        self.errors: list[str] = []     # search problems, shown in the UI
        self.lock = threading.Lock()


PRICE_RE = re.compile(
    r"(?:US\$|\$|£|€|Rs\.?\s?|PKR\s?)\s?\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|PKR|GBP|EUR)\b"
)


def find_prices(text: str, limit: int = 6) -> list[str]:
    """Pull price-looking strings out of a snippet so the agent sees real numbers."""
    seen: list[str] = []
    for m in PRICE_RE.finditer(text or ""):
        value = " ".join(m.group(0).split())
        if value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return seen


def _tavily_search(query: str, api_key: str, max_results: int):
    """Optional backup (free tier, no card). Returns results in ddgs format."""
    import httpx

    resp = httpx.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "max_results": max_results, "search_depth": TAVILY_DEPTH},
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")}
        for r in resp.json().get("results", [])
    ]


def web_search_with_fallback(query: str, session: "SearchSession", max_results: int = RESULTS_PER_SEARCH):
    """Try search engines one by one. Returns (results or None, last_error)."""
    last_error = None

    # 1) Tavily first when a key is set (free engines are often blocked on cloud servers).
    if session.tavily_key:
        try:
            found = _tavily_search(query, session.tavily_key, max_results)
            if found:
                return found, None
            last_error = RuntimeError("empty result")
        except Exception as exc:
            last_error = exc
        with session.lock:
            session.errors.append(f"tavily: {type(last_error).__name__}: {str(last_error)[:110]}")

    # 2) Free engines (no key needed).
    regions = [session.region_code] + (["us-en"] if session.region_code != "us-en" else [])
    for region in regions:
        for backend in SEARCH_BACKENDS:
            try:
                found = DDGS().text(
                    query,
                    region=region,
                    safesearch="moderate",
                    max_results=max_results,
                    backend=backend,
                )
                if found:
                    return found, None
                last_error = RuntimeError("empty result")
            except Exception as exc:  # blocked, rate limited, parse error...
                last_error = exc
            with session.lock:
                session.errors.append(f"{backend}/{region}: {type(last_error).__name__}: {str(last_error)[:110]}")
            time.sleep(0.3)
    return None, last_error


def make_search_tool(session: SearchSession):
    @tool("DuckDuckGo Web Search")
    def duckduckgo_search(query: str) -> str:
        """Search the live web with DuckDuckGo. Input must be one short search
        query string, for example: 'minimalist leather wallet price amazon'.
        Returns titles, URLs and short snippets of the top results."""
        query = (query or "").strip()
        if not query:
            return "Empty query. Send a short search query."

        with session.lock:
            if len(session.queries) >= MAX_SEARCHES:
                return (
                    "SEARCH LIMIT REACHED. Do NOT search again. "
                    "Write the final four-section report now using what you already found."
                )
            session.queries.append(query)

        results, last_error = web_search_with_fallback(query, session)

        if results is None:
            return (
                f"Search failed ({type(last_error).__name__}). "
                "Try a different, simpler query, or continue with the data you already have."
            )
        if not results:
            return "No results. Try a different, simpler query."

        lines = []
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            url = (r.get("href") or "").strip()
            body = " ".join((r.get("body") or "").split())[:SNIPPET_CHARS]
            if url:
                with session.lock:
                    session.urls.setdefault(url, title or url)
            prices = find_prices(f"{title} {r.get('body') or ''}")
            price_line = f"\n   Prices seen: {', '.join(prices)}" if prices else ""
            lines.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {body}{price_line}")
        return "\n".join(lines)

    return duckduckgo_search


# --------------------------------------------------------------------------- #
# CrewAI: LLM, agent, task
# --------------------------------------------------------------------------- #
def build_llm(api_key: str) -> LLM:
    # Groq is OpenAI-compatible, so we use CrewAI's native OpenAI client.
    # This avoids the optional LiteLLM package completely.
    return LLM(
        model=CREWAI_MODEL_STRING,
        custom_openai=True,
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        temperature=0.3,
        max_tokens=6000,
    )


def build_task_description(product: str, region_name: str, competitors: str) -> str:
    region = REGIONS[region_name]
    targets = competitors.strip() or "None given. Find the main competitors yourself."
    return f"""Investigate this product or niche: "{product}"

Market focus: {region['market']}.
Prices should be shown in: {region['currency']}.
Competitors or URLs to target first: {targets}

How to work (use at most {MAX_SEARCHES} searches, short queries, one topic per query):
1. PRICES (2 searches): e.g. "{product} price", "best {product} under $100" and "{product} premium best". Look at the "Prices seen" lines.
2. COMPETITORS (1 search): "best {product} brands".
3. COMPLAINTS (2 searches): e.g. "{product} problems complaints reddit" and "{product} negative reviews wobble broke returned". Real customer words only.
4. If the user gave competitors, spend the last search on them by name.

Rules:
- If the niche is vague, add shopping words and pick the most likely meaning (say which one).
- Use ONLY facts found in search results. NEVER invent or "infer" prices, ranges, brands or quotes.
- Prices: use only numbers shown in "Prices seen" or in a snippet. If you have fewer than 3 real prices, fill only the tiers you can prove and write "Not found in search results" for the rest. Do NOT write any price range outside the table that is not backed by a real number.
- Complaints: only count something as a complaint if a snippet shows customers or reviewers saying it. An ad that says "easy assembly" is NOT a complaint. If evidence is thin, say "Weak evidence".
- Cite by site name (for example "Reddit", "Amazon listing"). Do NOT use citation markers like 【1†L1-L3】 and do not invent URLs.
- If Section 1 or 2 has little real data, say so at the top of that section, and keep Sections 3 and 4 clearly marked as "ideas based on limited data".
- Be specific and practical. Short sentences. Simple words."""


EXPECTED_OUTPUT = """A clean Markdown report with EXACTLY these four sections, in this order:

## 📊 Section 1: Market Pricing Benchmark
A table with columns: Tier | Price (approx.) | Example product / store | Source.
Rows: Low-end, Average market rate, Premium. Then one line on the price pattern.

## ⚠️ Section 2: Competitor Flaws & Customer Complaints
5-8 bullets. Each bullet: the complaint, which competitor/product it applies to, and how often it seems to repeat.

## 💡 Section 3: The Unfair Advantage
Feature gaps and 5-7 concrete actions to make the product better than the competitors (materials, features, bundle, warranty, packaging, service, price position).

## 🎯 Section 4: Winning Hooks & Ad Angles
Exactly 3 ad angles. For each: Hook line, Which competitor weakness it attacks, Short ad copy (2 lines).

End with a short '### Sources' list of the URLs you used and a one-line 'Confidence' note (High / Medium / Low) with the reason."""


def build_crew(llm: LLM, search_tool, product: str, region_name: str, competitors: str) -> Crew:
    agent = Agent(
        role="Senior E-commerce Intelligence & Market Spy",
        goal=(
            "Analyze live competitor products, scrape market pricing, identify customer "
            "pain points from reviews, and formulate winning marketing angles."
        ),
        backstory=(
            "An elite e-commerce brand consultant who specializes in spotting competitors' "
            "weaknesses, analyzing real pricing structures, and helping store owners "
            "dominate product categories."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        max_rpm=20,
    )
    task = Task(
        description=build_task_description(product, region_name, competitors),
        expected_output=EXPECTED_OUTPUT,
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def clean_key(raw: str, prefix: str) -> str:
    """Pull a clean ASCII key out of messy pasted text (quotes, spaces, extra lines).
    Returns "" if no valid-looking key is found."""
    match = re.search(re.escape(prefix) + r"[A-Za-z0-9_\-]{10,}", raw or "")
    return match.group(0) if match else ""


def get_secret_key() -> str:
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except Exception:  # no secrets file locally
        return ""


def friendly_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "401" in low or "invalid api key" in low or "authentication" in low:
        return "Groq rejected the API key. Check it at console.groq.com/keys and try again."
    if "429" in low or "rate limit" in low or "rate_limit" in low or "tokens per minute" in low:
        return (
            "Groq rate limit reached. Wait about a minute and try again. "
            "If it keeps happening, use a shorter niche or upgrade your Groq plan."
        )
    if "model" in low and ("not found" in low or "does not exist" in low or "decommission" in low):
        return "Groq says the model name is not available for your account. Check the model list at console.groq.com/docs/models."
    if "timeout" in low or "timed out" in low or "connection" in low:
        return "Network problem or timeout while talking to Groq or DuckDuckGo. Please try again."
    return f"Something went wrong: {text[:500]}"


def build_report_file(product: str, region_name: str, body: str, sources=None) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if sources:
        body += "\n\n### Pages the agent looked at\n" + "\n".join(f"- {t[:90]} - {u}" for u, t in sources)
    return (
        f"# E-commerce Spy Report: {product}\n\n"
        f"- Market: {region_name}\n- Generated: {stamp}\n\n---\n\n{body}\n\n---\n"
        "_Data comes from public web search snippets. Check prices and reviews "
        "on the live pages before making business decisions._\n"
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    secret_key = get_secret_key()
    if secret_key:
        api_key = clean_key(secret_key, "gsk_")
        if api_key:
            st.success("Groq API key loaded from secrets ✅")
        else:
            st.error("GROQ_API_KEY in secrets looks wrong. It must start with gsk_ and have no extra text.")
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Your key is used only for this session and is never saved.",
        )
        api_key = clean_key(api_key, "gsk_") or api_key.strip().encode("ascii", "ignore").decode()
        st.markdown("Get a free key at [console.groq.com](https://console.groq.com/keys).")

    region_name = st.selectbox("Market Region", list(REGIONS.keys()), index=0)

    try:
        raw_tavily = str(st.secrets.get("TAVILY_API_KEY", "") or "")
    except Exception:
        raw_tavily = ""
    if not raw_tavily.strip():
        raw_tavily = st.text_input(
            "Backup search key (optional)",
            type="password",
            placeholder="tvly-...",
            help="Used first when set. Free key at tavily.com.",
        )
    tavily_key = clean_key(raw_tavily, "tvly-")
    if raw_tavily.strip() and not tavily_key:
        st.error("Tavily key looks wrong. It must start with tvly- and contain no quotes or extra text.")
    elif tavily_key:
        st.caption("Backup search key ready ✅")
    st.caption("Model: `openai/gpt-oss-120b` on Groq")
    st.caption(f"Search: DuckDuckGo + backups (max {MAX_SEARCHES} searches per run)")

    if st.button("🧪 Test web search", use_container_width=True):
        test_session = SearchSession(REGIONS[region_name]["code"], tavily_key)
        with st.spinner("Testing search engines..."):
            res, err = web_search_with_fallback("wireless earbuds price", test_session, 3)
        if res:
            st.success(f"Search works ✅ ({len(res)} results)")
        else:
            st.error("Search is blocked or empty ❌")
        if test_session.errors:
            with st.expander("Engine details"):
                st.code("\n".join(test_session.errors))

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <h1>🕵️ E-commerce Competitor & Product Spy Agent</h1>
      <p>Type a product or niche. The agent searches the live web, compares price brackets,
      finds what customers hate about competitors, and gives you a plan to beat them.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

product = st.text_input(
    "Product / Niche",
    placeholder="e.g. Minimalist RFID Leather Wallet  or  Wireless ANC Earbuds",
    max_chars=150,
)
competitors = st.text_input(
    "Specific competitors or URLs to target (optional)",
    placeholder="e.g. Ridge Wallet, Bellroy, https://www.amazon.com/...",
    max_chars=300,
)

run_clicked = st.button("Spy on Competitors & Generate Report 🚀", type="primary", use_container_width=True)

if run_clicked:
    product_clean = product.strip()
    if not product_clean:
        st.warning("Please enter a product or niche first.")
    elif not api_key:
        st.error("Missing Groq API key. Add it in the sidebar or in Streamlit secrets.")
    else:
        if not api_key.startswith("gsk_"):
            st.warning("This key does not look like a Groq key (they start with `gsk_`). Trying anyway...")

        session = SearchSession(REGIONS[region_name]["code"], tavily_key)
        box = {"result": None, "error": None}
        started = time.time()

        try:
            crew = build_crew(
                build_llm(api_key), make_search_tool(session), product_clean, region_name, competitors
            )
        except Exception as exc:
            st.error(friendly_error(exc))
            crew = None

        if crew is not None:

            def worker():
                try:
                    box["result"] = re.sub(r"【[^】]*】", "", str(crew.kickoff().raw))
                except Exception as exc:  # shown to the user below
                    box["error"] = exc

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            with st.status("🕵️ Spy agent is working...", expanded=True) as status:
                live = st.empty()
                while thread.is_alive():
                    with session.lock:
                        queries = list(session.queries)
                        n_sources = len(session.urls)
                    if queries:
                        recent = "\n".join(f"- 🔎 {q}" for q in queries[-5:])
                        live.markdown(
                            f"**Searches so far: {len(queries)}/{MAX_SEARCHES}** · Sources found: {n_sources}\n\n{recent}"
                        )
                    else:
                        live.markdown("Warming up and planning the first search...")
                    time.sleep(0.7)
                thread.join()

                if box["error"] is not None:
                    status.update(label="❌ The run failed", state="error", expanded=True)
                elif not (box["result"] or "").strip():
                    status.update(label="❌ The agent returned an empty report", state="error")
                else:
                    status.update(label="✅ Report ready", state="complete", expanded=False)

            if box["error"] is not None:
                st.error(friendly_error(box["error"]))
            elif not (box["result"] or "").strip():
                st.error("The agent returned an empty report. Please try again with a clearer product name.")
            else:
                with session.lock:
                    sources = list(session.urls.items())
                    n_searches = len(session.queries)
                    search_errors = list(session.errors)
                st.session_state["report"] = {
                    "product": product_clean,
                    "region": region_name,
                    "body": box["result"],
                    "searches": n_searches,
                    "sources": sources,
                    "seconds": time.time() - started,
                    "errors": search_errors,
                }

# --------------------------------------------------------------------------- #
# Show the saved report (survives reruns such as clicking Download)
# --------------------------------------------------------------------------- #
report = st.session_state.get("report")
if report:
    st.divider()
    st.subheader(f"Report: {report['product']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Searches run", report["searches"])
    c2.metric("Unique sources", len(report["sources"]))
    c3.metric("Time taken", f"{report['seconds']:.0f}s")
    c4.metric("Market", report["region"].split(" /")[0])

    if not report["sources"]:
        st.warning(
            "⚠️ The web search returned **no pages**. Sections about prices and complaints "
            "are NOT real market data. Use the **🧪 Test web search** button in the sidebar. "
            "Try a more specific product name too."
        )
    if report.get("errors"):
        with st.expander("Search problems (for debugging)"):
            st.code("\n".join(report["errors"][-12:]))

    with st.container(border=True):
        st.markdown(report["body"])

    if report["sources"]:
        with st.expander(f"🔗 All pages the agent looked at ({len(report['sources'])})"):
            for url, title in report["sources"]:
                st.markdown(f"- [{title[:90]}]({url})")

    file_text = build_report_file(report["product"], report["region"], report["body"], report["sources"])
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in report["product"])[:40].strip("_") or "report"
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download as Markdown (.md)",
        data=file_text,
        file_name=f"spy_report_{safe_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    d2.download_button(
        "⬇️ Download as Text (.txt)",
        data=file_text,
        file_name=f"spy_report_{safe_name}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.caption("⚠️ Data comes from public search snippets. Always verify prices and reviews on the live pages.")
else:
    st.info("Enter a product above and press the big button. A full report usually takes 30-90 seconds.")
