"""Streamlit frontend for the Clinical RAG Assistant.

Talks to the FastAPI backend over HTTP (decoupled — the UI holds no ML logic).
Layout:
  - Sidebar: note selector (dropdown) + light/dark toggle + scope control.
    The dropdown is the safety mechanism: you must pick ONE patient's note, so
    answers can't blend patients. "Search all notes" is an explicit opt-in.
  - Tabs: Ask (chat + citations) | Summary | Extract (entity tables)

Run:  streamlit run frontend/streamlit_app.py
Requires the API running:  uvicorn app.api:app --port 8000
"""
from __future__ import annotations

import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Clinical RAG Assistant", page_icon="🩺", layout="wide")


# ── theme state ──
if "dark" not in st.session_state:
    st.session_state.dark = False


def theme_colors(dark: bool) -> dict:
    if dark:
        return dict(
            bg="#131c26", card="#1c2836", text="#eef4f8", muted="#a7b4c0",
            border="#2c3b4b", accent1="#5ec9c1", accent2="#7fb4e8",
            hero_text="#0f2130",  # dark text on the light gradient
            pill_bg="#24424a", pill_text="#a7e8e0", shadow="rgba(0,0,0,0.35)",
        )
    return dict(
        bg="#f6fafd", card="#ffffff", text="#1a2e3d", muted="#5b7285",
        border="#e4eef4", accent1="#8fd8d2", accent2="#a9cdf0",
        hero_text="#12405a",  # dark text on the light gradient
        pill_bg="#e6f6f4", pill_text="#2a7d78", shadow="rgba(15,60,90,0.07)",
    )


def inject_css(c: dict):
    st.markdown(
        f"""
        <style>
        /* ── global text: make everything readable in BOTH themes ── */
        .stApp {{ background: {c['bg']}; color: {c['text']}; }}
        .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
        .stMarkdown, [data-testid="stWidgetLabel"] p,
        .stTabs [data-baseweb="tab"], h1, h2, h3, h4, h5, h6 {{ color: {c['text']}; }}
        header[data-testid="stHeader"] {{ background: {c['bg']}; }}
        section[data-testid="stSidebar"] {{ background: {c['card']}; border-right: 1px solid {c['border']}; }}
        section[data-testid="stSidebar"] * {{ color: {c['text']}; }}
        /* text input: readable field in both themes */
        .stTextInput input {{ color: {c['text']}; background: {c['bg']}; }}
        .stTextInput input::placeholder {{ color: {c['muted']}; }}
        /* selectbox (note dropdown): dark field in dark mode, readable text */
        div[data-baseweb="select"] > div {{
            background: {c['bg']} !important; border-color: {c['border']} !important;
        }}
        div[data-baseweb="select"] * {{ color: {c['text']} !important; }}
        /* dropdown popover menu */
        ul[role="listbox"], div[data-baseweb="popover"] {{ background: {c['card']} !important; }}
        ul[role="listbox"] * {{ color: {c['text']} !important; }}
        /* compact, quiet theme-toggle: the LAST horizontal block on the page (bottom) */
        div[data-testid="stHorizontalBlock"]:last-of-type .stButton>button {{
            background: transparent !important; color: {c['muted']} !important;
            border: 1px solid {c['border']} !important; border-radius: 8px;
            padding: 3px 10px !important; font-size: 0.8rem !important; font-weight: 500 !important;
            box-shadow: none !important; min-height: 0 !important;
        }}
        div[data-testid="stHorizontalBlock"]:last-of-type .stButton>button * {{ color: {c['muted']} !important; }}
        div[data-testid="stHorizontalBlock"]:last-of-type .stButton>button:hover {{
            border-color: {c['accent1']} !important; color: {c['text']} !important;
        }}
        /* ── light gradient hero (uses DARK text) ── */
        .hero {{
            background: linear-gradient(120deg, {c['accent1']} 0%, {c['accent2']} 100%);
            padding: 26px 30px; border-radius: 18px; margin-bottom: 22px;
            box-shadow: 0 6px 18px {c['shadow']};
        }}
        .hero h1 {{ color: {c['hero_text']} !important; margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing:-0.5px; }}
        .hero p {{ color: {c['hero_text']} !important; opacity:0.85; margin: 6px 0 0; font-size: 1.02rem; }}
        /* cards */
        .card {{
            background: {c['card']}; border: 1px solid {c['border']};
            border-radius: 14px; padding: 18px 20px; margin: 10px 0;
            box-shadow: 0 3px 12px {c['shadow']}; color: {c['text']} !important;
        }}
        .card p, .card li, .card span {{ color: {c['text']} !important; }}
        .pill {{
            display:inline-block; background:{c['pill_bg']}; color:{c['pill_text']} !important;
            padding:3px 12px; border-radius:999px; font-size:0.8rem; font-weight:600;
            margin-right:6px;
        }}
        .scope-chip {{
            display:inline-block; background:linear-gradient(120deg,{c['accent1']},{c['accent2']});
            color:{c['hero_text']} !important; padding:4px 14px; border-radius:999px;
            font-size:0.85rem; font-weight:700;
        }}
        .stButton>button {{
            background: linear-gradient(120deg,{c['accent1']},{c['accent2']});
            color:{c['hero_text']} !important; border:none; border-radius:10px; padding:8px 22px;
            font-weight:700; box-shadow:0 3px 10px {c['shadow']};
        }}
        .stButton>button * {{ color:{c['hero_text']} !important; }}
        .stButton>button:hover {{ filter:brightness(1.04); }}
        .stTabs [data-baseweb="tab"] {{ font-weight:600; }}
        /* ── tables: readable in both themes (st.table + st.dataframe) ── */
        table {{ background:{c['card']} !important; color:{c['text']} !important; }}
        thead th {{ background:{c['pill_bg']} !important; color:{c['text']} !important; font-weight:700 !important; }}
        tbody td, tbody th {{ background:{c['card']} !important; color:{c['text']} !important; border-color:{c['border']} !important; }}
        [data-testid="stTable"] * , [data-testid="stDataFrame"] * {{ color:{c['text']} !important; }}
        /* ── sidebar app title (bigger) ── */
        .sidebar-title {{ font-size:1.5rem !important; font-weight:800; letter-spacing:-0.4px; margin-bottom:2px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── helpers ──
def api_get(path):
    return requests.get(f"{API}{path}", timeout=10).json()


def api_post(path, payload):
    r = requests.post(f"{API}{path}", json=payload, timeout=120)
    if not r.ok:
        try:
            return {"_error": r.json().get("detail", r.text)}
        except Exception:
            return {"_error": r.text}
    return r.json()


@st.cache_data(ttl=30)
def get_notes():
    try:
        return api_get("/notes")["notes"]
    except Exception:
        return []


# ── apply theme ──
c = theme_colors(st.session_state.dark)
inject_css(c)


# ── sidebar ──
with st.sidebar:
    st.markdown('<div class="sidebar-title">🩺 Clinical RAG</div>', unsafe_allow_html=True)
    st.markdown("**Patient Note**")
    notes = get_notes()
    if not notes:
        st.error("Backend unreachable.\nStart it with:\n\n`uvicorn app.api:app --port 8000`")
        st.stop()

    labels = {f"{n['number']}. {n['title']}": n["note_id"] for n in notes}
    choice = st.selectbox("Select a note (one patient at a time):", list(labels.keys()))
    selected_note = labels[choice]

    st.divider()
    search_all = st.checkbox(
        "🔎 Search across ALL notes",
        value=False,
        help="Off = answers come only from the selected note (safe). "
             "On = searches every note — answers may span multiple patients.",
    )
    if search_all:
        st.warning("Cross-patient search ON — answers may mix patients.")


# ── hero header ──
st.markdown(
    """
    <div class="hero">
      <h1>🩺 Clinical RAG Assistant</h1>
      <p>Ask · Summarize · Extract — grounded in your clinical notes &nbsp;·&nbsp; <b>by Anagha</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── main tabs ──
tab_ask, tab_summary, tab_extract = st.tabs(["💬  Ask", "📋  Summary", "🧬  Extract"])

# ---- ASK ----
with tab_ask:
    scope_label = "🌐 ALL notes (cross-patient)" if search_all else f"📄 {choice}"
    st.markdown(f'<span class="scope-chip">Scope: {scope_label}</span>', unsafe_allow_html=True)
    st.write("")
    q = st.text_input("Ask a question about this note:",
                      placeholder="e.g. What medications is the patient on?")
    if st.button("Ask", type="primary") and q:
        with st.spinner("Retrieving + answering locally…"):
            res = api_post("/ask", {
                "question": q,
                "note_id": None if search_all else selected_note,
                "search_all": search_all,
            })
        if "_error" in res:
            st.error(res["_error"])
        else:
            st.markdown("### Answer")
            st.markdown(f'<div class="card">{res["answer"]}</div>', unsafe_allow_html=True)
            cites = res.get("citations", [])
            if cites:
                st.markdown("### 🔗 Sources")
                for c2 in cites:
                    with st.expander(
                        f"[{c2['marker']}]  {c2['note_title']} — {c2['section']}  "
                        f"· chars {c2['char_start']}–{c2['char_end']}"
                    ):
                        st.write(c2["snippet"])
            else:
                st.info("No sources cited — the answer was not grounded in the note "
                        "(the system refusing to fabricate).")
            if res.get("dropped_markers"):
                st.warning(f"Rejected fabricated citations: {res['dropped_markers']}")

# ---- SUMMARY ----
with tab_summary:
    st.markdown(f'<span class="scope-chip">📄 {choice}</span>', unsafe_allow_html=True)
    st.write("")
    if st.button("Generate summary", type="primary"):
        with st.spinner("Summarizing locally…"):
            res = api_post("/summarize", {"note_id": selected_note})
        if "_error" in res:
            st.error(res["_error"])
        else:
            st.markdown(f'<div class="card">{res["summary"]}</div>', unsafe_allow_html=True)

# ---- EXTRACT ----
with tab_extract:
    st.markdown(f'<span class="scope-chip">📄 {choice}</span>', unsafe_allow_html=True)
    st.caption("✓ = source verified in the note · ⚠ = model paraphrased, unverified")
    if st.button("Extract entities", type="primary"):
        with st.spinner("Extracting structured entities locally…"):
            res = api_post("/extract-entities", {"note_id": selected_note})
        if "_error" in res:
            st.error(res["_error"])
        else:
            counts = res.get("counts", {})
            chips = " ".join(
                f'<span class="pill">{k}: {v}</span>' for k, v in counts.items()
            )
            st.markdown(chips, unsafe_allow_html=True)
            st.write("")
            cats = [
                ("diagnoses", "🩺 Diagnoses"), ("medications", "💊 Medications"),
                ("allergies", "⚠️ Allergies"), ("procedures", "🔧 Procedures"),
                ("follow_ups", "📅 Follow-ups"),
            ]
            for key, title in cats:
                items = res.get(key, [])
                if not items:
                    continue
                st.markdown(f"#### {title}")
                rows = []
                for e in items:
                    row = dict(e["data"])
                    row["verified"] = "✓" if e["grounded"] else "⚠"
                    rows.append(row)
                st.table(rows)


# ── small theme toggle at the bottom-RIGHT of the page ──
st.write("")
st.divider()
_l, _r = st.columns([11, 1])
with _r:
    if st.button("🌙" if not st.session_state.dark else "☀️", help="Toggle light/dark"):
        st.session_state.dark = not st.session_state.dark
        st.rerun()
