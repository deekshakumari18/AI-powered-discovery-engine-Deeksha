"""Phase 5 UI.

Tiny Streamlit chat. Shows the retrieval answer as returned.
Does not add analysis and does not keep personal data.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from retrieval import answer, guard
import retrieval

WELCOME = "AI-Powered Discovery Engine"
NOTE = "Facts only. No investment advice."
UNAVAILABLE = "The model is unavailable right now. Try again."
EXAMPLES = [
    "Why do first-time investors start a SIP, and what do they expect?",
    "What leads people to pause, reduce, or stop a SIP?",
    "What confuses or worries users inside investing apps after they start?",
]

st.set_page_config(page_title="SIP Discovery", layout="centered")
st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stAppViewContainer"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    section.main {
        background: #f4f7fb !important;
        color: #1c2b3a;
    }
    .block-container {
        padding-top: 3.2rem;
        max-width: 760px;
    }
    h1, [data-testid="stHeading"] h1 {
        color: #003087 !important;
        font-weight: 650 !important;
        letter-spacing: -0.03em;
        line-height: 1.2 !important;
        margin-bottom: 0.4rem !important;
    }
    .note {
        color: #4d6280;
        font-size: 0.98rem;
        margin: 0 0 1.4rem 0;
    }
    .prompt-label {
        color: #1c2b3a;
        font-weight: 600;
        margin: 0 0 0.6rem 0;
    }
    [data-testid="stChatMessageAvatar"] { display: none; }
    div[data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e4ebf3;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 48, 135, 0.05);
        padding: 0.35rem 0.2rem;
    }
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #d5e0ec !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 24px rgba(0, 48, 135, 0.06);
    }
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #1c2b3a !important;
        caret-color: #003087;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #7b8da3 !important;
    }
    div.stButton > button {
        background: #ffffff !important;
        color: #1c2b3a !important;
        border: 1px solid #d5e0ec !important;
        border-radius: 14px !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.85rem 1rem !important;
        box-shadow: 0 4px 14px rgba(0, 48, 135, 0.04);
    }
    div.stButton > button p {
        color: #1c2b3a !important;
        text-align: left !important;
        font-size: 0.98rem !important;
    }
    div.stButton > button:hover {
        border-color: #003087 !important;
        background: #f7faff !important;
    }
    .st-key-refresh button {
        background: #003087 !important;
        border: none !important;
        width: auto !important;
    }
    .st-key-refresh button p { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _show(text: str) -> None:
    linked = []
    for line in text.splitlines():
        linked.append(re.sub(r"(https?://[^\s)]+)", r"[\1](\1)", line))
    st.markdown("\n\n".join(linked))


def _ask(question: str) -> None:
    blocked = guard(question)
    if blocked and blocked.startswith("We do not accept personal data"):
        st.session_state.history.append({"role": "assistant", "content": blocked})
        return
    st.session_state.history.append({"role": "user", "content": question})
    try:
        with st.spinner("Looking through the documents..."):
            reply = answer(question, st.session_state.history[:-1])
            st.session_state.last_hits = list(retrieval.last_hits)
    except Exception:
        reply = UNAVAILABLE
    st.session_state.history.append({"role": "assistant", "content": reply})


def _refresh() -> None:
    history = st.session_state.history
    if (
        len(history) >= 2
        and history[-1]["content"] == UNAVAILABLE
        and history[-2]["role"] == "user"
    ):
        question = history[-2]["content"]
        st.session_state.history = history[:-2]
        _ask(question)
        return
    st.session_state.history = []


if "history" not in st.session_state:
    st.session_state.history = []

st.title(WELCOME)
st.markdown(f'<p class="note">{NOTE}</p>', unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown('<p class="prompt-label">Try one of these:</p>', unsafe_allow_html=True)
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            _ask(example)
            st.rerun()

typed = st.chat_input("Ask a question")

if st.session_state.history and st.session_state.history[-1]["content"] == UNAVAILABLE:
    if st.button("Refresh", key="refresh"):
        _refresh()
        st.rerun()

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        _show(message["content"])

if typed:
    _ask(typed)
    st.rerun()
