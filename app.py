from pathlib import Path

import streamlit as st
from groq import Groq
from rag import Retriever
from core import (
    MODEL,
    EMERGENCY_MSG,
    is_emergency,
    build_system_prompt,
    smalltalk_reply,
)

# set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="Health Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
DEBUG = bool(st.secrets.get("DEBUG", False))  # DEBUG = true in secrets while testing only

FRONTEND = Path(__file__).parent / "frontend"


# ---------- Front-end helpers ----------
def load_css(filename="style.css"):
    css = (FRONTEND / filename).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_html(filename, **values):
    html = (FRONTEND / filename).read_text(encoding="utf-8")
    for key, value in values.items():
        html = html.replace("{{" + key + "}}", value)
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_retriever():
    return Retriever()


retriever = get_retriever()

EXAMPLES = [
    "What are the symptoms of dengue?",
    "How can I lower my blood pressure naturally?",
    "What foods help with iron deficiency?",
    "When should I see a doctor for a fever?",
]

load_css()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Read the input FIRST (the chat box is always pinned to the bottom of the page),
# so we know whether a conversation has started before drawing anything else.
pending = st.session_state.pop("pending", None)
prompt = pending or st.chat_input("Ask a health question...")
chatting = bool(st.session_state.messages) or bool(prompt)

if chatting:
    # During a conversation, the only extra control is New chat
    if st.button("➕ New chat", key="new_chat"):
        st.session_state.messages = []
        st.rerun()
else:
    # Start screen: banner and example questions
    render_html(
        "hero.html",
        title="Health Assistant",
        subtitle="Ask a health question and get clear, simple answers.",
    )
    st.write("**Try one of these:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

# ---------- Chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🩺"):
        st.markdown(m["content"])

# ---------- Handle input ----------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🩺"):
        placeholder = st.empty()
        quick = smalltalk_reply(prompt)  # pure greeting, thanks or goodbye

        if quick:
            text = quick
            placeholder.markdown(text)
        else:
            text = EMERGENCY_MSG if is_emergency(prompt) else ""
            results = retriever.search(prompt)
            system_prompt = build_system_prompt(results)

            try:
                with st.spinner("Thinking..."):
                    stream = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": system_prompt}]
                        + st.session_state.messages[-10:],
                        stream=True,
                    )
                    for chunk in stream:
                        text += chunk.choices[0].delta.content or ""
                        placeholder.markdown(text)

                if results:
                    text += "\n\n**Sources:** " + ", ".join(
                        sorted({r["source"] for r in results})
                    )
                    placeholder.markdown(text)
            except Exception as e:
                if DEBUG:
                    text += f"Error: {e}"
                else:
                    text += "Sorry, I'm having trouble right now. Please try again in a minute."
                placeholder.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": text})