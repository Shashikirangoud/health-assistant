from pathlib import Path

import streamlit as st
from groq import Groq
from rag import Retriever

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Health Assistant", page_icon="🩺", layout="centered")

MODEL = "openai/gpt-oss-120b"
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

# ---------- Content and settings ----------
SYSTEM = """You are a health information assistant.

Style rules (very important):
- Start directly with the answer. Do NOT open with sympathy or emotional phrases.
  Never say things like "I'm sorry you're dealing with that", "I'm sorry to hear that",
  "That sounds painful", or "I understand how you feel".
- Keep a neutral, factual, professional tone. No small talk and no filler openings.
- Treat questions like "what are the symptoms of X" as general information requests.
  Do not assume the user has the condition.

Content rules:
- Give clear, general health information in simple language.
- Never diagnose or prescribe medicines or doses.
- Suggest seeing a qualified doctor when appropriate.
- If key details are missing (age, duration, severity), ask one short follow-up question.
- If the user describes their own symptoms, list the possible general causes, what
  they can do at home, and which warning signs mean they should see a doctor.
- If you're unsure, say so."""

EMERGENCY_WORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "stroke", "unconscious", "severe bleeding", "suicide", "kill myself",
    "overdose", "seizure", "heart attack",
]
EMERGENCY_MSG = (
    "⚠️ **This may be an emergency.** Please call **112** (India) or go to the "
    "nearest hospital immediately. Don't wait for an online answer.\n\n"
)

EXAMPLES = [
    "What are the symptoms of dengue?",
    "How can I lower my blood pressure naturally?",
    "What foods help with iron deficiency?",
    "When should I see a doctor for a fever?",
]

# ---------- Apply front end ----------
load_css()

with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    render_html("sidebar.html")

if "messages" not in st.session_state:
    st.session_state.messages = []

render_html(
    "hero.html",
    title="Health Assistant",
    subtitle="Ask a health question and get clear, simple answers.",
)

if not st.session_state.messages:
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
pending = st.session_state.pop("pending", None)
prompt = pending or st.chat_input("Ask a health question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🩺"):
        text = EMERGENCY_MSG if any(w in prompt.lower() for w in EMERGENCY_WORDS) else ""
        placeholder = st.empty()

        results = retriever.search(prompt)
        context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
        system_prompt = SYSTEM
        if context:
            system_prompt += (
                "\n\nUse the reference material below when relevant, and mention that "
                "it comes from the listed source. If it doesn't cover the question, "
                "say so and answer cautiously from general knowledge.\n\n" + context
            )

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
                text += "\n\n**Sources:** " + ", ".join(sorted({r["source"] for r in results}))
                placeholder.markdown(text)
        except Exception as e:
            if DEBUG:
                text += f"Error: {e}"
            else:
                text += "Sorry, I'm having trouble right now. Please try again in a minute."
            placeholder.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": text})
    import re

    SYMPATHY_PATTERN = re.compile(
    r"^\s*(i['’]m|i am) (so )?sorry[^.!\n]*[.!]\s*", re.IGNORECASE
    )


def clean_reply(reply):
    return SYMPATHY_PATTERN.sub("", reply, count=1)