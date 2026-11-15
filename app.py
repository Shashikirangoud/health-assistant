import streamlit as st
from groq import Groq
from rag import Retriever

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_resource
def get_retriever():
    return Retriever()


retriever = get_retriever()

SYSTEM = """You are a health information assistant.
- Give clear, general health information in simple language.
- Never diagnose or prescribe medicines or doses.
- Suggest seeing a qualified doctor when appropriate.
- Ask a short follow-up question if key details (age, duration, severity) are missing.
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

st.set_page_config(page_title="Health Assistant", page_icon="🩺")
st.title("🩺 Health Assistant")
st.caption("General information only, not a substitute for a doctor. "
           "Please don't enter names, phone numbers, or other personal details.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a health question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        text = EMERGENCY_MSG if any(w in prompt.lower() for w in EMERGENCY_WORDS) else ""
        placeholder = st.empty()

        # ---- RAG part: this is the block that goes before the Groq call ----
        results = retriever.search(prompt)
        context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
        system_prompt = SYSTEM
        if context:
            system_prompt += (
                "\n\nUse the reference material below when relevant, and mention that "
                "it comes from the listed source. If it doesn't cover the question, "
                "say so and answer cautiously from general knowledge.\n\n" + context
            )
        # ---------------------------------------------------------------------

        try:
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
            text += f"Error: {e}"
            placeholder.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": text})