import streamlit as st
from groq import Groq

MODEL = "openai/gpt-oss-120b"  # model names change; check Groq's docs if you get a "model not found" error
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM}] + st.session_state.messages[-10:],
                stream=True,
            )
            for chunk in stream:
                text += chunk.choices[0].delta.content or ""
                placeholder.markdown(text)
        except Exception as e:
            text += f"Error: {e}"
            placeholder.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": text})