import re

MODEL = "openai/gpt-oss-120b"

SYSTEM = """You are a health information assistant.

Style rules (very important):
- Start directly with the answer. Do NOT open with sympathy or emotional phrases.
  Never say things like "I'm sorry you're dealing with that", "I'm sorry to hear that",
  "That sounds painful", or "I understand how you feel".
- Keep a neutral, factual, professional tone. No small talk and no filler openings.
- Treat questions like "what are the symptoms of X" as general information requests.
  Do not assume the user has the condition.

Medicine rules (very important):
- NEVER state a numeric dose, strength, quantity or schedule for any medicine
  (no mg, ml, number of tablets, or "every X hours"), even for common over-the-counter
  medicines, even for adults, and even if the user insists.
- Instead, say the right amount depends on age, weight, health conditions and the
  specific product, and tell them to follow the product label or ask a doctor or
  pharmacist.
- You may still explain what a medicine is used for and general safety points.

Content rules:
- Give clear, general health information in simple language.
- Never diagnose. Suggest seeing a qualified doctor when appropriate.
- When the user asks whether they have a specific disease, whether a mole, lump or
  other finding is serious or cancer, or what a test reading means for them, say
  clearly that this cannot be determined online, give only brief general context,
  and end by recommending they see a doctor (a dermatologist for skin changes).
- When the user describes symptoms, always finish with a short "When to see a doctor"
  line covering which warning signs need urgent care and recommending a doctor visit
  if the symptoms persist.
- If key details are missing (age, duration, severity), ask one short follow-up question.
- If the user describes their own symptoms, list the possible general causes, what
  they can do at home, and which warning signs mean they should see a doctor.
- If you're unsure, say so."""

# ---------------------------------------------------------------------------
# Emergency detection. Text is lowercased first.
# Design: a miss is far worse than a false alarm, so clear danger signs ALWAYS fire.
# ---------------------------------------------------------------------------
ALWAYS_PATTERNS = [
    # chest
    r"chest (pain|pressure|tightness)",
    r"pain (in|around) (my |the |his |her |their )?chest",
    r"chest (feels?|is|feeling|has been)\s+(very |really |so |extremely )?(tight|heavy|crushing|squeezed)",
    # breathing ("cant" without apostrophe included)
    r"(can'?t|cannot|can not|unable to|couldn'?t|could not)\s+(even\s+)?breath",
    r"(difficulty|trouble|struggling|hard) (to |in |with )?breath",
    r"(not|isn't|stopped|stops) breathing",
    # stroke signs
    r"face drooping|drooping face|slurred speech",
    # consciousness
    r"\b(unconscious|unresponsive|won't wake up|not waking up)\b",
    # bleeding
    r"\b(severe|uncontrolled|profuse)\b[^.?!]{0,20}\bbleeding\b",
    r"\bbleeding\b[^.?!]{0,30}\b(won't|will not|not) stop",
    # poisoning
    r"\b(swallowed|drank|ate|ingested|consumed)\b[^.?!]{0,30}\b(poison|bleach|pesticide|phenyl|kerosene|acid)\b",
    # self-harm
    r"suicid",
    r"kill myself|end my (own )?life|take my (own )?life|want to die|better off dead|don't want to live",
    r"(hurt|harm|cut)(ting|ing)? myself|self[- ]harm",
    # Hinglish (romanized Hindi). Spellings vary, so extend this list from real messages.
    r"\b(seene|seena|sine|chhati|chaati|chati)\b[^.?!]{0,15}\b(dard|pain)\b",
    r"\b(saans|sans)\b[^.?!]{0,20}\b(nahi|nhi|nahin|band|ruk\w*|takleef|dikkat)\b",
    r"\bmarne\s+(ka|ki)\s+(mann?|man|dil|iccha|ichha)\b",
    r"\bmar\s+jaa?na\s+(chahta|chahti)\b",
    r"\bjeene\s+(ka|ki)\s+(mann?|man|iccha|ichha)\s+(nahi|nhi|nahin)\b",
    r"\b(khud\s+ko|apni\s+jaan)\s+(khatam|maar|end)\b",
    r"\bkhoon\b[^.?!]{0,30}\b(beh|bah|nikal|ruk)\w*",
]

# These words also appear in informational questions, so they can be suppressed.
CONTEXT_PATTERNS = [
    r"heart\s*at+a+c?k",  # heart attack, atack, attak
    r"\b(having|has|had|got|suffering|suspect|think)\b[^.?!]{0,30}\bstroke\b",
    r"\b(having|has|had|got)\b[^.?!]{0,20}\bseizures?\b(?!\s+disorder)",
    r"\boverdos(e|ed|ing)\b",
]

INFO_GUARD = re.compile(
    r"\b(risk|prevent\w*|prevention|difference between|history of|years? ago|"
    r"last (year|month|week)|recover(ing|ed|y) from|"
    r"how (to|can i|do i) (reduce|lower|avoid))\b"
)

ALWAYS_REGEX = [re.compile(p) for p in ALWAYS_PATTERNS]
CONTEXT_REGEX = [re.compile(p) for p in CONTEXT_PATTERNS]

EMERGENCY_MSG = (
    "⚠️ **This may be an emergency.** Please call **112** (India) or go to the "
    "nearest hospital immediately. Don't wait for an online answer.\n\n"
)


def is_emergency(text):
    t = text.lower().replace("’", "'")
    if any(p.search(t) for p in ALWAYS_REGEX):
        return True
    if INFO_GUARD.search(t):
        return False
    return any(p.search(t) for p in CONTEXT_REGEX)


def build_system_prompt(results):
    if not results:
        return SYSTEM
    context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
    return (
        SYSTEM
        + "\n\nUse the reference material below when relevant, and mention that "
        "it comes from the listed source. If it doesn't cover the question, "
        "say so and answer cautiously from general knowledge.\n\n" + context
    )