import difflib
import re

MODEL = "openai/gpt-oss-120b"

SYSTEM = """You are a health information assistant.

Style rules (very important):
- Start directly with the answer. Do NOT open with sympathy or emotional phrases.
  Never say things like "I'm sorry you're dealing with that", "I'm sorry to hear that",
  "That sounds painful", or "I understand how you feel".
- Keep a neutral, factual, professional tone. No small talk beyond the conversation
  rules below, and no filler openings.
- Treat questions like "what are the symptoms of X" as general information requests.
  Do not assume the user has the condition.

Conversation rules:
- Greetings ("hi", "hello", "hey", "namaste", "good morning"): reply with one short,
  friendly greeting and invite them to ask a health question. Give no health
  information and mention no sources in that reply.
- "How are you" and similar pleasantries, even with spelling mistakes ("how ar eyou",
  "hw r u"): reply with one short friendly sentence and invite a health question.
  Do NOT describe what you can do and do NOT list your limitations.
- Thanks or goodbye: reply with one short, polite sentence.
- Questions about you, only when clearly asked ("who are you", "what can you do",
  "what are you"): say you are a health information assistant that explains symptoms,
  conditions, nutrition and wellness in plain language, and that you cannot diagnose,
  prescribe, or replace a doctor.
- Messages unrelated to health (coding, sports, politics, entertainment, general
  trivia): say politely in one or two sentences that you can only help with health
  topics, and invite a health question. Do not answer the unrelated question.
- If a greeting comes together with a health question, answer the health question
  directly.
- Reply in the language the user writes in.

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


# ---------------------------------------------------------------------------
# Small talk. Only messages that are NOTHING BUT a greeting, pleasantry, thanks or
# goodbye are answered here (instantly, without calling the model). Anything else,
# such as "hi, what causes headache?", goes to the model as usual.
# Two layers: exact patterns first, then a typo-tolerant match that ignores spaces
# (so "how ar eyou" still counts as "how are you").
# ---------------------------------------------------------------------------
GREETING_REPLY = (
    "Hello! 👋 I'm your health information assistant. "
    "What health question can I help you with today?"
)
HOW_ARE_YOU_REPLY = (
    "I'm doing well, thank you! I'm here to help with health questions. "
    "What would you like to know?"
)
THANKS_REPLY = "You're welcome! Ask me anything else about your health whenever you like."
BYE_REPLY = "Take care! Come back any time you have a health question."

_EXACT = [
    (re.compile(
        r"^(hi+|hel+o+|helo|hlo|hey+|heya|hola|namaste|namaskar|namaskaram|vanakkam|"
        r"good (morning|afternoon|evening))"
        r"( (there|everyone|all|team|doctor|bot))?$"), GREETING_REPLY),
    (re.compile(
        r"^((hi+|hel+o+|hey+) )?(how are you|how are you doing|how are u|how r you|how r u|"
        r"hw are u|hw r u|hows it going|how do you do|whats up|sup)( today)?$"), HOW_ARE_YOU_REPLY),
    (re.compile(
        r"^(ok(ay)? )?(thanks|thank you|thank u|thx|ty|many thanks|dhanyavad|shukriya)"
        r"( a lot| so much| very much)?$"), THANKS_REPLY),
    (re.compile(
        r"^(ok(ay)? )?(bye|bye bye|goodbye|good bye|good night|see you|see ya|take care)$"),
     BYE_REPLY),
]

# Longer phrases only (squashed, no spaces), so short words cannot match by accident
_FUZZY = [
    ("hello", GREETING_REPLY),
    ("namaste", GREETING_REPLY),
    ("goodmorning", GREETING_REPLY),
    ("goodafternoon", GREETING_REPLY),
    ("goodevening", GREETING_REPLY),
    ("howareyou", HOW_ARE_YOU_REPLY),
    ("howareyoudoing", HOW_ARE_YOU_REPLY),
    ("whatsup", HOW_ARE_YOU_REPLY),
    ("thanks", THANKS_REPLY),
    ("thankyou", THANKS_REPLY),
    ("goodbye", BYE_REPLY),
    ("goodnight", BYE_REPLY),
]
_FUZZY_MIN_SCORE = 0.82


def _normalize(text):
    t = re.sub(r"['’]", "", text.lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def smalltalk_reply(text):
    """Return a fixed reply for a pure greeting, pleasantry, thanks or goodbye, else None."""
    t = _normalize(text)
    if not t or len(t) > 30:
        return None

    for pattern, reply in _EXACT:
        if pattern.match(t):
            return reply

    squashed = t.replace(" ", "")
    best_reply, best_score = None, 0.0
    for target, reply in _FUZZY:
        score = difflib.SequenceMatcher(None, squashed, target).ratio()
        if score > best_score:
            best_reply, best_score = reply, score
    return best_reply if best_score >= _FUZZY_MIN_SCORE else None


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