import difflib
import re

MODEL = "openai/gpt-oss-120b"

SYSTEM = """You are a health information assistant.

Style rules (very important):
- Start directly with the answer. Do NOT open with sympathy or emotional phrases.
  Never say things like "I'm sorry you're dealing with that", "I'm sorry to hear that",
  "That sounds painful", or "I understand how you feel".
- Keep a neutral, factual, professional tone and no filler openings, except in the
  conversation rules below, where a friendly tone is right.
- Treat questions like "what are the symptoms of X" as general information requests.
  Do not assume the user has the condition.

Conversation rules:
- Questions about YOU are NOT off-topic. This includes "who are you", "what are you",
  "what can you do", "what is your name", "who made you", "are you a human or a
  doctor", and "how are you" (also with spelling mistakes). Answer them directly in
  2 or 3 plain sentences: you are an AI health assistant that explains symptoms,
  conditions, nutrition and wellness in plain language; you are not a human or a
  doctor and cannot diagnose, prescribe or replace a doctor. Then ask what health
  question you can help with. Never refuse these and never say you can only help
  with health topics.
- Greetings ("hi", "hello", "hey", "namaste", "good morning"): reply with one short,
  friendly greeting and invite them to ask a health question. Give no health
  information and mention no sources in that reply.
- "How are you" and similar pleasantries: reply with one short friendly sentence and
  invite a health question. Do not list your limitations.
- Thanks or goodbye: reply with one short, polite sentence.
- Messages unrelated to health AND unrelated to you (coding, sports, politics,
  entertainment, general trivia): say politely in one or two sentences that you can
  only help with health topics, and invite a health question. Do not answer the
  unrelated question.
- If a greeting or pleasantry comes together with a health question, answer the health
  question directly.
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
# Small talk and questions about the assistant. Only messages that are NOTHING BUT
# one of these are answered here (instantly, without calling the model). Anything
# else, such as "hi, what causes headache?", goes to the model as usual.
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
PRESENT_REPLY = "Yes, I'm here! What health question can I help you with?"
WHO_ARE_YOU_REPLY = (
    "I'm a health assistant: an AI that explains symptoms, conditions, nutrition and "
    "wellness in plain language, using trusted health sources. I can't diagnose "
    "illnesses, prescribe medicines or replace a doctor. "
    "What health question can I help you with?"
)
WHAT_CAN_YOU_DO_REPLY = (
    "I can help you with:\n\n"
    "- Understanding symptoms and common conditions\n"
    "- Nutrition, fitness and healthy-lifestyle questions\n"
    "- Explaining what a medical term or test means in general\n"
    "- Knowing when it's time to see a doctor\n\n"
    "I can't diagnose, prescribe medicines or replace a doctor. "
    "What would you like to know?"
)
NAME_REPLY = (
    "I don't have a personal name. I'm a Health Assistant, an AI for general health "
    "information. What can I help you with?"
)
HUMAN_REPLY = (
    "I'm an AI assistant, not a human and not a doctor. I can share general health "
    "information, but for diagnosis or treatment please see a qualified doctor. "
    "What would you like to know?"
)
MAKER_REPLY = (
    "I'm an AI health assistant, built as a software project that combines a language "
    "model with a library of trusted health sources. I'm not a doctor. "
    "What health question can I help you with?"
)
HELP_REPLY = (
    "Of course, I'm here to help. Tell me your health question, or the symptoms you're "
    "worried about. If this is an emergency, call 112 right away."
)

# optional lead-in words: "hi who are you", "tell me what can you do", "ok thanks"
_LEAD = r"^((hi+|hel+o+|hey+|ok(ay)?|so|and|please|pls|tell me|can you tell me|could you tell me) )*"

_EXACT = [
    (re.compile(
        r"^(hi+|hel+o+|helo|hlo|hey+|heya|hola|namaste|namaskar|namaskaram|vanakkam|"
        r"good (morning|afternoon|evening))"
        r"( (there|everyone|all|team|doctor|bot))?$"), GREETING_REPLY),
    (re.compile(
        _LEAD + r"(how are you|how are you doing|how are u|how r you|how r u|hw are u|"
        r"hw r u|hows it going|how do you do|whats up|sup)( today)?$"), HOW_ARE_YOU_REPLY),
    (re.compile(
        r"^(ok(ay)? )?(thanks|thank you|thank u|thx|ty|many thanks|dhanyavad|shukriya)"
        r"( a lot| so much| very much)?$"), THANKS_REPLY),
    (re.compile(
        r"^(ok(ay)? )?(bye|bye bye|goodbye|good bye|good night|see you|see ya|take care)$"),
     BYE_REPLY),
    (re.compile(r"^(are you there|are u there|you there|anyone there|is anyone there)$"),
     PRESENT_REPLY),
    (re.compile(
        _LEAD + r"(who are you|who r you|who r u|who are u|who is this|who am i talking to|"
        r"what are you|what r you|what r u|what are u|what is this|what is this app|"
        r"what is this bot|what is this chatbot|introduce yourself|tell me about yourself|"
        r"about yourself|about you)( exactly| really| then| bot)?$"), WHO_ARE_YOU_REPLY),
    (re.compile(
        _LEAD + r"(what can you do|what can u do|what do you do|what all can you do|"
        r"what all can u do|how can you help( me)?|how can u help( me)?|how do you help( me)?|"
        r"what are your features|what are you capable of|what are your capabilities|"
        r"what can you help( me)? with|what do you know|how does this work|"
        r"how do i use this|how to use this)$"), WHAT_CAN_YOU_DO_REPLY),
    (re.compile(
        _LEAD + r"(what is your name|whats your name|what are you called|your name|"
        r"what should i call you|may i know your name|tell me your name|do you have a name|"
        r"what do they call you)$"), NAME_REPLY),
    (re.compile(
        _LEAD + r"(are (you|u) (a |an )?(real )?(human|person|robot|bot|ai|doctor|chatbot|"
        r"machine|physician)|r u (a |an )?(real )?(human|person|robot|bot|ai|doctor)|"
        r"are you real|is this (a )?(real )?(doctor|human|person|bot)|"
        r"am i talking to (a |an )?(human|person|doctor|bot|ai|robot))$"), HUMAN_REPLY),
    (re.compile(
        _LEAD + r"(who (made|created|built|developed|designed|trained|programmed) "
        r"(you|u|this|this app|this bot)|who is your (creator|developer|owner|maker|founder))$"),
     MAKER_REPLY),
    (re.compile(
        _LEAD + r"(help|help me|i need help|can you help( me)?|could you help( me)?|"
        r"can u help( me)?|need your help|can i ask you something|can i ask something|"
        r"can i ask a question|i have a question|i have a doubt|i want to ask something|"
        r"i want to ask a question)$"), HELP_REPLY),
]

# Typo-tolerant fallback. Phrases are squashed (no spaces). "who are you" and
# "what are you" share one reply, so confusing them with each other is harmless.
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
    ("areyouthere", PRESENT_REPLY),
    ("whoareyou", WHO_ARE_YOU_REPLY),
    ("whatareyou", WHO_ARE_YOU_REPLY),
    ("whatcanyoudo", WHAT_CAN_YOU_DO_REPLY),
    ("whatisyourname", NAME_REPLY),
    ("areyouahuman", HUMAN_REPLY),
    ("areyouadoctor", HUMAN_REPLY),
    ("whomadeyou", MAKER_REPLY),
    ("canyouhelpme", HELP_REPLY),
]
_FUZZY_MIN_SCORE = 0.82


def _normalize(text):
    t = re.sub(r"['’]", "", text.lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def smalltalk_reply(text):
    """Return a fixed reply for small talk or a question about the assistant, else None."""
    t = _normalize(text)
    if not t or len(t) > 40:
        return None

    for pattern, reply in _EXACT:
        if pattern.match(t):
            return reply

    squashed = t.replace(" ", "")
    if len(squashed) < 5:
        return None
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