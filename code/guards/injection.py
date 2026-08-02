"""Detection of message text that tries to instruct the router rather than inform it.

Decision 7: text arriving from ``dataset/`` is an observed fact to report, never an
instruction to obey. This module recognises the *attempt*. It never routes, never
vetoes and never mutates state; the matched phrase it returns is recorded as evidence
about the message, and the original routing task continues unchanged.

The public API returns the **matched phrase** rather than a boolean so the downstream
``reason`` cell can quote what was detected instead of asserting an unexplained verdict.

Attack shapes recognised
------------------------
Patterns are written against the *shape* of the attack, not against strings observed in
any particular corpus:

1. ``_ADDRESSES_THE_ROUTER`` -- a header naming the notification router, assistant or
   classifier as the addressee of a directive.
2. ``_ASSIGNS_ROUTING_FIELDS`` -- machine-readable ``field=value`` metadata for a
   routing or trust field, embedded in prose a human wrote.
3. ``_COMMANDS_A_ROUTING_OUTCOME`` -- a directive verb whose object is the routing
   decision itself.
4. ``_OVERRIDES_CONSTRAINTS`` -- an instruction to discard rules, policy, safety or
   sender risk.
5. ``_CLAIMS_SYSTEM_VOICE`` -- first-person or role-play claim to be the system,
   operator or platform.
6. ``_ASSERTS_SENDER_METADATA`` -- a claim of verified / trusted / official standing,
   which is a fact the dataset owns and a sender is not entitled to assert.
7. ``_BREAKS_THE_FRAME`` -- a delimiter separating an injected preamble from the
   "real" payload.

False-positive surface
----------------------
This corpus contains legitimate imperative admin traffic and legitimate business
self-description, and neither is an injection. The patterns are therefore built to
*not* fire on:

* imperative instructions aimed at a person ("please use the society app",
  "complete before 5 PM") -- shape 3 requires the object to be a routing outcome, so a
  real-world object never matches;
* ``ignore`` used conversationally ("if already paid, ignore", "kindly ignore this
  message", "do not ignore") -- shape 4 requires a *constraint noun* as the object;
* ``mute`` used by a human about a chat thread ("mute the thread if scores get too
  much") -- a bare outcome word without a directive verb never matches, and negative
  lookaheads exclude a human addressee or a chat-object;
* a business describing itself ("this update is from Flipkart") -- shape 6 requires a
  trust adjective the dataset owns, and shape 5 requires a system-role noun, so a brand
  name alone never matches;
* a residential-society admin, whose "admin notice:" voice is a legitimate human role
  here and is deliberately *not* treated as operator voice. Only a claim to be the
  notification system, platform or moderator qualifies.

Language coverage -- what this does and does not cover
-----------------------------------------------------
Lexical coverage is **English only, deliberately**.

Every routing-directed instruction attempt in this corpus is written in English. Hindi
appears exclusively as romanised Hinglish in Latin script -- there is no Devanagari in
any participant-facing text -- and French appears in Latin script as well. In both
cases the non-English traffic is benign: Hinglish imperatives such as "receipt group me
mat bhejna" and French politeness such as "merci de venir le recuperer" are ordinary
human instructions to a person. For this module those languages are therefore a
*false-positive surface*, and the accompanying tests assert that they return ``None``.

The limitation this leaves is real and worth stating plainly: an attacker who
code-switched an injection into Hinglish, French or a non-Latin script would not be
caught by these patterns. Two things bound that gap. Shapes 2 and 3 key on English
routing vocabulary (``action=notify``, ``notify``, ``mute``, ``digest``) that a
code-switched injection must still carry to be understood by an English-language model,
so partial detection survives code-switching of the surrounding grammar. And this
function is a *precision-oriented pre-flag*, not the only line of defence: the message
text also reaches the model inside an explicitly named untrusted fence, and the model is
told to report an instruction attempt the lexical layer missed. Speculative patterns for
languages that carry no attacks here were deliberately not written, because a pattern
that cannot fire cannot be defended.
"""

import re
import unicodedata
from re import Pattern


# Longer matches are truncated so a single pattern cannot flood the reason cell.
_MAX_PHRASE_CHARS = 80

# Zero-width and bidirectional marks are a standard way to hide an instruction from a
# human reader while leaving it legible to a model, so they are stripped before matching.
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE = re.compile(r"\s+")

# The system components an injection has to name in order to address them.
_ROUTER_NOUN = (
    r"(?:notification\s+)?(?:router|routing\s+system|notification\s+system"
    r"|message\s+router|message\s+filter|assistant|ai\s+(?:assistant|model|system)"
    r"|language\s+model|classifier|moderation\s+system)"
)
# The nouns that turn a mention of the router into a directive aimed at it.
_DIRECTIVE_NOUN = (
    r"(?:note|notice|instruction|instructions|directive|command|metadata|prompt"
    r"|override|config|configuration)"
)
# Fields the router owns. A sender asserting one is asserting a fact it does not hold.
_ROUTING_FIELD = (
    r"(?:action|confidence|verified_business|user_priority|sender_trust|trust_score"
    r"|risk_score|is_spam|priority|urgency|risk|trust|verified|category"
    r"|classification|label|route)"
)
# Values those fields take, which keep the assignment pattern anchored to routing.
_ROUTING_VALUE = (
    r"(?:notify|digest|mute|true|false|high|low|urgent|critical|safe|spam"
    r"|\d+(?:\.\d+)?)"
)
# Roles that speak for the platform, as opposed to roles a human legitimately holds.
_SYSTEM_ROLE = (
    r"(?:system|admin|administrator|operator|moderator|router|assistant|ai|bot|agent"
    r"|security\s+team|support\s+team|whatsapp)"
)


# Shape 1: a directive that names the router, assistant or classifier as its addressee.
_ADDRESSES_THE_ROUTER: tuple[Pattern[str], ...] = (
    re.compile(rf"\b{_DIRECTIVE_NOUN}\s+(?:for|to)\s+(?:the\s+|your\s+)?{_ROUTER_NOUN}\b"),
    re.compile(
        rf"\b(?:internal\s+|system\s+)?(?:router|routing|assistant|agent|system)\s+"
        rf"{_DIRECTIVE_NOUN}\b"
    ),
)

# Shape 2: machine-readable routing metadata asserted inside a human message. The "="
# form is machine syntax and effectively never appears in conversational prose; the ":"
# form does, so it is restricted to fields that can only describe the router's own
# decision, keeping a work message reading "Priority: high" out of the match set.
_ASSIGNS_ROUTING_FIELDS: tuple[Pattern[str], ...] = (
    re.compile(rf"\b{_ROUTING_FIELD}\s*=\s*{_ROUTING_VALUE}\b"),
    re.compile(
        r"\b(?:action|confidence|verified_business|user_priority|sender_trust"
        r"|trust_score|risk_score|is_spam)\s*:\s*"
        r"(?:notify|digest|mute|true|false|high|low|urgent|critical|\d+(?:\.\d+)?)\b"
    ),
)

# Shape 3: a directive verb whose object is the routing decision. The lookahead keeps
# ordinary completions ("mark as read", "mark as paid") out, and the required routing
# outcome keeps real-world imperatives ("complete before 5 PM") out without any lookahead
# at all -- the object does that work.
_COMMANDS_A_ROUTING_OUTCOME: tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:set|sets|mark|marks|marked|marking|flag|flags|flagged|classify"
        r"|classified|classifying|categorize|categorise|categorized|categorised"
        r"|route|routes|treat|treats|tag|tags|tagged|label|labels|labelled|labeled"
        r"|escalate|escalates|force|forced)\s+"
        r"(?:this\s+|it\s+|the\s+)?(?:message\s+|msg\s+|one\s+)?(?:as\s+)?"
        r"(?!read\b|done\b|complete\b|completed\b|paid\b|received\b|resolved\b|sent\b)"
        r"(?:action\s*=\s*)?"
        r"(?:notify|notification|digest|mute|muted|high\s+priority|priority|urgent"
        r"|important|critical|safe|trusted|verified|not\s+spam)\b"
    ),
    # A modal routing command. The lookahead is what separates a command to the router
    # from a human asking a person to notify or mute on their behalf.
    re.compile(
        r"\b(?:always|never|do\s+not|don't|must|should|need\s+to)\s+(?:be\s+)?"
        r"(?:notify|notified|mute|muted|digest|suppress|suppressed|silence|silenced"
        r"|filter|filtered)"
        r"(?!\s+(?:me|us|him|her|them|everyone|anyone|the\s+group|this\s+group"
        r"|the\s+thread|this\s+thread|the\s+chat|this\s+chat))"
    ),
)

# Shape 4: discard the constraints. The determiner run may consume "this", because the
# *noun* is what discriminates: "ignore this rule" matches, "ignore this message" and a
# bare conversational "if already paid, ignore" do not.
_OVERRIDES_CONSTRAINTS: tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:ignore|ignores|ignoring|disregard|disregards|forget|forgets|bypass"
        r"|bypasses|skip|skips|override|overrides|overrule|disable|disables)\s+"
        r"(?:all\s+|any\s+|the\s+|your\s+|our\s+|this\s+|these\s+|those\s+|previous\s+"
        r"|prior\s+|above\s+|earlier\s+|sender\s+|safety\s+|system\s+)*"
        r"(?:instruction|rule|policy|policies|guideline|guardrail|filter|risk|safety"
        r"|check|restriction|warning|constraint|prompt|direction|setting|protocol)s?\b"
    ),
)

# Shape 5: a claim to speak as the system. The role-play lookahead is the "act as a
# customer" versus "act as the system" distinction: a non-system role is excluded before
# the system-role vocabulary is even consulted.
_CLAIMS_SYSTEM_VOICE: tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:you\s+are\s+now|you\s+must\s+now|act\s+as|acting\s+as|pretend\s+to\s+be"
        r"|behave\s+(?:as|like)|respond\s+as|from\s+now\s+on\s+you(?:\s+are)?)\s+"
        r"(?:a\s+|an\s+|the\s+)?"
        r"(?!customer\b|client\b|guest\b|parent\b|student\b|friend\b|member\b"
        r"|resident\b|neighbour\b|neighbor\b|colleague\b)"
        rf"{_SYSTEM_ROLE}\b"
    ),
    re.compile(
        r"\b(?:i\s+am|this\s+is|we\s+are)\s+(?:the\s+|an?\s+)?"
        r"(?:automated\s+|official\s+)?"
        r"(?:system|notification\s+system|routing\s+system|router"
        r"|whatsapp\s+(?:security|support|team)"
        r"|platform\s+(?:admin|administrator|operator))\b"
    ),
)

# Shape 6: assert standing the dataset owns. A brand naming itself is not this; the
# trust adjective is mandatory, which is what keeps "this update is from Flipkart" out.
_ASSERTS_SENDER_METADATA: tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:i\s+am|this\s+is|we\s+are|sender\s+is|forwarded\s+by|sent\s+by"
        r"|message\s+is\s+from)\s+(?:a\s+|an\s+|the\s+)?"
        r"(?:verified|trusted|official|authorised|authorized|whitelisted|approved"
        r"|pre-?approved)\s+"
        r"(?:sender|account|business|admin|administrator|contact|source|number"
        r"|partner|user)\b"
    ),
)

# Shape 7: the delimiter an attacker uses to separate an injected preamble from the
# payload it wraps. This catches the residue when the preamble itself is reworded.
_BREAKS_THE_FRAME: tuple[Pattern[str], ...] = (
    re.compile(
        r"\b(?:actual|real|original|true|user|customer|end\s*user)\s+"
        r"(?:message|content|text|payload)\s*:"
    ),
)


# Evaluated in order, so the phrase returned is the most explanatory available match.
_PATTERNS: tuple[Pattern[str], ...] = (
    _ADDRESSES_THE_ROUTER
    + _ASSIGNS_ROUTING_FIELDS
    + _COMMANDS_A_ROUTING_OUTCOME
    + _OVERRIDES_CONSTRAINTS
    + _CLAIMS_SYSTEM_VOICE
    + _ASSERTS_SENDER_METADATA
    + _BREAKS_THE_FRAME
)


def _normalise(text: str) -> str:
    """Fold case and hidden characters without discarding machine punctuation.

    Deliberately preserves ``=``, ``:`` and ``_``, because the routing-field assignment
    shape lives entirely in that punctuation and a general-purpose text normaliser that
    strips it would erase the strongest signal in this module.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", _ZERO_WIDTH.sub("", folded)).strip()


def looks_like_injection(text: str) -> str | None:
    """Return the phrase by which ``text`` tries to instruct the router, else ``None``.

    The return value is the matched phrase rather than a boolean so the routing decision
    can quote the detection in its reason. A match is evidence *about* the message and
    never an instruction to act on.
    """
    if not text:
        return None
    normalised = _normalise(text)
    if not normalised:
        return None
    for pattern in _PATTERNS:
        match = pattern.search(normalised)
        if match is not None:
            phrase = match.group(0).strip()
            if len(phrase) > _MAX_PHRASE_CHARS:
                return f"{phrase[:_MAX_PHRASE_CHARS]}…"
            return phrase
    return None
