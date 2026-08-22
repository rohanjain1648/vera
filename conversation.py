"""
Vera Message Engine — Conversation State Management
Tracks conversation turns, detects auto-replies, intent transitions,
hostile messages, and manages suppression logic.
"""

import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone


# ─── Auto-reply patterns ──────────────────────────────────────────────────────

AUTO_REPLY_PATTERNS = [
    r"thank\s*you\s*for\s*contact",
    r"thank\s*you\s*for\s*reaching\s*out",
    r"our\s*team\s*will\s*respond",
    r"we\s*will\s*get\s*back\s*to\s*you",
    r"i\s*am\s*an?\s*(automated|virtual|auto)\s*(assistant|bot|responder)",
    r"this\s*is\s*an?\s*(automated|automatic)\s*(message|reply|response)",
    r"we\s*have\s*received\s*your\s*(message|query|enquiry)",
    r"our\s*business\s*hours",
    r"currently\s*(unavailable|away|closed)",
    r"aapki\s*jaankari\s*ke\s*liye\s*bahut.*shukriya",
    r"main\s*aapki.*baatein.*team\s*tak.*pahuncha",
    r"automated\s*assistant",
    r"main\s*ek\s*(automated|virtual)",
    r"shukriya.*jaankari",
    r"reply\s*from\s*our\s*side",
]

AUTO_REPLY_RE = re.compile(
    "|".join(AUTO_REPLY_PATTERNS),
    re.IGNORECASE
)


# ─── Intent commitment patterns ───────────────────────────────────────────────

INTENT_COMMITMENT_PATTERNS = [
    r"\byes\b",
    r"\byes\s*please\b",
    r"\byes[,!.]",
    r"\bgo\s*ahead\b",
    r"\blet'?s\s*do\s*it\b",
    r"\bok\s*let'?s\b",
    r"\bconfirm(ed)?\b",
    r"\bproceed\b",
    r"\bdo\s*it\b",
    r"\bsend\s*it\b",
    r"\bgo\s*for\s*it\b",
    r"\bsounds\s*good\b",
    r"\bwhat'?s\s*next\b",
    r"\bchalega\b",
    r"\bthik\s*hai\b",
    r"\bhaan\b",
    r"\bok\b.{0,20}\bnext\b",
    r"\bstart\b.{0,20}\bnow\b",
    r"\bready\b",
]

INTENT_COMMITMENT_RE = re.compile(
    "|".join(INTENT_COMMITMENT_PATTERNS),
    re.IGNORECASE
)


# ─── Hostility / opt-out patterns ────────────────────────────────────────────

HOSTILE_PATTERNS = [
    r"\bstop\s*(messaging|texting|contacting|this)\b",
    r"\bnot\s*interested\b",
    r"\bdo\s*not\s*(message|text|contact|disturb|bother)\b",
    r"\bleave\s*me\s*alone\b",
    r"\bspam\b",
    r"\bunsubscribe\b",
    r"\bstop\s+sending\b",
    r"\bfuck\s*off\b",
    r"\bkya\s*bakwaas\b",
    r"\bbandh\s*karo\b",
    r"\bmat\s*bhejo\b",
    r"\bmujhe\s*mat\b",
    r"\bblock\s*kar\b",
    r"\bwhy\s*are\s*you\s*bothering\b",
    r"\buseless\s*(spam|message|bot)\b",
    r"\bstop\s+it\b",
]

HOSTILE_RE = re.compile(
    "|".join(HOSTILE_PATTERNS),
    re.IGNORECASE
)


# ─── Out-of-scope patterns ────────────────────────────────────────────────────

OUT_OF_SCOPE_PATTERNS = [
    r"\bgst\s*(filing|return|number)\b",
    r"\bincome\s*tax\b",
    r"\blegal\s*(advice|issue|case)\b",
    r"\bca\s*(help|advice)\b",
    r"\bloan\b.{0,20}\bapply\b",
    r"\bemi\b.{0,20}\bcalculate\b",
    r"\bjob\s*(opening|vacancy|hiring)\b",
    r"\bhelp\s*(me|with).{0,30}(gst|tax|legal|loan|insurance)\b",
]

OUT_OF_SCOPE_RE = re.compile(
    "|".join(OUT_OF_SCOPE_PATTERNS),
    re.IGNORECASE
)


# ─── Conversation State ───────────────────────────────────────────────────────

class ConversationState:
    def __init__(self, conv_id: str, merchant_id: str,
                 customer_id: Optional[str], initial_body: str,
                 trigger: Dict):
        self.conv_id = conv_id
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.trigger = trigger
        self.turns: List[Dict] = [
            {
                "from": "vera",
                "body": initial_body,
                "ts": datetime.now(timezone.utc).isoformat()
            }
        ]
        self.auto_reply_count = 0
        self.last_auto_reply_body: Optional[str] = None
        self.ended = False
        self.suppressed = False

    @property
    def last_body(self) -> str:
        for turn in reversed(self.turns):
            if turn["from"] == "vera":
                return turn["body"]
        return ""

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def add_turn(self, from_role: str, body: str):
        self.turns.append({
            "from": from_role,
            "body": body,
            "ts": datetime.now(timezone.utc).isoformat()
        })

    def to_dict(self) -> Dict:
        return {
            "conv_id": self.conv_id,
            "merchant_id": self.merchant_id,
            "customer_id": self.customer_id,
            "trigger": self.trigger,
            "last_body": self.last_body,
            "turn_count": self.turn_count,
            "auto_reply_count": self.auto_reply_count,
            "ended": self.ended,
        }


# ─── Conversation Manager ─────────────────────────────────────────────────────

class ConversationManager:
    def __init__(self):
        self._conversations: Dict[str, ConversationState] = {}

    def create(self, conv_id: str, merchant_id: str, customer_id: Optional[str],
               initial_body: str, trigger: Dict) -> ConversationState:
        state = ConversationState(conv_id, merchant_id, customer_id, initial_body, trigger)
        self._conversations[conv_id] = state
        return state

    def exists(self, conv_id: str) -> bool:
        return conv_id in self._conversations

    def get(self, conv_id: str) -> Optional[Dict]:
        state = self._conversations.get(conv_id)
        return state.to_dict() if state else None

    def get_state(self, conv_id: str) -> Optional[ConversationState]:
        return self._conversations.get(conv_id)

    def add_turn(self, conv_id: str, from_role: str, body: str):
        state = self._conversations.get(conv_id)
        if state:
            state.add_turn(from_role, body)

    def record_auto_reply(self, conv_id: str, body: str = ""):
        state = self._conversations.get(conv_id)
        if state:
            state.auto_reply_count += 1
            state.last_auto_reply_body = body

    def count_auto_replies(self, conv_id: str) -> int:
        state = self._conversations.get(conv_id)
        return state.auto_reply_count if state else 0

    def get_last_vera_body(self, conv_id: str) -> str:
        state = self._conversations.get(conv_id)
        return state.last_body if state else ""

    def mark_ended(self, conv_id: str):
        state = self._conversations.get(conv_id)
        if state:
            state.ended = True

    def is_ended(self, conv_id: str) -> bool:
        state = self._conversations.get(conv_id)
        return state.ended if state else False

    def get_all_conv_ids(self) -> List[str]:
        return list(self._conversations.keys())


# ─── Message Classifiers ──────────────────────────────────────────────────────

def is_auto_reply(message: str) -> bool:
    """Detect WhatsApp Business auto-reply messages."""
    if not message or len(message) > 500:
        return False
    return bool(AUTO_REPLY_RE.search(message))


def is_intent_commitment(message: str) -> bool:
    """Detect explicit merchant commitment to take action."""
    if not message:
        return False
    return bool(INTENT_COMMITMENT_RE.search(message))


def is_hostile_or_optout(message: str) -> bool:
    """Detect hostile messages or explicit opt-out requests."""
    if not message:
        return False
    return bool(HOSTILE_RE.search(message))


def is_out_of_scope(message: str) -> bool:
    """Detect requests outside Vera's domain (GST, legal, loans, etc.)."""
    if not message:
        return False
    return bool(OUT_OF_SCOPE_RE.search(message))


def classify_reply(message: str) -> str:
    """
    Classify a merchant reply into one of:
    auto_reply | hostile | intent_commitment | out_of_scope | normal
    """
    if is_hostile_or_optout(message):
        return "hostile"
    if is_auto_reply(message):
        return "auto_reply"
    if is_intent_commitment(message):
        return "intent_commitment"
    if is_out_of_scope(message):
        return "out_of_scope"
    return "normal"
