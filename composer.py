"""
Vera Message Engine — Core Composer
Uses Groq LLM with trigger-kind-dispatched prompts to compose high-quality
merchant messages grounded strictly in provided context.
"""

import os
import json
import re
from typing import Optional, Dict, Any
from prompts import build_system_prompt, build_user_prompt, build_followup_prompt, build_action_prompt

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Groq client (stdlib-only HTTP)
from urllib import request as urlrequest
import time


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Use official Groq SDK (handles Cloudflare + proper headers)
try:
    from groq import Groq as GroqClient
    _groq_client = None

    def _get_groq_client():
        global _groq_client
        if _groq_client is None:
            # Re-read key at call time in case dotenv loaded after module import
            api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
            _groq_client = GroqClient(api_key=api_key)
        return _groq_client

except ImportError:
    GroqClient = None
    def _get_groq_client():
        return None


# ─── OpenAI fallback ──────────────────────────────────────────────────────────
# Groq is the primary provider, but this key's 8000 TPM org-wide cap means a burst of
# concurrent /v1/tick compositions (or Groq having a bad moment) can exhaust the retry
# budget in composer.py's _call_groq(). Rather than let the trigger silently produce no
# action, fall back to OpenAI when configured. Optional: if OPENAI_API_KEY isn't set,
# behavior is unchanged (Groq-only, original exception propagates as before).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None


def _openai_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)) and OpenAIClient is not None


def _call_openai_once(system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, str]:
    """Single OpenAI call attempt. Returns (text response, finish_reason)."""
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    model = os.environ.get("OPENAI_MODEL", OPENAI_MODEL)
    client = OpenAIClient(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = completion.choices[0]
    return (choice.message.content or "").strip(), (choice.finish_reason or "")


# openai/gpt-oss-120b (served via Groq) is a REASONING model: it spends tokens on a
# hidden chain-of-thought before emitting the JSON answer. At the old max_tokens=800,
# reasoning alone routinely exhausted the budget (finish_reason="length", empty content),
# silently dropping into the fact-free fallback prompt below and causing fabricated output.
# Fix: cap reasoning effort ("low") and give enough headroom for reasoning + full JSON body.
DEFAULT_MAX_TOKENS = 1600
DEFAULT_REASONING_EFFORT = "low"


def _extract_retry_after(err: Exception) -> float:
    """Parse Groq's 429 'Please try again in Xs' hint out of the error message."""
    m = re.search(r"try again in ([\d.]+)s", str(err))
    return float(m.group(1)) if m else 2.0


def _call_groq_once(system: str, user: str, temperature: float,
                     max_tokens: int, reasoning_effort: str) -> tuple[str, str]:
    """Single Groq call attempt. Returns (text response, finish_reason)."""
    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    model = os.environ.get("GROQ_MODEL", GROQ_MODEL)

    kwargs = {}
    if reasoning_effort and "gpt-oss" in model:
        kwargs["reasoning_effort"] = reasoning_effort

    # Use official SDK if available
    if GroqClient is not None:
        client = GroqClient(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = completion.choices[0]
        return (choice.message.content or "").strip(), (choice.finish_reason or "")

    # Fallback: raw HTTP with proper headers
    from urllib import request as urlrequest
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(kwargs)
    body = json.dumps(payload).encode("utf-8")

    req = urlrequest.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "groq-python/1.4.0",
        }
    )
    resp = urlrequest.urlopen(req, timeout=25)
    data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]
    content = (choice["message"]["content"] or "").strip()
    return content, choice.get("finish_reason", "")


def _call_groq(system: str, user: str, temperature: float = 0.0,
                max_tokens: int = DEFAULT_MAX_TOKENS,
                reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                deadline: Optional[float] = None) -> tuple[str, str]:
    """
    Call Groq API and return (text response, finish_reason). Falls back to OpenAI
    (if OPENAI_API_KEY is configured) when Groq can't complete the call at all.

    Under concurrent /v1/tick load this account's Groq key hits its TPM rate limit
    (429 tokens_per_minute). Left unhandled, main.py's per-trigger try/except swallows
    that exception and the trigger silently produces NO action at all for the whole
    tick — a bigger scoring risk than the reasoning-truncation bug, since it's the same
    key shipped with the submission. Retry once after the server-suggested backoff,
    bounded by the caller's deadline (the tick handler's own 30s-safe wall-clock budget)
    so a single stuck call can never blow the harness timeout. If Groq still fails
    (rate limit exhausted, or any other error), hand off to OpenAI as a last resort
    before giving up — same grounded prompt, just a different provider.
    """
    try:
        return _call_groq_once(system, user, temperature, max_tokens, reasoning_effort)
    except Exception as e:
        is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
        if not is_rate_limit:
            return _fallback_to_openai(system, user, temperature, max_tokens, e)

        wait = _extract_retry_after(e)
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0.5:
                return _fallback_to_openai(system, user, temperature, max_tokens, e)
            wait = min(wait, max(0.1, remaining - 0.5))
        time.sleep(wait)

        try:
            return _call_groq_once(system, user, temperature, max_tokens, reasoning_effort)
        except Exception as e2:
            return _fallback_to_openai(system, user, temperature, max_tokens, e2)


def _fallback_to_openai(system: str, user: str, temperature: float,
                         max_tokens: int, groq_error: Exception) -> tuple[str, str]:
    """Try OpenAI when Groq has exhausted its own retries. Re-raises the original
    Groq error if OpenAI isn't configured, so behavior is unchanged when no
    OPENAI_API_KEY is present."""
    if not _openai_configured():
        raise groq_error
    try:
        return _call_openai_once(system, user, temperature, max_tokens)
    except Exception:
        # If OpenAI also fails, surface the original Groq error — it's the primary
        # provider and its failure mode is the one callers already handle.
        raise groq_error


def _parse_llm_output(raw: str) -> Dict[str, Any]:
    """
    Parse LLM JSON output. Tries strict JSON first, then regex extraction.
    Expected keys: body, cta, send_as, suppression_key, rationale
    """
    # Try to find JSON block
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: extract fields manually
    result = {}
    for field in ["body", "cta", "send_as", "suppression_key", "rationale"]:
        m = re.search(rf'"{field}"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw)
        if m:
            result[field] = m.group(1).replace('\\"', '"').replace('\\n', '\n')
    return result


def _validate_output(composed: Dict, trigger: Dict, merchant: Dict,
                     customer: Optional[Dict]) -> Dict:
    """
    Validate and fix composed output:
    - No URLs
    - Correct send_as
    - Valid CTA
    - Non-empty body
    """
    # Fix send_as
    trigger_scope = trigger.get("scope", "merchant")
    if trigger_scope == "customer" and customer:
        composed["send_as"] = "merchant_on_behalf"
    else:
        composed["send_as"] = "vera"

    # Remove URLs (hard fail according to spec)
    body = composed.get("body", "")
    body = re.sub(r'https?://\S+', '', body).strip()
    composed["body"] = body

    # Validate CTA
    valid_ctas = {"binary_yes_no", "binary_confirm_cancel", "open_ended",
                  "multi_choice_slot", "none"}
    if composed.get("cta") not in valid_ctas:
        # Default CTA by trigger kind
        kind = trigger.get("kind", "")
        if kind in ("recall_due", "chronic_refill_due", "appointment_tomorrow"):
            composed["cta"] = "multi_choice_slot"
        elif kind in ("research_digest", "curious_ask_due", "category_seasonal"):
            composed["cta"] = "open_ended"
        else:
            composed["cta"] = "binary_yes_no"

    # Suppression key fallback
    if not composed.get("suppression_key"):
        composed["suppression_key"] = trigger.get("suppression_key", f"trigger:{trigger.get('id', 'unknown')}")

    return composed


def compose(
    category: Dict,
    merchant: Dict,
    trigger: Dict,
    customer: Optional[Dict] = None,
    deadline: Optional[float] = None
) -> Dict[str, Any]:
    """
    Core composition function.
    Returns dict with: body, cta, send_as, suppression_key, rationale
    `deadline` (time.time()-based) bounds retry waits so a caller with its own
    wall-clock budget (e.g. the /v1/tick handler) never blocks past it.
    """
    system_prompt = build_system_prompt(category, merchant, trigger, customer)
    user_prompt = build_user_prompt(category, merchant, trigger, customer)

    raw, finish_reason = _call_groq(system_prompt, user_prompt, temperature=0.0, deadline=deadline)
    composed = _parse_llm_output(raw)

    if not composed.get("body") and finish_reason == "length":
        # The model's hidden reasoning ate the whole token budget before it could write
        # the JSON answer. Retry the SAME grounded prompt with more headroom rather than
        # dropping into a fact-free fallback — this keeps specificity/grounding intact.
        raw, finish_reason = _call_groq(system_prompt, user_prompt, temperature=0.0,
                                         max_tokens=DEFAULT_MAX_TOKENS * 2, deadline=deadline)
        composed = _parse_llm_output(raw)

    if not composed.get("body"):
        # Last resort: simpler prompt, but still carrying the real grounding facts
        # (never fact-free — a fact-free fallback is what causes fabrication).
        fallback_prompt = _build_fallback_prompt(category, merchant, trigger, customer)
        raw2, _ = _call_groq(
            "You are Vera, magicpin's merchant growth assistant. Compose a short, specific "
            "WhatsApp message using ONLY the facts given below. Never invent numbers, offers, "
            "or claims not present in the context.",
            fallback_prompt,
            temperature=0.1, deadline=deadline
        )
        composed = _parse_llm_output(raw2)
        if not composed.get("body"):
            composed["body"] = raw2.strip()[:500]
            composed["cta"] = "open_ended"
            composed["rationale"] = "Fallback composition"

    composed = _validate_output(composed, trigger, merchant, customer)
    return composed


def compose_action_response(
    category: Dict,
    merchant: Dict,
    trigger: Dict,
    conv_state: Optional[Dict] = None,
    deadline: Optional[float] = None
) -> str:
    """
    Compose an ACTION message when merchant says 'yes/let's do it'.
    Returns just the body string.
    """
    system = (
        "You are Vera, magicpin's merchant AI assistant. "
        "The merchant just COMMITTED to taking action. DO NOT ask any more qualifying questions. "
        "Immediately give them ONE concrete next step with a clear confirmation CTA. "
        "Be specific, brief, and action-oriented. No preambles."
    )
    user = build_action_prompt(category, merchant, trigger, conv_state)
    raw, finish_reason = _call_groq(system, user, temperature=0.0, deadline=deadline)
    if not raw and finish_reason == "length":
        raw, _ = _call_groq(system, user, temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS * 2, deadline=deadline)
    # Strip JSON if returned
    data = _parse_llm_output(raw)
    if data.get("body"):
        return data["body"]
    return raw.strip()[:400]


def compose_followup(
    category: Dict,
    merchant: Dict,
    trigger: Dict,
    conv_state: Optional[Dict],
    merchant_message: str,
    deadline: Optional[float] = None
) -> str:
    """
    Compose a follow-up reply given the merchant's latest message.
    Returns just the body string.
    """
    system = (
        "You are Vera, magicpin's merchant AI assistant. "
        "Reply to the merchant's message. Be brief, specific, helpful. "
        "Advance the conversation to the next useful step. "
        "Never ask more than one question. No preambles."
    )
    user = build_followup_prompt(category, merchant, trigger, conv_state, merchant_message)
    raw, finish_reason = _call_groq(system, user, temperature=0.0, deadline=deadline)
    if not raw and finish_reason == "length":
        raw, _ = _call_groq(system, user, temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS * 2, deadline=deadline)
    data = _parse_llm_output(raw)
    if data.get("body"):
        return data["body"]
    return raw.strip()[:400]


def _build_fallback_prompt(category, merchant, trigger, customer):
    """
    Last-resort fallback prompt when main composition fails twice.
    IMPORTANT: must still carry real grounding facts. A fact-free fallback is what
    causes the model to fabricate plausible-sounding but invented numbers/claims —
    that was the original root cause of low specificity/fabrication penalties.
    """
    identity = merchant.get("identity", {})
    merchant_name = identity.get("owner_first_name") or identity.get("name", "")
    kind = trigger.get("kind", "")
    sup_key = trigger.get("suppression_key", "")
    perf = merchant.get("performance", {})
    active_offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    trg_payload = trigger.get("payload", {})

    digest = category.get("digest", [])
    relevant_digest = None
    if trg_payload.get("top_item_id"):
        relevant_digest = next((d for d in digest if d.get("id") == trg_payload["top_item_id"]), None)

    facts = [
        f"Merchant: {identity.get('name', '')} ({merchant_name}) in {identity.get('locality', '')}",
        f"Performance (30d): views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, ctr={perf.get('ctr', '?')}",
        f"Active offers: {active_offers or 'none'}",
        f"Signals: {signals}",
        f"Trigger kind: {kind}, payload: {json.dumps(trg_payload)}",
    ]
    if relevant_digest:
        facts.append(
            f"Digest finding: {relevant_digest.get('title', '')} "
            f"(source: {relevant_digest.get('source', '')}, trial_n={relevant_digest.get('trial_n', '')}) "
            f"— {relevant_digest.get('summary', '')}"
        )
    if customer:
        cust_identity = customer.get("identity", {})
        facts.append(f"Customer: {cust_identity.get('name', '')}, language: {cust_identity.get('language_pref', '')}")

    facts_block = "\n".join(f"- {f}" for f in facts)

    return (
        f"Write a short WhatsApp message from Vera to {merchant_name}.\n\n"
        f"ONLY use these facts — do not invent any number, claim, or offer not listed here:\n"
        f"{facts_block}\n\n"
        f"Be specific and useful. End with one clear question or CTA. "
        f"Return JSON: {{\"body\": \"...\", \"cta\": \"open_ended\", "
        f"\"send_as\": \"vera\", \"suppression_key\": \"{sup_key}\", "
        f"\"rationale\": \"...\"}}"
    )
