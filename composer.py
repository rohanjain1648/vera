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


def _call_groq(system: str, user: str, temperature: float = 0.0) -> str:
    """Call Groq API and return text response."""
    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    model = os.environ.get("GROQ_MODEL", GROQ_MODEL)

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
            max_tokens=800,
        )
        return completion.choices[0].message.content.strip()

    # Fallback: raw HTTP with proper headers
    from urllib import request as urlrequest
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "max_tokens": 800,
    }).encode("utf-8")

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
    return data["choices"][0]["message"]["content"].strip()


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
    customer: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Core composition function.
    Returns dict with: body, cta, send_as, suppression_key, rationale
    """
    system_prompt = build_system_prompt(category, merchant, trigger, customer)
    user_prompt = build_user_prompt(category, merchant, trigger, customer)

    raw = _call_groq(system_prompt, user_prompt, temperature=0.0)
    composed = _parse_llm_output(raw)

    if not composed.get("body"):
        # Retry once with a simpler prompt
        fallback_prompt = _build_fallback_prompt(category, merchant, trigger, customer)
        raw2 = _call_groq(
            "You are Vera, magicpin's merchant growth assistant. Compose a short, specific WhatsApp message.",
            fallback_prompt,
            temperature=0.1
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
    conv_state: Optional[Dict] = None
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
    raw = _call_groq(system, user, temperature=0.0)
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
    merchant_message: str
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
    raw = _call_groq(system, user, temperature=0.0)
    data = _parse_llm_output(raw)
    if data.get("body"):
        return data["body"]
    return raw.strip()[:400]


def _build_fallback_prompt(category, merchant, trigger, customer):
    """Simple fallback prompt when main composition fails."""
    merchant_name = merchant.get("identity", {}).get("owner_first_name") or \
                    merchant.get("identity", {}).get("name", "")
    kind = trigger.get("kind", "")
    sup_key = trigger.get("suppression_key", "")
    return (
        f"Write a short WhatsApp message from Vera to {merchant_name}. "
        f"Trigger: {kind}. "
        f"Merchant: {merchant.get('identity', {}).get('name', '')} in "
        f"{merchant.get('identity', {}).get('locality', '')}. "
        f"Be specific and useful. End with one clear question. "
        f"Return JSON: {{\"body\": \"...\", \"cta\": \"open_ended\", "
        f"\"send_as\": \"vera\", \"suppression_key\": \"{sup_key}\", "
        f"\"rationale\": \"...\"}}"
    )
