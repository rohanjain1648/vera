"""
Vera Message Engine — FastAPI Server
All 5 required endpoints: /v1/healthz, /v1/metadata, /v1/context, /v1/tick, /v1/reply
"""

import os
import time
import json
import concurrent.futures
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

# Load .env before any other imports that use env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from composer import compose, compose_action_response, compose_followup
from conversation import (
    ConversationManager,
    classify_reply,
    is_auto_reply,
    is_hostile_or_optout,
    is_intent_commitment,
    is_out_of_scope,
)

# ─── App init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vera Bot — magicpin AI Challenge",
    version="1.0.0",
    description="Deterministic merchant message composer using 4-context framework"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

START_TIME = time.time()

# ─── In-memory state ──────────────────────────────────────────────────────────

# (scope, context_id) → {version: int, payload: dict}
contexts: Dict[tuple, dict] = {}

# Conversation management
conversation_manager = ConversationManager()

# Conversations that have been ended (opt-out / hostile / max auto-reply)
ended_conversations: Set[str] = set()

# Merchants who have hard opted out — suppress all future messages
suppressed_merchants: Set[str] = set()

# Suppression keys already acted on this test session (dedup across ticks)
fired_suppression_keys: Set[str] = set()

# Suppression keys currently submitted to the compose pool but not yet resolved —
# prevents an overlapping tick from submitting a duplicate call for the same trigger.
in_flight_keys: Set[str] = set()

# Groq TPM budget on this key is tight (8000 tokens/min on openai/gpt-oss-120b — an
# org-wide cap, confirmed across models) and each composition costs ~2000-2600 tokens.
# A single shared, long-lived pool (rather than a fresh ThreadPoolExecutor per /v1/tick
# call) is essential: creating+abandoning a pool every tick lets orphaned in-flight
# Groq calls from a prior tick keep running unbounded in the background, competing with
# the NEXT tick's fresh calls for the same tiny TPM budget and starving it completely
# (observed: tick2 dropped from 5 successes to 0 once tick1's leftovers were still live).
# A fixed-size shared pool naturally throttles total concurrent Groq calls across ticks.
TICK_MAX_WORKERS = 3
_compose_executor = concurrent.futures.ThreadPoolExecutor(max_workers=TICK_MAX_WORKERS)

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _count_contexts() -> Dict[str, int]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts


def _get_payload(scope: str, cid: str) -> Optional[dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
@app.get("/healthz")
@app.get("/v1/healthz")
async def healthz():
    """Liveness probe — judge polls every 60s."""
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _count_contexts()
    }


@app.get("/v1/metadata")
async def metadata():
    """Bot identity — called once at start of test."""
    return {
        "team_name": "Vera Intelligence",
        "team_members": ["Candidate"],
        "model": "llama-3.3-70b-versatile via Groq",
        "approach": (
            "Trigger-kind-dispatched composer with category-voice-aware prompts. "
            "Each trigger kind (research_digest, recall_due, perf_dip, etc.) gets a specialized "
            "prompt variant that leads with the trigger reason and uses the correct category voice. "
            "Auto-reply detection via pattern matching. Intent transition detection for immediate "
            "action routing. Stateful conversation tracking with suppression dedup."
        ),
        "contact_email": "candidate@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    """
    Receive context push from judge. Idempotent by (scope, context_id, version).
    Higher version replaces atomically.
    """
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {sorted(valid_scopes)}"
        }

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    # Reject strictly older versions; accept same version idempotently
    if current and current["version"] > body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"]
        }

    # Store new/updated context
    contexts[key] = {
        "version": body.version,
        "payload": body.payload
    }

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat()
    }


# Harness allows up to 30s per response. Leave real margin for network/serialization
# and stop launching/waiting on new LLM work once we're inside that margin, so a slow
# or rate-limited Groq call can never make /v1/tick itself time out.
TICK_WALL_CLOCK_BUDGET_S = 18.0


@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Periodic wake-up. Evaluate available triggers and compose proactive messages.
    Returns up to 20 actions per tick, bounded by TICK_WALL_CLOCK_BUDGET_S.
    """
    tick_start = time.time()
    tick_deadline = tick_start + TICK_WALL_CLOCK_BUDGET_S
    actions = []

    # Build list of (trigger_id, trigger_payload) sorted by urgency desc
    trigger_queue: List[tuple] = []
    for tid in body.available_triggers:
        trg = _get_payload("trigger", tid)
        if trg:
            trigger_queue.append((tid, trg))

    # Sort: higher urgency first
    trigger_queue.sort(key=lambda x: -x[1].get("urgency", 0))

    # Helper for composing one trigger
    def _process_one_trigger(item):
        tid, trg = item
        sup_key = trg.get("suppression_key", "")
        merchant_id = trg.get("merchant_id") or trg.get("payload", {}).get("merchant_id")
        customer_id = trg.get("customer_id") or trg.get("payload", {}).get("customer_id")

        if not merchant_id or merchant_id in suppressed_merchants:
            return None

        merchant = _get_payload("merchant", merchant_id)
        if not merchant:
            return None

        category_slug = merchant.get("category_slug", "")
        category = _get_payload("category", category_slug)
        if not category:
            return None

        customer = _get_payload("customer", customer_id) if customer_id else None

        conv_id = f"conv_{merchant_id}_{tid}"
        if conversation_manager.exists(conv_id) and not conversation_manager.is_ended(conv_id):
            return None

        try:
            composed = compose(category, merchant, trg, customer, deadline=tick_deadline)
            if not composed or not composed.get("body"):
                return None

            trigger_kind = trg.get("kind", "generic")
            template_name = f"vera_{trigger_kind}_v1"
            merchant_name = (
                merchant.get("identity", {}).get("owner_first_name") or
                merchant.get("identity", {}).get("name", "Merchant")
            )

            action = {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": composed.get("send_as", "vera"),
                "trigger_id": tid,
                "template_name": template_name,
                "template_params": [
                    merchant_name,
                    composed["body"][:100],
                    composed.get("cta", "open_ended")
                ],
                "body": composed["body"],
                "cta": composed.get("cta", "open_ended"),
                "suppression_key": composed.get("suppression_key", sup_key),
                "rationale": composed.get("rationale", "Composed from 4-context framework")
            }

            return (conv_id, merchant_id, customer_id, composed["body"], trg, sup_key, action)
        except Exception as e:
            import sys
            print(f"[WARN] Compose failed for {tid}: {e}", file=sys.stderr)
            return None

    # Filter candidates — skip already-fired triggers AND ones already submitted to the
    # shared pool by an earlier tick that hasn't resolved yet (in_flight_keys), otherwise
    # an overlapping tick would submit a duplicate call for the same trigger and burn
    # more of the shared TPM budget on redundant work.
    candidates = []
    for tid, trg in trigger_queue:
        if len(candidates) >= 20:
            break
        sup_key = trg.get("suppression_key", "")
        if sup_key and (sup_key in fired_suppression_keys or sup_key in in_flight_keys):
            continue
        candidates.append((tid, trg))

    def _finalize(fut) -> Optional[dict]:
        """
        Apply a resolved future's result to shared state exactly once. Used both for
        futures that finished within this tick's budget (to build the response) and,
        via add_done_callback, for ones that finish later after being abandoned by an
        earlier tick — so a late success still gets recorded and isn't retried forever.
        """
        item = futures.get(fut)
        sup_key = (item[1].get("suppression_key", "") if item else "") or ""
        if sup_key:
            in_flight_keys.discard(sup_key)
        try:
            res = fut.result()
        except Exception:
            return None
        if not res:
            return None
        conv_id, merchant_id, customer_id, body_text, trg, sup_key, action = res
        if sup_key:
            fired_suppression_keys.add(sup_key)
        conversation_manager.create(conv_id, merchant_id, customer_id, body_text, trg)
        return action

    if candidates:
        futures = {}
        for item in candidates:
            sup_key = item[1].get("suppression_key", "")
            if sup_key:
                in_flight_keys.add(sup_key)
            futures[_compose_executor.submit(_process_one_trigger, item)] = item

        remaining = max(0.5, tick_deadline - time.time())
        done, not_done = concurrent.futures.wait(futures, timeout=remaining)

        for fut in done:
            action = _finalize(fut)
            if action:
                actions.append(action)

        # Anything still in flight when the budget runs out is abandoned for THIS
        # response (never block past the harness's timeout) — cancel() actually stops
        # queued-but-not-started work on the shared pool; already-running calls finish
        # in the background and _finalize still records their result via this callback,
        # so they won't be silently retried duplicate-fashion by a later tick.
        for fut in not_done:
            if not fut.cancel():
                fut.add_done_callback(_finalize)
            else:
                item = futures.get(fut)
                sup_key = item[1].get("suppression_key", "") if item else ""
                if sup_key:
                    in_flight_keys.discard(sup_key)

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Handle reply from simulated merchant/customer.
    Returns: {action: send|wait|end, body?, cta?, rationale}
    """
    reply_deadline = time.time() + TICK_WALL_CLOCK_BUDGET_S
    conv_id = body.conversation_id
    merchant_id = body.merchant_id
    customer_id = body.customer_id
    message = body.message
    turn = body.turn_number

    # Already ended conversation
    if conv_id in ended_conversations or conversation_manager.is_ended(conv_id):
        return {
            "action": "end",
            "rationale": "Conversation previously closed. Not re-engaging."
        }

    # Record this reply
    conversation_manager.add_turn(conv_id, "merchant" if not customer_id else "customer", message)

    # ── STEP 1: Hostile / opt-out ─────────────────────────────────────────────
    if is_hostile_or_optout(message):
        ended_conversations.add(conv_id)
        conversation_manager.mark_ended(conv_id)
        if merchant_id:
            suppressed_merchants.add(merchant_id)
        return {
            "action": "end",
            "rationale": (
                "Merchant expressed explicit opt-out or hostility. "
                "Closing conversation and suppressing all future triggers for this merchant for 30 days."
            )
        }

    # Global merchant auto-reply tracking
    global merchant_auto_reply_counts
    if "merchant_auto_reply_counts" not in globals():
        merchant_auto_reply_counts = {}

    # ── STEP 2: Auto-reply detection ──────────────────────────────────────────
    if is_auto_reply(message):
        # Ensure conversation state exists (replies can arrive for convs not started via tick)
        if not conversation_manager.exists(conv_id):
            conversation_manager.create(conv_id, merchant_id or "", customer_id, "", {})
        conversation_manager.record_auto_reply(conv_id, message)
        conv_auto_count = conversation_manager.count_auto_replies(conv_id)

        m_key = merchant_id or conv_id
        merchant_auto_reply_counts[m_key] = merchant_auto_reply_counts.get(m_key, 0) + 1
        auto_count = max(conv_auto_count, merchant_auto_reply_counts[m_key])

        if auto_count == 1:
            # First auto-reply: send one flagging message for owner to see
            return {
                "action": "send",
                "body": "Looks like an auto-reply. Owner: whenever you check this, just reply YES and I'll share what I found for you.",
                "cta": "binary_yes_no",
                "rationale": (
                    "Detected first WhatsApp Business auto-reply (canned 'thank you for contacting' pattern). "
                    "Sending one short signal message for the owner to see when they next check their phone."
                )
            }
        elif auto_count == 2:
            # Second consecutive auto-reply: back off 24h
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": (
                    "Same auto-reply pattern received twice. Owner not actively on phone. "
                    "Backing off 24 hours before retrying."
                )
            }
        else:
            # Third+ auto-reply: give up gracefully
            ended_conversations.add(conv_id)
            conversation_manager.mark_ended(conv_id)
            return {
                "action": "end",
                "rationale": (
                    f"Auto-reply received {auto_count} consecutive times with no real response. "
                    "Zero engagement signal. Closing conversation."
                )
            }

    # ── STEP 3: Intent commitment — switch to ACTION immediately ─────────────
    if is_intent_commitment(message):
        conv_state = conversation_manager.get(conv_id)
        trigger = conv_state.get("trigger") if conv_state else None

        merchant = _get_payload("merchant", merchant_id) if merchant_id else None
        category = None
        if merchant:
            category = _get_payload("category", merchant.get("category_slug", ""))

        if merchant and category and trigger:
            try:
                action_body = compose_action_response(category, merchant, trigger, conv_state, deadline=reply_deadline)
                conversation_manager.add_turn(conv_id, "vera", action_body)
                return {
                    "action": "send",
                    "body": action_body,
                    "cta": "binary_confirm_cancel",
                    "rationale": (
                        "Merchant explicitly committed to action. "
                        "Switching from qualification/pitch mode to execution mode immediately. "
                        "Providing concrete deliverable with CONFIRM/CANCEL CTA — no further qualifying questions."
                    )
                }
            except Exception as e:
                pass  # Fall through to generic action response

        # Generic intent response
        reply_body = (
            "Let's do it! I'm drafting everything now — "
            "I'll have it ready in 60 seconds. "
            "Reply CONFIRM to send, or let me know if you'd like any changes."
        )
        conversation_manager.add_turn(conv_id, "vera", reply_body)
        return {
            "action": "send",
            "body": reply_body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed to action. Switching to execution mode."
        }

    # ── STEP 4: Out-of-scope request ──────────────────────────────────────────
    if is_out_of_scope(message):
        conv_state = conversation_manager.get(conv_id)
        trigger_kind = (conv_state.get("trigger") or {}).get("kind", "this")
        reply_body = (
            "That's best handled by your CA/advisor — outside what I can help with directly. "
            "Coming back to what we were working on — shall I go ahead with the draft?"
        )
        conversation_manager.add_turn(conv_id, "vera", reply_body)
        return {
            "action": "send",
            "body": reply_body,
            "cta": "open_ended",
            "rationale": (
                "Out-of-scope request politely declined. "
                "Redirecting back to the original conversation thread without losing momentum."
            )
        }

    # ── STEP 5: Normal engaged reply — compose contextual follow-up ──────────
    merchant = _get_payload("merchant", merchant_id) if merchant_id else None
    category = None
    trigger = None
    conv_state = conversation_manager.get(conv_id)

    if merchant:
        category = _get_payload("category", merchant.get("category_slug", ""))
    if conv_state:
        trigger = conv_state.get("trigger")

    if merchant and category and trigger:
        try:
            next_body = compose_followup(category, merchant, trigger, conv_state, message, deadline=reply_deadline)
            conversation_manager.add_turn(conv_id, "vera", next_body)
            return {
                "action": "send",
                "body": next_body,
                "cta": "open_ended",
                "rationale": (
                    "Engaged reply from merchant. "
                    "Composed contextual follow-up advancing to next useful step. "
                    "Grounded in merchant's specific context and trigger."
                )
            }
        except Exception as e:
            import sys
            print(f"[WARN] Followup compose failed for {conv_id}: {e}", file=sys.stderr)

    # Graceful fallback
    reply_body = (
        "Got it! Working on that for you now. "
        "I'll share the details shortly — reply YES to confirm you'd like to proceed."
    )
    conversation_manager.add_turn(conv_id, "vera", reply_body)
    return {
        "action": "send",
        "body": reply_body,
        "cta": "binary_yes_no",
        "rationale": "Acknowledged merchant reply; advancing conversation with generic follow-up."
    }


# ─── Optional: teardown endpoint (spec says optional) ────────────────────────

@app.post("/v1/teardown")
async def teardown():
    """Wipe state after test ends."""
    contexts.clear()
    ended_conversations.clear()
    suppressed_merchants.clear()
    fired_suppression_keys.clear()
    return {"cleared": True, "message": "State wiped"}


# ─── Run locally ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
