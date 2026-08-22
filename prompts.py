"""
Vera Message Engine — Prompt Templates
Trigger-kind-dispatched prompts optimized for each judge scoring dimension.
All prompts enforce: no fabrication, category voice, merchant specificity,
clear trigger rationale, single CTA.
"""

import json
from typing import Dict, Any, Optional


# ─── Category Voice Guides ────────────────────────────────────────────────────

CATEGORY_VOICE = {
    "dentists": {
        "tone": "peer/clinical — collegial doctor-to-doctor register",
        "salutation": "Dr. {first_name}",
        "style": "Technical dental vocabulary is welcome (fluoride varnish, scaling, bruxism, IOPA, RVG, caries, periodontal). Peer tone, not promotional. Source-cite research. No overclaims.",
        "taboos": "Never say: guaranteed, 100% safe, completely cure, miracle, best in city",
        "cta_style": "Open-ended invitation or single binary YES/STOP. For patient-facing: multi-choice slots.",
        "examples": [
            "Worth a look — JIDA Oct 2026 p.14",
            "This one likely affects your high-risk adult cohort",
        ],
        "emojis": "Minimal — 🦷 only for patient-facing messages"
    },
    "salons": {
        "tone": "warm/practical — fellow operator, friendly and efficient",
        "salutation": "{first_name}",
        "style": "Warm but efficient. Names matter. Reference the service and price directly. Seasonal trends are fair game.",
        "taboos": "No medical claims. No 'best in city'.",
        "cta_style": "Low-friction: binary YES/book or open-ended question. Emojis OK.",
        "examples": [
            "Spotted: bridal-trial searches in Kapra +28% this week",
            "What's most asked-for this week at Studio11?",
        ],
        "emojis": "💇 💍 ✨ — use sparingly but warmly"
    },
    "restaurants": {
        "tone": "operator-to-operator — practical, numbers-first",
        "salutation": "{first_name}",
        "style": "Speak like a smart business partner. Use restaurant vocabulary: covers, AOV, delivery share, dine-in. Counter-intuitive data earns trust.",
        "taboos": "No 'amazing deals'. No empty hype.",
        "cta_style": "Binary action or open-ended continuation. Keep it brief.",
        "examples": [
            "Saturday IPL matches usually shift -12% restaurant covers (people watch at home)",
            "Your weekday thali is doing 18 orders/day — corporate-bulk version ready to draft",
        ],
        "emojis": "🍕 🍛 — only when contextually appropriate"
    },
    "gyms": {
        "tone": "coach/motivational — data-informed, no-shame",
        "salutation": "{first_name}",
        "style": "Sound like a knowledgeable coach, not a salesperson. Use fitness vocabulary: members, churn, retention, trial-to-paid, seasonal dip, HIIT. For customer messages: warm, no-shame, no guilt-trip.",
        "taboos": "No guilt-trip language. No 'you should work out more'.",
        "cta_style": "Single binary commitment. For winback: no-commitment trial offer.",
        "examples": [
            "April-June lull is normal — every metro gym sees -25 to -35%",
            "57 days — happens to most members at some point, no judgment",
        ],
        "emojis": "💪 🏋️ — use for customer-facing, sparingly"
    },
    "pharmacies": {
        "tone": "trustworthy/precise — pharmacist-to-pharmacist",
        "salutation": "{first_name}",
        "style": "Precise molecule names, batch numbers, dose thresholds. No drama. Trustworthy and calm. For senior customers: respectful, clear, namaste opening.",
        "taboos": "No alarming language. No self-diagnosis claims.",
        "cta_style": "Binary confirm/call for customer messages. Open-ended for merchant.",
        "examples": [
            "Sub-potency, no safety risk, but customers should be informed",
            "Ramesh, urgent: voluntary recall on 2 atorvastatin batches (AT2024-1102, AT2024-1108)",
        ],
        "emojis": "💊 🏥 — only for customer-facing messages"
    }
}


# ─── Trigger-kind Prompt Variants ────────────────────────────────────────────

TRIGGER_INSTRUCTIONS = {
    "research_digest": """
TRIGGER TYPE: Research Digest
Lead with the research finding + source. Connect it to THIS merchant's specific patient/customer cohort.
Offer to do something useful (pull abstract, draft patient-ed content, update GBP).
Source citation at the end is mandatory. No speculation beyond the digest.
""",
    "regulation_change": """
TRIGGER TYPE: Regulation/Compliance Change  
Lead with the urgency and deadline. State the specific change (dose, limit, rule).
Tell the merchant exactly what action to take before the deadline.
Sound like a knowledgeable colleague alerting them, not a compliance robot.
""",
    "cde_opportunity": """
TRIGGER TYPE: CDE / Webinar Opportunity
Share the topic, speaker, credits, date, and cost (free or fee).
Make it relevant to this merchant's case-mix. Low friction — one question to confirm interest.
""",
    "recall_due": """
TRIGGER TYPE: Recall Due (customer-facing, send_as = merchant_on_behalf)
Address the customer by name. State how long since last visit. Name the service due.
Offer 2 specific time slots that match their preference. Include the price.
Language must match customer's preference. Warm, not clinical. No medical claims.
""",
    "chronic_refill_due": """
TRIGGER TYPE: Chronic Refill Due (customer-facing, send_as = merchant_on_behalf)
Name all molecules/medicines. State the exact date they run out.
Show total with any discounts pre-applied. Offer confirmed delivery.
For senior customers: Namaste opening, speak via son/family if channel says so.
""",
    "perf_spike": """
TRIGGER TYPE: Performance Spike
Tell them what went up and by exactly how much. Hypothesize the likely driver from context.
Recommend one action to capitalize on the spike while it's live.
Show what their peer benchmark is for comparison.
""",
    "perf_dip": """
TRIGGER TYPE: Performance Dip
Be honest about what dropped. Do NOT catastrophize. 
If there's a known reason (seasonal, competitor), name it and reframe.
Give one concrete action to arrest the dip.
""",
    "seasonal_perf_dip": """
TRIGGER TYPE: Seasonal Performance Dip
This is an EXPECTED seasonal drop — NOT a crisis. Lead with that reassurance.
Give the typical industry range for this season. Recommend they save ad spend for peak.
Suggest a retention action for existing members/customers instead.
""",
    "milestone_reached": """
TRIGGER TYPE: Milestone Reached
Celebrate the achievement with a specific number. Make it feel earned, not hollow.
Immediately propose the next logical growth action to maintain momentum.
""",
    "festival_upcoming": """
TRIGGER TYPE: Festival Upcoming
Name the festival, days until it, and why it matters for THIS category.
Recommend a specific offer or content action that works for this merchant's catalog.
Time-bound: "before the rush" framing.
""",
    "ipl_match_today": """
TRIGGER TYPE: IPL Match Today
Check: is it a weeknight (increases footfall) or weekend (people watch at home)?
Weeknight: push delivery/dine-in promo for match crowd.
Weekend: COUNTER-INTUITIVE — skip match promo, use existing offer as delivery-only special instead.
Name the exact match, time, and venue.
""",
    "competitor_opened": """
TRIGGER TYPE: Competitor Opened Nearby
Lead with curiosity/voyeur hook — make them want to know.
Share what you know about the competitor (distance, their offer).
Recommend a defensive action: reinforce their own differentiator, not slash prices.
Never name the competitor in a way that feels threatening.
""",
    "review_theme_emerged": """
TRIGGER TYPE: Review Theme Emerged
State the theme, how many mentions in 30 days, and the trend direction.
Quote a real customer line if available. 
Recommend one operational fix, not just "respond to reviews".
""",
    "curious_ask_due": """
TRIGGER TYPE: Curious Ask (weekly check-in)
Ask ONE interesting question about their business this week.
Offer to turn their answer into something useful (GBP post, reply template, WhatsApp story).
Low-stakes, low-effort ask. Conversational tone. Short message.
""",
    "active_planning_intent": """
TRIGGER TYPE: Active Planning Intent (merchant asked for something)
The merchant already said YES to exploring this idea. DO NOT re-qualify.
Immediately present a concrete draft, plan, or artifact they can review.
Give them something to react to, not another question.
""",
    "winback_eligible": """
TRIGGER TYPE: Merchant Winback (subscription lapsed)
Acknowledge the gap without dwelling on it. Lead with what they've been missing.
Show concrete performance impact since lapse (dip in calls/views).
Offer one specific action to re-activate. Single binary CTA.
""",
    "dormant_with_vera": """
TRIGGER TYPE: Dormant with Vera (merchant hasn't replied in N days)
Re-engage with a genuinely useful hook, not a "just checking in" message.
Use a curiosity lever or share something new and relevant to their category.
Very short. No guilt. No re-introduction.
""",
    "customer_lapsed_soft": """
TRIGGER TYPE: Customer Lapsed Soft (3-6 months, send_as = merchant_on_behalf)
Warm, no-shame re-engagement. Address by name. Mention elapsed time casually.
Reference their previous service/goal. Offer something specific with no commitment.
Single binary CTA: "Reply YES" or "free trial".
""",
    "customer_lapsed_hard": """
TRIGGER TYPE: Customer Lapsed Hard (6+ months, send_as = merchant_on_behalf)
Warm, no-shame, no-guilt. "Happens to most members at some point."
Reference their previous goal/history. Offer a no-commitment trial.
Name a specific new offering that matches their previous focus.
""",
    "supply_alert": """
TRIGGER TYPE: Supply Alert / Drug Recall
State the specific molecules, batch numbers, and manufacturer.
Quantify exactly how many of THIS merchant's customers are affected (from customer_aggregate).
Offer to draft the customer WhatsApp + the replacement-pickup workflow in one step.
Urgency: yes. Alarm: no. "Sub-potency, no safety risk" framing where appropriate.
""",
    "category_seasonal": """
TRIGGER TYPE: Category Seasonal Trend
State the specific trend items with exact percentage shifts.
Recommend a concrete shelf/inventory action based on the trends.
Give the merchant a simple way to act on it today.
""",
    "renewal_due": """
TRIGGER TYPE: Subscription Renewal Due
Show the concrete business impact of the subscription (performance metrics, leads, reviews).
Frame renewal as protecting their momentum, not just paying a bill.
State days remaining clearly. Single binary CTA.
""",
    "gbp_unverified": """
TRIGGER TYPE: Google Business Profile Unverified
State the estimated uplift from verification (if in trigger payload).
Explain the verification path simply. Offer to guide them through it.
One-step ask: "Want me to start the verification now?"
""",
    "trial_followup": """
TRIGGER TYPE: Trial Followup (customer just completed a trial)
The experience is fresh — capitalize on it. Address by name.
Reference the specific trial they did. Offer the next session slot.
No-pressure, curiosity-based: "Want to hold a spot?"
""",
    "perf_spike": """
TRIGGER TYPE: Performance Spike
State what spiked (views/calls/CTR) and the exact percentage. 
Attribute to a likely driver if available. Recommend capitalizing on it now.
""",
    "wedding_package_followup": """
TRIGGER TYPE: Wedding Package / Bridal Followup (customer-facing)
Count down to wedding day (exact days). Reference the trial they completed.
Recommend the next logical preparation step with timing urgency.
Include price and preferred slot. Warm, excited tone (appropriate for weddings).
""",
}


# ─── Output Format Instructions ───────────────────────────────────────────────

OUTPUT_FORMAT = """
Return ONLY this JSON (no markdown, no explanation outside):
{
  "body": "<the WhatsApp message body — no URLs, no fake data>",
  "cta": "<one of: binary_yes_no | binary_confirm_cancel | open_ended | multi_choice_slot | none>",
  "send_as": "<vera | merchant_on_behalf>",
  "suppression_key": "<use the suppression_key from trigger, or derive logically>",
  "rationale": "<1-2 sentences: why this message now, what compulsion levers used>"
}
"""


# ─── Main Prompt Builders ─────────────────────────────────────────────────────

def _context_summary(category: Dict, merchant: Dict, trigger: Dict,
                     customer: Optional[Dict]) -> str:
    """Build a compact, fact-rich context summary for the LLM."""
    cat_slug = category.get("slug", "unknown")
    cat_voice = CATEGORY_VOICE.get(cat_slug, {})

    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    sub = merchant.get("subscription", {})
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    conv_history = merchant.get("conversation_history", [])
    signals = merchant.get("signals", [])
    customer_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])

    # Category digest - pull relevant items
    digest = category.get("digest", [])
    peer_stats = category.get("peer_stats", {})

    # Find the specific digest item referenced in trigger if any
    trg_payload = trigger.get("payload", {})
    relevant_digest = None
    if trg_payload.get("top_item_id"):
        tid = trg_payload["top_item_id"]
        relevant_digest = next((d for d in digest if d.get("id") == tid), None)
    if not relevant_digest and digest:
        relevant_digest = digest[0]

    # Seasonal beats
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])

    lines = [
        f"=== CATEGORY: {cat_slug.upper()} ===",
        f"Voice: {cat_voice.get('tone', '')}",
        f"Salutation: {cat_voice.get('salutation', '')}",
        f"Style: {cat_voice.get('style', '')}",
        f"Taboos: {cat_voice.get('taboos', '')}",
        f"CTA style: {cat_voice.get('cta_style', '')}",
        "",
        f"Peer benchmarks: CTR avg={peer_stats.get('avg_ctr', '?')}, "
        f"calls/30d={peer_stats.get('avg_calls_30d', '?')}, "
        f"views/30d={peer_stats.get('avg_views_30d', '?')}, "
        f"retention_6mo={peer_stats.get('retention_6mo_pct', '?')}",
    ]

    if relevant_digest:
        lines += [
            "",
            f"=== RELEVANT DIGEST ITEM ===",
            f"Title: {relevant_digest.get('title', '')}",
            f"Source: {relevant_digest.get('source', '')}",
            f"Summary: {relevant_digest.get('summary', '')}",
            f"Trial N: {relevant_digest.get('trial_n', '')}",
            f"Patient segment: {relevant_digest.get('patient_segment', '')}",
            f"Actionable: {relevant_digest.get('actionable', '')}",
        ]

    if seasonal:
        lines.append(f"Seasonal: {json.dumps(seasonal)}")
    if trends:
        lines.append(f"Trends: {json.dumps(trends[:3])}")

    lines += [
        "",
        f"=== MERCHANT ===",
        f"Name: {identity.get('name', '')}",
        f"Owner first name: {identity.get('owner_first_name', '')}",
        f"City: {identity.get('city', '')}, Locality: {identity.get('locality', '')}",
        f"Languages: {identity.get('languages', [])}",
        f"Verified GBP: {identity.get('verified', '?')}",
        f"Subscription: {sub.get('status', '?')} | Plan: {sub.get('plan', '?')} | "
        f"Days remaining: {sub.get('days_remaining', sub.get('days_since_expiry', '?'))}",
        "",
        f"Performance (30d): views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, "
        f"directions={perf.get('directions', '?')}, CTR={perf.get('ctr', '?')}, leads={perf.get('leads', '?')}",
        f"7d delta: views {delta.get('views_pct', 'n/a')}, calls {delta.get('calls_pct', 'n/a')}, "
        f"CTR {delta.get('ctr_pct', 'n/a')}",
        f"Peer CTR: {peer_stats.get('avg_ctr', '?')} | "
        f"{'BELOW' if float(perf.get('ctr', 0) or 0) < float(peer_stats.get('avg_ctr', 1) or 1) else 'ABOVE'} peer median",
        "",
        f"Active offers: {[o.get('title') for o in active_offers] or 'NONE'}",
        f"Signals: {signals}",
        f"Customer aggregate: {json.dumps(customer_agg)}",
    ]

    if review_themes:
        lines.append(f"Review themes: {json.dumps(review_themes)}")

    if conv_history:
        lines.append("")
        lines.append("=== RECENT CONVERSATION HISTORY ===")
        for turn in conv_history[-3:]:
            lines.append(f"[{turn.get('from', '?')}]: {turn.get('body', '')[:120]}")
            lines.append(f"  engagement: {turn.get('engagement', '?')}")

    lines += [
        "",
        f"=== TRIGGER ===",
        f"Kind: {trigger.get('kind', '?')}",
        f"Scope: {trigger.get('scope', '?')}",
        f"Source: {trigger.get('source', '?')}",
        f"Urgency: {trigger.get('urgency', '?')} / 5",
        f"Suppression key: {trigger.get('suppression_key', '?')}",
        f"Expires: {trigger.get('expires_at', '?')}",
        f"Payload: {json.dumps(trg_payload, indent=2)}",
    ]

    if customer:
        cust_identity = customer.get("identity", {})
        cust_rel = customer.get("relationship", {})
        cust_prefs = customer.get("preferences", {})
        cust_consent = customer.get("consent", {})
        lines += [
            "",
            f"=== CUSTOMER (for merchant_on_behalf send) ===",
            f"Name: {cust_identity.get('name', '')}",
            f"Language preference: {cust_identity.get('language_pref', '')}",
            f"Age band: {cust_identity.get('age_band', '')}",
            f"State: {customer.get('state', '?')}",
            f"First visit: {cust_rel.get('first_visit', '?')}",
            f"Last visit: {cust_rel.get('last_visit', '?')}",
            f"Visits total: {cust_rel.get('visits_total', '?')}",
            f"Services received: {cust_rel.get('services_received', [])}",
            f"Lifetime value: ₹{cust_rel.get('lifetime_value', '?')}",
            f"Preferences: {json.dumps(cust_prefs)}",
            f"Consent scope: {cust_consent.get('scope', [])}",
        ]
        # Add extra fields like favourite dish, chronic conditions, wedding date
        if customer.get("identity", {}).get("senior_citizen"):
            lines.append("IMPORTANT: Senior citizen — use respectful tone, Namaste opening")
        if cust_rel.get("favourite_dish"):
            lines.append(f"Favourite dish: {cust_rel['favourite_dish']}")
        if cust_rel.get("chronic_conditions"):
            lines.append(f"Chronic conditions: {cust_rel['chronic_conditions']}")
        if cust_prefs.get("wedding_date"):
            lines.append(f"Wedding date: {cust_prefs['wedding_date']}")

    return "\n".join(lines)


def build_system_prompt(category: Dict, merchant: Dict, trigger: Dict,
                        customer: Optional[Dict]) -> str:
    """Build the system prompt with category voice and hard rules."""
    cat_slug = category.get("slug", "unknown")
    trigger_kind = trigger.get("kind", "generic")
    trigger_scope = trigger.get("scope", "merchant")

    # Get category voice
    voice = CATEGORY_VOICE.get(cat_slug, {})

    # Get trigger-specific instructions
    trigger_instruct = TRIGGER_INSTRUCTIONS.get(
        trigger_kind,
        "Compose a helpful, specific message relevant to the trigger. Lead with WHY NOW."
    )

    if trigger_scope == "customer" and customer:
        cust_name = customer.get("identity", {}).get("name", "there")
        is_senior = customer.get("identity", {}).get("senior_citizen", False)
        expected_salutation = f"Namaste {cust_name}" if is_senior else f"Hi {cust_name}"
        business_name = merchant.get("identity", {}).get("name", "our clinic/business")
        role_description = f"You are drafting an outbound WhatsApp message sent from '{business_name}' directly TO their customer '{cust_name}'. Do NOT mention Vera or MagicPin. Address '{cust_name}' directly."
        send_as_rule = f"send_as MUST be 'merchant_on_behalf'. Recipient is customer '{cust_name}'."
    else:
        owner_name = merchant.get("identity", {}).get("owner_first_name", "")
        salutation_template = voice.get("salutation", "{first_name}")
        expected_salutation = salutation_template.replace("{first_name}", owner_name) if owner_name else "Hi"
        role_description = "You are Vera, magicpin's AI assistant messaging the merchant owner directly."
        send_as_rule = "send_as MUST be 'vera'. Recipient is merchant owner."

    return f"""{role_description}

CATEGORY: {cat_slug}
VOICE: {voice.get('tone', 'professional')}
REGISTER: {voice.get('style', '')}
TABOOS (never use): {voice.get('taboos', '')}
EXACT OPENING SALUTATION: "{expected_salutation}"
CTA STYLE: {voice.get('cta_style', 'single binary')}

TRIGGER-KIND INSTRUCTIONS:
{trigger_instruct}

SEND DIRECTION: {send_as_rule}

ABSOLUTE RULES (violating any = score cap at 5/10):
1. DO NOT fabricate any data or statistics. Only use data, findings, numbers, dates, names, available slots, and offers from the context provided below.
2. If RELEVANT DIGEST ITEM is in context, you MUST use its exact finding, trial size, and source citation (e.g., JIDA Oct 2026, p.14). Never make up marketing stats.
3. DO NOT include URLs of any kind.
4. ONE clear CTA only — at the end of the message.
5. DO NOT re-introduce yourself.
6. NO preambles like "I hope you're well" or "Here's a quick digest".
7. Start directly with the salutation: "{expected_salutation}".
8. Honor language preference — if customer/merchant prefers Hindi-English mix, write in natural code-mix (e.g., 'Apke liye slots ready hain').
9. Match voice to category — dentists get peer clinical tone for merchant, warm clinical for customer.
10. Body must be concise (under 350 characters for customer, under 450 for merchant).
11. Reference actual active offers only — never expired ones.

COMPULSION LEVERS (use 1-3 per message):
- Specificity/verifiability: concrete numbers, dates, source citations
- Loss aversion: "you're missing X" / "before this window closes"
- Social proof: "N similar merchants in your area did Y"
- Effort externalization: "I've drafted X — just say go" / "5-min setup"
- Curiosity: "want to see who?" / "want the full list?"
- Reciprocity: "I noticed Y, thought you'd want to know"
- Single binary commitment: Reply YES / STOP

{OUTPUT_FORMAT}"""


def build_user_prompt(category: Dict, merchant: Dict, trigger: Dict,
                      customer: Optional[Dict]) -> str:
    """Build the user prompt with full context summary."""
    ctx = _context_summary(category, merchant, trigger, customer)
    trigger_kind = trigger.get("kind", "")
    if customer:
        cust_name = customer.get('identity', {}).get('name', 'customer')
        recipient_note = f"- Recipient is customer: {cust_name}. Address them directly. Mention {merchant.get('identity', {}).get('name', 'the business')}. Use available slots from trigger if provided."
    else:
        recipient_note = f"- Recipient is merchant: {merchant.get('identity', {}).get('owner_first_name', '')}."

    return f"""Compose the Vera message for this context.

{ctx}

Remember:
- Lead with the specific trigger: {trigger_kind}
{recipient_note}
- Category: {category.get('slug', '')} — match the voice exactly
- No fabrication, no URLs
- Single CTA at the end

Return the JSON now."""


def build_action_prompt(category: Dict, merchant: Dict, trigger: Dict,
                        conv_state: Optional[Dict]) -> str:
    """Prompt for when merchant commits to action — switch to execution immediately."""
    identity = merchant.get("identity", {})
    first_name = identity.get("owner_first_name", identity.get("name", ""))
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    customer_agg = merchant.get("customer_aggregate", {})
    trigger_kind = trigger.get("kind", "")
    prev_body = conv_state.get("last_body", "") if conv_state else ""

    return f"""The merchant ({first_name}) just committed to taking action. 

Previous Vera message was about: {trigger_kind}
Last message body preview: {prev_body[:150]}

Active offers: {[o.get('title') for o in active_offers]}
Customer aggregate: {json.dumps(customer_agg)}

NOW compose an ACTION message (not a question):
- Tell them exactly what you're doing / drafting right now
- Give a concrete next step with a specific number if possible (e.g., "40 patients", "3 posts")
- End with CONFIRM / CANCEL CTA

Return JSON: {{"body": "...", "cta": "binary_confirm_cancel", "send_as": "vera", 
"suppression_key": "{trigger.get('suppression_key', '')}", "rationale": "..."}}"""


def build_followup_prompt(category: Dict, merchant: Dict, trigger: Dict,
                           conv_state: Optional[Dict],
                           merchant_message: str) -> str:
    """Prompt for continuing a conversation based on merchant's reply."""
    identity = merchant.get("identity", {})
    first_name = identity.get("owner_first_name", identity.get("name", ""))
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    trigger_kind = trigger.get("kind", "")
    last_vera_body = conv_state.get("last_body", "") if conv_state else ""

    return f"""Continue this conversation:

Merchant: {first_name} ({category.get('slug', '')} in {identity.get('locality', '')})
Trigger context: {trigger_kind}
Last Vera message: {last_vera_body[:200]}
Merchant just said: "{merchant_message}"

Active offers: {[o.get('title') for o in active_offers]}

Compose the next Vera message:
- Directly respond to what merchant said
- Advance to the next useful step (don't repeat what you already said)
- Be brief and specific
- One question or CTA max

Return JSON: {{"body": "...", "cta": "open_ended", "send_as": "vera",
"suppression_key": "{trigger.get('suppression_key', '')}", "rationale": "..."}}"""
