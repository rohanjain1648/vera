# Vera Bot — magicpin AI Challenge Submission

## Approach

**Core idea**: Every trigger kind deserves its own prompt variant. A `research_digest` needs clinical source-citation framing. A `recall_due` needs slot-offering patient-context framing. A `perf_dip` needs data-anchored, contrarian reasoning. A generic prompt handles none of these well — so I dispatched on `trigger.kind` and wrote specialized prompts for each of the 22+ trigger kinds in the dataset.

## Architecture

```
POST /v1/context  →  in-memory context store (scope, context_id) → {version, payload}
POST /v1/tick     →  sort triggers by urgency → compose() for each → return actions[]
POST /v1/reply    →  classify reply → route to: end/wait/action/followup handler
GET  /v1/healthz  →  uptime + context counts
GET  /v1/metadata →  bot identity
```

### Composition pipeline (`composer.py`)
1. `compose(category, merchant, trigger, customer?)` called per trigger
2. `prompts.py` builds a **system prompt** (category voice rules, taboos, CTA style, trigger-kind instructions) + **user prompt** (full context summary with real numbers)
3. Groq `llama-3.3-70b-versatile` at temperature=0 for determinism
4. Output parsed as JSON → validated (no URLs, correct send_as, valid CTA)
5. Retry once with simpler prompt on parse failure

### What makes each dimension score well

| Dimension | Design decision |
|---|---|
| **Specificity** | Context summary passes merchant's exact CTR, view counts, peer benchmark, active offer prices, digest trial_n and source citations directly into the prompt |
| **Category fit** | `CATEGORY_VOICE` dict in `prompts.py` — each category has tone, salutation format, vocabulary style, taboos, CTA style, and example phrases |
| **Merchant fit** | Owner first name always used. Active offers only (expired filtered). Language preference passed. Conversation history included in context. |
| **Trigger relevance** | `TRIGGER_INSTRUCTIONS` dict — each trigger kind has dedicated instructions that lead with "WHY NOW" and reference the specific payload fields |
| **Engagement compulsion** | Prompt requires 1-3 compulsion levers: specificity, loss aversion, social proof, effort externalization, curiosity, or reciprocity. Single CTA last. |

### Reply handling (`conversation.py` + `main.py`)
- **Auto-reply**: Pattern-matched against 15+ canned phrases. Turn 1 → signal message. Turn 2 → wait 24h. Turn 3+ → end.
- **Intent commitment**: Detected via regex ("yes", "let's do it", "chalega", "thik hai", etc.). Routes immediately to action composition — no qualifying questions.
- **Hostile / opt-out**: Detected + merchant suppressed for session. Graceful `end`.
- **Out-of-scope**: GST, taxes, legal, loans → politely declined + redirected.
- **Normal**: Full contextual follow-up composed via LLM.

## Model choice

**Groq + llama-3.3-70b-versatile**: Free tier, ~1-3s latency (well within 30s limit), strong instruction following, handles Hindi-English code-mix well.

## Tradeoffs

- In-memory state (no Redis) — suitable for a 60-min test window; would use Redis for production
- Single LLM call per composition — faster than multi-step but slightly less nuanced than chain-of-thought
- Suppression dedup is per-session — if bot restarts, suppression resets (not a concern here)

## What additional context would have helped most

1. **Exact merchant time zone** — needed for precise slot labels in recall messages
2. **Real slot availability** — recall/appointment triggers reference slots but don't always have them; I use trigger payload slots when available, otherwise generic slot labels
3. **Merchant's current Google post cadence** — would enable tighter "you haven't posted in X days" hooks

## Local testing

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Run bot
uvicorn main:app --host 0.0.0.0 --port 8080

# 4. In another terminal, run the judge simulator
cd ..
python judge_simulator.py
```

## Deployment (Render.com)

1. Push `vera_bot/` to a GitHub repo
2. Connect to Render → New Web Service → select the repo
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add env var: `GROQ_API_KEY=<your key>`
6. Deploy → copy the HTTPS URL → submit to magicpin
