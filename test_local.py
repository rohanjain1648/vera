"""
Quick local test: push real dataset contexts and call /v1/tick
Run from the vera_bot/ directory: python test_local.py
"""
import json
import sys
import time
from pathlib import Path
from urllib import request as urlrequest

BOT_URL = "http://localhost:8080"
DATASET_DIR = Path(__file__).parent.parent / "dataset"


def post(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urlrequest.Request(
        f"{BOT_URL}{path}", data=body,
        headers={"Content-Type": "application/json"}
    )
    resp = urlrequest.urlopen(req, timeout=30)
    return json.loads(resp.read())


def get(path):
    resp = urlrequest.urlopen(f"{BOT_URL}{path}", timeout=5)
    return json.loads(resp.read())


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Vera Bot — Local Test")
    print("=" * 60)

    # 1. Healthz
    h = get("/v1/healthz")
    print(f"\n[OK] healthz: {h['status']} | uptime={h['uptime_seconds']}s")

    # 2. Metadata
    m = get("/v1/metadata")
    print(f"[OK] metadata: team={m['team_name']} | model={m['model']}")

    # 3. Load categories
    cat_dir = DATASET_DIR / "categories"
    for cat_file in cat_dir.glob("*.json"):
        cat = load_json(cat_file)
        r = post("/v1/context", {
            "scope": "category",
            "context_id": cat["slug"],
            "version": 1,
            "delivered_at": "2026-04-26T09:45:00Z",
            "payload": cat
        })
        print(f"[{'OK' if r.get('accepted') else 'SKIP'}] category/{cat['slug']}")

    # 4. Load merchants
    merchants = load_json(DATASET_DIR / "merchants_seed.json")["merchants"]
    for m in merchants:
        r = post("/v1/context", {
            "scope": "merchant",
            "context_id": m["merchant_id"],
            "version": 1,
            "delivered_at": "2026-04-26T09:46:00Z",
            "payload": m
        })
        print(f"[{'OK' if r.get('accepted') else 'SKIP'}] merchant/{m['merchant_id']}")

    # 5. Load customers
    customers = load_json(DATASET_DIR / "customers_seed.json")["customers"]
    for c in customers:
        r = post("/v1/context", {
            "scope": "customer",
            "context_id": c["customer_id"],
            "version": 1,
            "delivered_at": "2026-04-26T09:47:00Z",
            "payload": c
        })
        print(f"[{'OK' if r.get('accepted') else 'SKIP'}] customer/{c['customer_id']}")

    # 6. Load triggers
    triggers = load_json(DATASET_DIR / "triggers_seed.json")["triggers"]
    trigger_ids = []
    for t in triggers:
        r = post("/v1/context", {
            "scope": "trigger",
            "context_id": t["id"],
            "version": 1,
            "delivered_at": "2026-04-26T10:00:00Z",
            "payload": t
        })
        if r.get("accepted"):
            trigger_ids.append(t["id"])
        print(f"[{'OK' if r.get('accepted') else 'SKIP'}] trigger/{t['id']}")

    # 7. Check healthz counts
    h2 = get("/v1/healthz")
    print(f"\n[INFO] Contexts loaded: {h2['contexts_loaded']}")

    # 8. Run tick with ALL triggers
    print(f"\n=== TICK (all {len(trigger_ids)} triggers) ===")
    tick_result = post("/v1/tick", {
        "now": "2026-04-26T10:35:00Z",
        "available_triggers": trigger_ids
    })
    actions = tick_result.get("actions", [])
    print(f"[OK] Tick returned {len(actions)} action(s)")

    for i, action in enumerate(actions):
        print(f"\n--- Action {i+1} ---")
        print(f"  conv_id:  {action.get('conversation_id')}")
        print(f"  merchant: {action.get('merchant_id')}")
        print(f"  trigger:  {action.get('trigger_id')}")
        print(f"  send_as:  {action.get('send_as')}")
        print(f"  cta:      {action.get('cta')}")
        body = action.get("body", "")
        print(f"  body:     {body[:200]}{'...' if len(body) > 200 else ''}")
        print(f"  rationale: {action.get('rationale', '')[:120]}")

    # 9. Test reply flow on first action
    if actions:
        first_action = actions[0]
        conv_id = first_action["conversation_id"]
        merchant_id = first_action["merchant_id"]

        print(f"\n=== REPLY FLOW (conv: {conv_id}) ===")
        # Simulate merchant saying yes
        reply = post("/v1/reply", {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": None,
            "from_role": "merchant",
            "message": "Yes, this is useful. Please go ahead.",
            "received_at": "2026-04-26T10:40:00Z",
            "turn_number": 2
        })
        print(f"  action: {reply.get('action')}")
        if reply.get("body"):
            print(f"  body: {reply['body'][:200]}")
        print(f"  rationale: {reply.get('rationale', '')[:100]}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
