"""
Inference script for the Store Manager environment.

Runs an LLM agent through all three difficulty tasks (easy → medium → hard)
and reports a deterministic grade (0.0–1.0) per task.

Usage:
    python inference.py
    python inference.py --url http://localhost:8000

Required environment variables:
    HF_TOKEN       API key for the LLM provider (Hugging Face token or OpenAI key)
    API_BASE_URL   Base URL of the LLM API  (e.g. https://api-inference.huggingface.co/v1/)
    MODEL_NAME     Model identifier          (e.g. meta-llama/Llama-3.1-8B-Instruct)

Optional environment variables:
    STORE_ENV_URL  Base URL of the running Store Manager server (default: http://localhost:8000)

Structured log format emitted to stdout:
    [START] {"task": "...", "seed": N, "model": "...", "max_steps": N, "profit_target": N}
    [STEP]  {"task": "...", "step": N, "action": {...}, "reward": ..., "cumulative_profit": ...}
    [END]   {"task": "...", "cumulative_profit": ..., "score": ..., "profit_target": ...}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

try:
    import websocket  # websocket-client library
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websocket-client", "-q"])
    import websocket  # noqa: F811

try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
    from openai import OpenAI

# Add project root to path so tasks.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS, grade


# ── Prompt helpers ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert retail store manager optimizing profit.

Each step you choose EXACTLY ONE action. There are three action types:

── 1. DISCOUNT ──────────────────────────────────────────────────────────────────
Apply a price discount to a product for this step only (ephemeral).
Discounts boost customer pick probability and clear near-expiry stock.
  {"action_type": "discount", "product_id": <int>, "discount_pct": <0|10|20|50>}

Discount tiers and effect:
  0%  -> no change   | 10% -> +10% relative boost | 20% -> +20% boost | 50% -> +50% boost

PRICE FLOOR RULES (selling below cost is penalized unless near expiry):
  In(steps) >= 4  -> floor = cost_price   (no below-cost selling allowed)
  In(steps) == 3  -> floor = 70% of cost  (max 30% loss/unit)
  In(steps) == 2  -> floor = 40% of cost  (max 60% loss/unit)
  In(steps) == 1  -> floor = 10% of cost  (max 90% loss/unit — clear it!)

  Selling below cost when In(steps) >= 4 incurs an EXTRA penalty equal to the
  per-unit loss × units sold. Avoid below-cost discounts unless near expiry.

When to discount: near-expiry items (avoid expiry penalty = remaining_qty x cost),
high-margin products (Margin > $0) to drive volume, or oversupplied items.
Near expiry (In <= 3): aggressive discounts are worthwhile to avoid the full expiry penalty.

── 2. RESTOCK ───────────────────────────────────────────────────────────────────
Place a reorder for a product. Stock arrives after restock_lead_time steps.
Costs restock_cost_per_unit per unit (upfront, this step). One order per product
at a time -- wait for delivery before reordering the same product.
  {"action_type": "restock", "product_id": <int>, "restock_quantity": <int>}

When to restock: low stock on a high-demand product, enough steps left before expiry
to sell through the restocked units, and available budget.

── 3. PLACE ─────────────────────────────────────────────────────────────────────
Move a product to a different shelf zone. Costs a flat placement fee ($3.00).
  {"action_type": "place", "product_id": <int>, "target_zone": <1|2|3>}

Zone multipliers on pick probability:
  Zone 1 (premium/entrance) -> x1.5   [capacity: 2 products max]
  Zone 2 (standard)         -> x1.0   [capacity: 4 products max]
  Zone 3 (back/low-traffic) -> x0.6   [unlimited]

When to place: move high-value or near-expiry products to zone 1 to increase
their pick probability. Move slow-movers to zone 3.

── COSTS CHARGED EVERY STEP ─────────────────────────────────────────────────────
- Holding cost: each unit in stock costs holding_cost_rate x cost_price per step.
  Minimize idle inventory by balancing restocks with sales velocity.
- Expiry penalty: remaining_qty x cost_price for each expired product.

Goal: maximize cumulative profit = revenue - COGS - expiry_penalty - holding_cost
      - placement_cost - restock_cost.

Respond ONLY with a single JSON object (one of the three formats above).
No explanation, no extra text -- just the JSON.
"""


def _format_inventory(obs: dict) -> str:
    """Format inventory state as a readable table for the LLM prompt."""
    # JSON serializes int dict keys as strings; coerce back to int for lookup
    zm_raw = obs.get("zone_multipliers", {})
    zc_raw = obs.get("zone_capacity", {})
    zo_raw = obs.get("zone_occupancy", {})
    zm = {int(k): v for k, v in zm_raw.items()} if zm_raw else {1: 1.5, 2: 1.0, 3: 0.6}
    zc = {int(k): v for k, v in zc_raw.items()} if zc_raw else {1: 2, 2: 4, 3: 99}
    zo = {int(k): v for k, v in zo_raw.items()} if zo_raw else {1: 0, 2: 0, 3: 0}

    lines = [
        f"Step {obs['current_step'] + 1}/{obs['max_steps']} "
        f"| Steps remaining: {obs['steps_remaining']} "
        f"| Cumulative profit: ${obs['cumulative_profit']:.2f}",
        f"Task difficulty: {obs.get('task_name', 'unknown')}",
        f"Holding cost this step: ${obs.get('holding_cost_this_step', 0.0):.4f}",
        "",
        "Zone info: "
        + "  ".join(
            f"Zone {z}: x{zm.get(z, 1.0)} ({zo.get(z, 0)}/{zc.get(z, '?')} slots)"
            for z in [1, 2, 3]
        ),
        "",
        f"{'ID':<4} {'Name':<10} {'Cost':>6} {'Price':>7} {'Margin':>7} {'Qty':>5} "
        f"{'In(steps)':>9} {'Zone':>5} {'PickProb':>9} {'PopMult':>8} {'HoldRate':>9} "
        f"{'RestockCost':>11} {'Lead':>5} {'PendingQty':>10} {'Active':>7}",
        "-" * 122,
    ]
    for p in obs["inventory"]:
        pending_str = (
            f"{p['pending_restock_quantity']}@s{p['pending_restock_arrives_at_step']}"
            if p.get("pending_restock_quantity", 0) > 0
            else "-"
        )
        margin = p.get("margin_per_unit", round(p["selling_price"] - p["cost_price"], 4))
        steps_left = p.get("steps_until_expiry", p.get("expiry_step", 0) - obs.get("current_step", 0))
        lines.append(
            f"{p['product_id']:<4} {p['name']:<10} "
            f"${p['cost_price']:>5.2f} ${p['selling_price']:>6.2f} "
            f"${margin:>+6.2f} "
            f"{p['quantity']:>5} "
            f"{steps_left:>9} "
            f"{p.get('zone', 2):>5} "
            f"{p['effective_pick_prob']:>9.4f} "
            f"{p.get('popularity_multiplier', 1.0):>8.3f} "
            f"{p.get('holding_cost_rate', 0.0):>9.4f} "
            f"${p.get('restock_cost_per_unit', 0.0):>10.2f} "
            f"{p.get('restock_lead_time', 2):>5} "
            f"{pending_str:>10} "
            f"{'YES' if p['is_active'] else 'no':>7}"
        )

    if obs.get("last_expired_products"):
        lines.append(f"\nJust expired: {', '.join(obs['last_expired_products'])}")

    return "\n".join(lines)


def _parse_action(text: str, active_ids: list[int]) -> dict:
    """Extract a valid action from LLM output; fall back to safe default."""
    def _validate(data: dict) -> dict | None:
        action_type = data.get("action_type", "discount")
        pid = int(data.get("product_id", active_ids[0]))
        if pid not in active_ids:
            return None

        if action_type == "discount":
            disc = int(data.get("discount_pct", 0))
            if disc in (0, 10, 20, 50):
                return {"action_type": "discount", "product_id": pid, "discount_pct": disc}

        elif action_type == "restock":
            qty = data.get("restock_quantity")
            if qty is not None and int(qty) > 0:
                return {"action_type": "restock", "product_id": pid, "restock_quantity": int(qty)}

        elif action_type == "place":
            zone = data.get("target_zone")
            if zone is not None and int(zone) in (1, 2, 3):
                return {"action_type": "place", "product_id": pid, "target_zone": int(zone)}

        return None

    # Try direct JSON parse
    try:
        result = _validate(json.loads(text.strip()))
        if result:
            return result
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Try extracting JSON from surrounding text
    match = re.search(r'\{[^{}]+\}', text)
    if match:
        try:
            result = _validate(json.loads(match.group(0)))
            if result:
                return result
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # Safe fallback
    return {"action_type": "discount", "product_id": active_ids[0], "discount_pct": 0}


# ── WebSocket helpers ──────────────────────────────────────────────────────────

def _ws_url(http_url: str) -> str:
    """Convert http(s) URL to ws(s)."""
    return http_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"


def _recv_json(ws: websocket.WebSocket) -> dict:
    """Receive one WebSocket message and parse as JSON."""
    return json.loads(ws.recv())


def _unpack(msg: dict) -> dict:
    """
    Unpack a WebSocket message into a flat observation dict.

    The server sends: {"type": "observation", "data": {"observation": {...}, "done": bool, "reward": float}}
    This extracts the inner observation and injects done/reward at the top level.
    """
    data = msg.get("data", msg)
    obs = data.get("observation", data)
    obs["done"] = data.get("done", obs.get("done", False))
    obs["reward"] = data.get("reward", obs.get("reward", 0.0))
    return obs


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    ws_url: str,
    task_name: str,
    client: OpenAI,
    model: str,
) -> tuple[float, float]:
    """
    Run a full episode for `task_name` using the LLM agent.

    Returns:
        (cumulative_profit, score)
    """
    task = TASKS[task_name]

    ws = websocket.create_connection(ws_url, timeout=60)
    cumulative = 0.0
    score = 0.0
    try:
        # ── Reset ──────────────────────────────────────────────────────────────
        ws.send(json.dumps({"type": "reset", "data": {"task": task_name, "seed": task.seed}}))
        obs = _unpack(_recv_json(ws))

        print(
            f"[START] {json.dumps({'task': task_name, 'seed': task.seed, 'model': model, 'max_steps': task.max_steps, 'profit_target': task.profit_target})}",
            flush=True,
        )

        # ── Episode loop ───────────────────────────────────────────────────────
        step_num = 0
        while True:
            active_ids = [
                p["product_id"]
                for p in obs.get("inventory", [])
                if p.get("is_active") and p.get("quantity", 0) > 0
            ]
            if not active_ids or obs.get("done", False):
                break

            # Build LLM prompt
            user_msg = _format_inventory(obs)

            # Call LLM
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=64,
            )
            llm_text = response.choices[0].message.content or ""
            action = _parse_action(llm_text, active_ids)

            # Send action to environment
            ws.send(json.dumps({"type": "step", "data": action}))
            obs = _unpack(_recv_json(ws))

            step_num += 1
            print(
                f"[STEP] {json.dumps({'task': task_name, 'step': step_num, 'action': action, 'reward': round(obs.get('reward', 0.0) or 0.0, 4), 'cumulative_profit': round(obs.get('cumulative_profit', 0.0), 4), 'holding_cost': round(obs.get('holding_cost_this_step', 0.0), 4), 'placement_cost': obs.get('placement_cost_this_step', 0.0), 'restock_cost': round(obs.get('restock_cost_this_step', 0.0), 4), 'unjustified_penalty': round(obs.get('unjustified_discount_penalty', 0.0), 4)})}",
                flush=True,
            )

            if obs.get("done", False):
                break

        cumulative = obs.get("cumulative_profit", 0.0)
        score = grade(task_name, cumulative)

        return cumulative, score

    finally:
        ws.close()
        print(
            f"[END] {json.dumps({'task': task_name, 'cumulative_profit': round(cumulative, 4), 'score': round(score, 4), 'profit_target': task.profit_target})}",
            flush=True,
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM agent on Store Manager tasks")
    parser.add_argument(
        "--url",
        default=os.getenv("STORE_ENV_URL", "http://localhost:8000"),
        help="Base URL of the running Store Manager server (env: STORE_ENV_URL)",
    )
    args = parser.parse_args()

    # ── Mandatory env vars (per hackathon spec) ────────────────────────────────
    hf_token = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not hf_token:
        print(
            "ERROR: HF_TOKEN environment variable not set. "
            "Set it to your Hugging Face token or OpenAI API key.",
            file=sys.stderr,
        )
        sys.exit(1)

    api_base_url = os.getenv("API_BASE_URL")
    model = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Build OpenAI-compatible client
    client_kwargs: dict = {"api_key": hf_token}
    if api_base_url:
        client_kwargs["base_url"] = api_base_url

    client = OpenAI(**client_kwargs)
    ws_url = _ws_url(args.url)

    results = {}
    for task_name in ["easy", "medium", "hard", "expert"]:
        try:
            profit, score = run_episode(ws_url, task_name, client, model)
            results[task_name] = {"profit": round(profit, 4), "score": round(score, 4)}
        except Exception as exc:
            print(f"[ERROR] task={task_name} error={exc}", file=sys.stderr, flush=True)
            results[task_name] = {"profit": 0.0, "score": 0.0}
        # Brief pause between tasks to avoid overwhelming the server
        time.sleep(1)

    # Final summary
    print("\n=== Summary ===")
    for task_name, r in results.items():
        print(f"  {task_name:<8} profit={r['profit']:>10.2f}  score={r['score']:.4f}")
    overall = sum(r["score"] for r in results.values()) / len(results)
    print(f"  {'AVERAGE':<8}                   score={overall:.4f}")


if __name__ == "__main__":
    main()
