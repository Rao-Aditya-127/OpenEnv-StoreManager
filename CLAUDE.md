# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Hackathon Submission Requirements

This project is being submitted to a hackathon. Every change must keep the following requirements satisfied — never regress them.

### Mandatory env variables (inference.py must use exactly these names)

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | API key passed to OpenAI client (`api_key=`) |
| `API_BASE_URL` | LLM base URL passed to OpenAI client (`base_url=`) |
| `MODEL_NAME` | Model identifier used for all LLM calls |

`STORE_ENV_URL` is a separate optional variable for the environment server URL (default: `http://localhost:8000`). Do not confuse it with `API_BASE_URL`.

### Structured log format (exact field names required — do not rename)

```
[START] {"task": "...", "seed": N, "model": "...", "max_steps": N, "profit_target": N}
[STEP]  {"task": "...", "step": N, "action": {...}, "reward": ..., "cumulative_profit": ...}
[END]   {"task": "...", "cumulative_profit": ..., "score": ..., "profit_target": ...}
```

Extra fields in `[STEP]` (e.g. `holding_cost`, `placement_cost`, `restock_cost`) are allowed but the above fields must always be present with exact names.

### Pre-submission checklist

- [ ] `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` read by inference.py ✅
- [ ] `inference.py` in root directory ✅
- [ ] `[START]`/`[STEP]`/`[END]` logs emitted to stdout ✅
- [ ] 3 tasks (easy/medium/hard) with deterministic graders scoring 0.0–1.0 ✅
- [ ] `openenv.yaml` present and valid ✅
- [ ] `server/Dockerfile` builds cleanly ✅
- [ ] `reset()`, `step()`, `state()` endpoints exposed via `create_app()` ✅
- [ ] Typed Pydantic models for Action, Observation, State ✅
- [ ] Meaningful reward function (non-sparse, multiple components) ✅
- [ ] README has: env description, action/obs space, task descriptions, setup instructions, baseline scores ✅
- [ ] Inference runtime < 20 min (3 tasks × ≤30 steps × fast model) ✅
- [ ] Runs on vcpu=2, memory=8gb (pure Python, no GPU dependencies) ✅

### Observation extraction in inference.py (do not regress)

The WebSocket server sends: `{"type": "observation", "data": {"observation": {...}, "done": bool, "reward": float}}`.

The `_unpack(msg)` helper in `inference.py` correctly extracts the inner observation and injects `done`/`reward` at the top level. **Never replace this with `msg.get("data", msg)` directly** — that returns the outer wrapper and `obs.get("inventory", [])` would always be `[]`, scoring every task 0.0.

### Scoring weights (for prioritizing what to improve)

| Criterion | Weight |
|---|---|
| Real-world utility | 30% |
| Task & grader quality | 25% |
| Environment design | 20% |
| Code quality & spec compliance | 15% |
| Creativity & novelty | 10% |

---

## Project Goal

This is an **OpenEnv** reinforcement learning environment simulating a retail store manager. The RL task (see [docs/PROBLEM.md](docs/PROBLEM.md)) involves an LLM agent that manages store inventory to maximize profit. The agent can apply discounts, restock inventory, and move products between shelf zones. The environment is fully implemented in [server/StoreManager_environment.py](server/StoreManager_environment.py).

## Common Commands

```bash
# Install dependencies
uv sync

# Run dev server (with auto-reload)
uvicorn server.app:app --reload

# Run server directly
uv run --project . server
python -m StoreManager.server.app --port 8000

# Test environment logic directly (no HTTP server needed)
python3 server/StoreManager_environment.py

# Build Docker image
docker build -t StoreManager-env:latest -f server/Dockerfile .

# Deploy to Hugging Face Spaces
openenv push
openenv push --repo-id my-org/my-env --private

# Run tests
pytest
pytest --cov
```

## Architecture

The codebase follows the OpenEnv pattern: a FastAPI server wraps a Python environment class, and a typed client communicates via WebSocket.

**Data flow:**
1. Client (`client.py` → `StoremanagerEnv`) sends actions over a persistent WebSocket
2. Server (`server/app.py`) receives them via `openenv.core.env_server.http_server.create_app`
3. Environment (`server/StoreManager_environment.py` → `StoremanagerEnvironment`) processes actions and returns observations
4. Models (`models.py`) define the Pydantic schemas for `StoremanagerAction` and `StoremanagerObservation`

**Concurrency:** `SUPPORTS_CONCURRENT_SESSIONS = True` in the environment class; `max_concurrent_envs=1` in `server/app.py` (increase for parallel sessions).

**Package layout:** The root directory is both the Python package root (`StoreManager/`) and the Docker build context. `PYTHONPATH=/app/env` is set in the container so imports resolve correctly.

---

## Implemented Mechanics

### Core: Discount + Customer Simulation + Expiry

The base mechanic. Each step the agent discounts one product; customers sample products weighted by `base_probability × zone_multiplier × (1 + discount_pct / 100)`. Products that reach their `expiry_step` are removed and trigger a penalty of `remaining_quantity × cost_price`.

### Feature 1: Restocking / Procurement

**What it models:** The agent can replenish stock by placing a purchase order. Stock doesn't arrive instantly — there is a configurable lead time. Holding inventory costs money every step regardless of whether it sells.

**Action format:**
```json
{ "action_type": "restock", "product_id": 1, "restock_quantity": 15 }
```

**How it works in the environment (`server/StoreManager_environment.py`):**

- `_pending_orders` is a list of dicts `{product_id, quantity, arrive_at_step}` maintained on the environment instance.
- At the **start** of every `step()` call, pending orders whose `arrive_at_step <= _current_step` are delivered: `product.quantity += order["quantity"]` and the pending fields are cleared.
- When a `"restock"` action is dispatched, a new entry is appended to `_pending_orders` with `arrive_at_step = _current_step + product.restock_lead_time`. The cost `restock_quantity × restock_cost_per_unit` is charged immediately from this step's profit.
- `holding_cost` is computed every step for all active products: `sum(p.quantity × p.cost_price × p.holding_cost_rate for all active p)`. This is always charged, regardless of action type.

**Per-product restock parameters (set randomly in `reset()`):**

| Field | Range | Description |
|---|---|---|
| `restock_lead_time` | 1–3 steps | Steps until a placed order is delivered |
| `restock_cost_per_unit` | `cost_price × 1.00–1.25` | Cost to order one additional unit |
| `holding_cost_rate` | 0.01–0.04 | Fraction of `cost_price` charged per unit per step |

**Observation fields added:**

| Field | Description |
|---|---|
| `restock_cost_this_step` | Ordering cost charged this step (0 if no restock) |
| `holding_cost_this_step` | Total holding cost charged this step |
| `pending_restock_quantity` (on ProductState) | Units currently on order |
| `pending_restock_arrives_at_step` (on ProductState) | Step when pending order delivers |

**Error cases:** Returns an error observation (step does not advance) if:
- `restock_quantity` is None or ≤ 0
- The product already has a pending order (`pending_restock_quantity > 0`)

**Strategic tension:** Over-restocking inflates holding costs and risks expiry penalties if the new stock can't sell through. Under-restocking leads to stockouts on high-demand products.

---

### Feature 2: Zone / Shelf Placement

**What it models:** The physical location of a product on the shelf affects how often customers pick it. Premium zones (entrance, eye-level) get more foot traffic. Moving products costs staff labor.

**Action format:**
```json
{ "action_type": "place", "product_id": 3, "target_zone": 1 }
```

**How it works in the environment:**

- Each `ProductState` carries a `zone: int` field (1, 2, or 3).
- When a `"place"` action is dispatched, the product's zone is updated and `PLACEMENT_COST = $3.00` is deducted from this step's profit.
- The pick-probability formula uses the zone multiplier: `effective_weight[i] = base_probability[i] × ZONE_MULTIPLIERS[zone[i]] × (1 + discount_pct[i] / 100)`.

**Zone constants (defined at the top of `StoreManager_environment.py`):**

```python
ZONE_MULTIPLIERS = {1: 1.5, 2: 1.0, 3: 0.6}
ZONE_CAPACITY    = {1: 2, 2: 4, 3: 99}
PLACEMENT_COST   = 3.0
```

**Initial zone assignment in `reset()`:** Products are assigned zones greedily in order — first 2 products go to zone 1, next 4 to zone 2, remainder to zone 3. This means zone 1 is always full at episode start.

**Observation fields added:**

| Field | Description |
|---|---|
| `zone_capacity` | `{1: 2, 2: 4, 3: 99}` — max products per zone |
| `zone_occupancy` | `{1: N, 2: N, 3: N}` — active products currently in each zone |
| `zone_multipliers` | `{1: 1.5, 2: 1.0, 3: 0.6}` — pick-prob multipliers |
| `placement_cost_this_step` | $3.00 if a placement was made, else $0.00 |
| `zone` (on ProductState) | Current zone of the product |

**Error cases:** Returns an error observation (step does not advance) if:
- `target_zone` is None or not in {1, 2, 3}
- The product is already in `target_zone`
- `target_zone` is already at capacity (`zone_occupancy[target_zone] >= ZONE_CAPACITY[target_zone]`)

**Strategic tension:** Zone 1 has only 2 slots but gives a 1.5× boost — the agent must decide which products deserve premium placement at any given moment. Moving a product out to make room for a more urgent one costs $3.00 each way.

---

## Reward Formula (full)

```
step_reward = step_revenue
            − cost_of_goods_sold
            − expiry_penalty         # remaining_qty × cost_price for expired products
            − holding_cost           # Σ(qty × cost_price × holding_cost_rate) all active products
            − placement_cost         # $3.00 flat, only if action_type == "place"
            − restock_cost           # restock_qty × restock_cost_per_unit, only if action_type == "restock"
```

All components are available individually in the `metadata` dict of each observation.

---

## Key Files

| File | Role |
|---|---|
| [models.py](models.py) | Pydantic schemas: `ProductState`, `StoremanagerAction`, `StoremanagerObservation`, `StoremanagerState` |
| [server/StoreManager_environment.py](server/StoreManager_environment.py) | Full environment logic: `reset()`, `step()`, zone constants, pending orders |
| [client.py](client.py) | WebSocket client: `_step_payload()`, `_parse_result()`, `_parse_state()` |
| [inference.py](inference.py) | LLM agent runner: system prompt, inventory table, action parsing for all 3 types |
| [tasks.py](tasks.py) | Task configs (easy/medium/hard) + `grade()` function |

---

## OpenEnv Framework

- Environments inherit from `openenv.core.env_server.interfaces.Environment`
- Actions/Observations inherit from `openenv.core.env_server.types.Action` / `Observation`
- `openenv.yaml` declares the runtime config; `create_app()` auto-generates `/reset`, `/step`, `/state`, `/schema` HTTP endpoints and a `/ws` WebSocket endpoint
- The client base class `EnvClient` (from `openenv.core`) requires implementing `_step_payload()`, `_parse_result()`, and `_parse_state()`

## Reference Implementation: Wordle

[Examples/wordle/](Examples/wordle/) is a complete, working OpenEnv environment. Key things to learn from it:

- **Richer models** ([Examples/wordle/models.py](Examples/wordle/models.py)): Shows how to add a `State` class with custom fields and nested sub-models inside an `Observation`.
- **Non-trivial environment logic** ([Examples/wordle/server/environment.py](Examples/wordle/server/environment.py)): Shows how to manage external state, clear accumulated state on `reset()`, compute rewards, and snapshot internal state for the `state` property.
- **Reward providers** ([Examples/wordle/rewards.py](Examples/wordle/rewards.py)): A clean pattern for separating reward computation into pluggable `RewardProvider` classes.
- **Factory mode for concurrency** ([Examples/wordle/server/app.py](Examples/wordle/server/app.py)): Passes a factory function instead of a class to `create_app()` when environment construction needs arguments.
- **OpenEnv core source** ([Examples/wordle/src/core/](Examples/wordle/src/core/)): The full source of `openenv-core` is vendored here.

Also see [.claude/docs/PATTERNS.md](.claude/docs/PATTERNS.md) for the canonical OpenEnv coding conventions.
