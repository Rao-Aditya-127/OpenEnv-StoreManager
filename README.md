---
title: Store Manager Environment
emoji: 🛒
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Store Manager Environment

A retail store simulator where an LLM agent acts as the store manager. The agent observes full inventory state each step and chooses one action — discount a product, restock inventory, or move a product to a better shelf zone — to influence customer demand and maximize cumulative profit over a fixed episode horizon.

---

## What Makes This Environment Interesting

This environment models three interconnected real-world retail mechanics that require the agent to reason about **compounding consequences**:

- **Discounts are permanent within an episode.** Every discount cuts the current (already-discounted) price further. A 20% discount applied twice yields 36% off the original price. The agent must decide how aggressively to discount — deep cuts clear stock fast but permanently erode margin.
- **Restocking has a time lag.** Stock doesn't arrive instantly. Each product has a `restock_lead_time` (1–3 steps). The agent must plan ahead — ordering too late leaves the shelf empty; ordering too much risks expiry penalties and inflated holding costs.
- **Zone placement directly multiplies pick probability.** Zone 1 (premium) boosts visibility 1.5×, but only holds 2 products. The agent must continuously rotate which products deserve premium placement.

These three mechanics combine into a rich multi-step decision problem where each action affects future state: a discount today changes tomorrow's revenue; a restock today affects holding costs for the next several steps; a zone move today costs $3 but can double a product's sell-through rate.

---

## Task Overview

| Component | Description |
|---|---|
| **Products** | N products, each with cost price, selling price, quantity, expiry step, shelf zone, and a base pick probability |
| **Customers** | Fixed number of customers per step pick products according to a normalized probability distribution |
| **Agent Action** | Exactly one of three action types per step: `discount`, `restock`, or `place` |
| **Reward** | Normalized sigmoid reward in `[0.0, 1.0]` based on step profit vs. per-step target |
| **Expiry penalty** | `remaining_quantity × cost_price` charged when a product reaches its expiry step |
| **Episode end** | After `max_steps` steps, or when all products are inactive |

### Pick Probability Formula

Customer pick probability combines zone placement and discount multiplier:

```
effective_weight[i] = base_probability[i] × zone_multiplier[zone[i]] × (1 + discount_pct[i] / 100)
pick_prob[i]        = effective_weight[i] / sum(effective_weight for all active products)
```

---

## Actions

The agent submits exactly one action per step.

### 1. Discount

Reduce the selling price of a product. Discounts **compound across steps within an episode** — each discount cuts the current price further, not the original base price.

```json
{ "action_type": "discount", "product_id": 2, "discount_pct": 20 }
```

| Field | Type | Values |
|---|---|---|
| `action_type` | string | `"discount"` |
| `product_id` | int | any valid product ID |
| `discount_pct` | int | `0`, `10`, `20`, or `50` |

**Compounding example** (base price $10.00):

| Step | Discount | Selling Price | Total off base |
|------|----------|--------------|----------------|
| 1 | 20% | $8.00 | −20% |
| 2 | 20% | $6.40 | −36% |
| 3 | 20% | $5.12 | −49% |
| 4 | 50% | $2.56 | −74% |

- `base_selling_price` is never modified — it serves as a reference for how far prices have been cut.
- `current_discount_pct` reflects only the discount tier applied **this step** (used in pick-probability formula).
- Prices reset to `base_selling_price` at the start of each new episode (`reset()`).
- A `0%` discount leaves the price unchanged but still advances the step.

**When to use:** Apply deep discounts to near-expiry stock to clear inventory before the expiry penalty hits. Use light discounts on high-margin products to drive volume without destroying margin.

---

### 2. Restock

Place a purchase order for more units of a product. Stock arrives after `restock_lead_time` steps. The ordering cost is charged **immediately** this step.

```json
{ "action_type": "restock", "product_id": 1, "restock_quantity": 15 }
```

| Field | Type | Values |
|---|---|---|
| `action_type` | string | `"restock"` |
| `product_id` | int | any valid product ID |
| `restock_quantity` | int | any positive integer |

**How it works:**
- On restock: `pending_restock_quantity = restock_quantity`, `pending_restock_arrives_at_step = current_step + restock_lead_time`
- When delivery step is reached: `quantity += restock_quantity`, pending fields cleared
- Only **one pending order per product** at a time — wait for delivery before reordering

**Costs:**
- `restock_cost = restock_quantity × restock_cost_per_unit` charged upfront this step
- `holding_cost = quantity × cost_price × holding_cost_rate` charged every step on all stock regardless of action

**Per-product restock parameters (randomised at episode start):**

| Field | Range | Description |
|---|---|---|
| `restock_lead_time` | 1–3 steps | Steps until delivery |
| `restock_cost_per_unit` | `cost_price × 1.00–1.25` | Ordering cost per unit |
| `holding_cost_rate` | 0.01–0.04 | Fraction of `cost_price` charged per unit per step |

**Error cases:** Invalid if `restock_quantity ≤ 0` or product already has a pending order.

**When to use:** Replenish a fast-selling product before it stocks out, only when enough steps remain to sell through the new inventory before expiry.

---

### 3. Place (Zone Placement)

Move a product to a different shelf zone. Takes effect immediately. Charges a flat **labor fee of $3.00** per move.

```json
{ "action_type": "place", "product_id": 3, "target_zone": 1 }
```

| Field | Type | Values |
|---|---|---|
| `action_type` | string | `"place"` |
| `product_id` | int | any valid product ID |
| `target_zone` | int | `1`, `2`, or `3` |

**Zone multipliers and capacity:**

| Zone | Location | Pick Probability Multiplier | Max Products |
|------|----------|-----------------------------|--------------|
| 1 | Premium / entrance / eye-level | **×1.5** | 2 |
| 2 | Standard shelves | **×1.0** | 4 |
| 3 | Back / low-traffic | **×0.6** | unlimited |

**Initial zone assignment at episode start:** products fill zone 1 first (up to capacity 2), then zone 2, remainder in zone 3.

**Error cases:** Invalid if `target_zone` is not in {1, 2, 3}, product is already in `target_zone`, or target zone is at capacity.

**When to use:** Rotate zone 1 to whichever product most needs visibility — near-expiry items benefit most since clearance speed directly avoids penalty. Demote slow-movers to zone 3 to free up premium space.

---

## Reward Function

Each step returns a **normalized reward in `[0.0, 1.0]`** using a sigmoid centred at break-even:

```
step_profit = step_revenue − cost_of_goods_sold − expiry_penalty
            − holding_cost − placement_cost − restock_cost

x      = step_profit / per_step_target
reward = sigmoid(2x) = 1 / (1 + exp(−2x))
```

| Step profit vs. target | reward |
|---|---|
| Catastrophic loss (−∞) | → 0.0 |
| Break-even (0) | = 0.50 |
| On target | ≈ 0.88 |
| 2× target | ≈ 0.98 |

The raw per-step cost breakdown is available in `metadata`:

```json
{
  "step_revenue": 45.20,
  "step_cost": 18.40,
  "expiry_penalty": 0.0,
  "holding_cost": 12.35,
  "placement_cost": 3.0,
  "restock_cost": 0.0,
  "step_profit": 11.45
}
```

---

## Difficulty Tasks

Four pre-configured tasks with increasing complexity. Each task has a fixed seed for deterministic, reproducible grading.

| Task | Products | Steps | Customers/step | Expiry range | Profit target | Dominant mechanic |
|------|----------|-------|----------------|--------------|--------------|-------------------|
| **easy** | 4 | 20 | 15 | 18–30 steps | $120 | Discount |
| **medium** | 8 | 28 | 12 | 12–22 steps | $200 | Discount + zone placement |
| **hard** | 10 | 30 | 10 | 8–18 steps | $240 | Discount + zone + expiry timing |
| **expert** | 12 | 25 | 8 | 4–10 steps | $190 | All three actions coordinated |

### Task Descriptions

**easy** — Products expire well after the episode ends; no expiry pressure. The agent learns to identify high-margin products and apply targeted discounts to boost their pick probability.

**medium** — Several products expire mid-episode. Zone 1 has only 2 slots for 8 competing products — the agent must combine discounts with smart zone rotation to clear stock before expiry.

**hard** — Most products expire in 8–18 steps. Expiry management is critical: rotate zone 1 frequently to accelerate clearance, and consider restocking fast-sellers that run out early.

**expert** — Aggressive expiry (4–10 steps) with limited initial stock. All three actions must be coordinated: discount near-expiry items heavily, restock high-margin products before they stock out, and rotate zone 1 continuously.

### Grading

```
score = clamp(cumulative_profit / profit_target, 0.0, 1.0)
```

---

## Baseline Scores

Expected performance using `gpt-4o-mini` (deterministic, fixed seeds):

| Task | Profit Target | Expected Profit | Expected Score |
|------|--------------|-----------------|----------------|
| easy | $120 | ~$140–180 | ~0.90–1.0 |
| medium | $200 | ~$160–220 | ~0.80–1.0 |
| hard | $240 | ~$180–260 | ~0.75–1.0 |
| expert | $190 | ~$100–170 | ~0.50–0.90 |

Scores are fully reproducible. Re-run `inference.py` with the same model and seeds to verify.

---

## Quick Start

```python
from StoreManager import StoremanagerAction, StoremanagerEnv

with StoremanagerEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print([p.name for p in result.observation.inventory])

    # Discount: compounds on current price each time applied
    result = env.step(StoremanagerAction(action_type="discount", product_id=0, discount_pct=20))
    print(f"Reward: {result.reward:.2f}")

    # Restock: stock arrives after restock_lead_time steps
    result = env.step(StoremanagerAction(action_type="restock", product_id=1, restock_quantity=10))
    print(f"Restock cost: {result.observation.restock_cost_this_step:.2f}")

    # Place: move product to premium zone (costs $3.00 labor)
    result = env.step(StoremanagerAction(action_type="place", product_id=2, target_zone=1))
    print(f"Placement cost: {result.observation.placement_cost_this_step:.2f}")
    print(f"Zone occupancy: {result.observation.zone_occupancy}")
```

---

## Running Locally

```bash
# Install dependencies
uv sync

# Run dev server (auto-reload, default config)
uvicorn server.app:app --reload

# Run with a specific difficulty task baked in at startup
STORE_TASK=hard uvicorn server.app:app --reload

# Test environment logic directly (no server needed)
python server/StoreManager_environment.py

# Run the WebSocket test suite (server must be running)
python test_ws.py
```

---

## Running the Inference Script

The `inference.py` script runs an LLM agent through all four tasks sequentially and reports a score for each.

### Required Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | API key — your Hugging Face token or OpenAI API key |
| `API_BASE_URL` | LLM API base URL (e.g. `https://api-inference.huggingface.co/v1/`) |
| `MODEL_NAME` | Model identifier (e.g. `meta-llama/Llama-3.1-8B-Instruct` or `gpt-4o-mini`) |

```bash
# Server must be running first
uvicorn server.app:app &

# Run with Hugging Face Inference API
HF_TOKEN=hf_... \
API_BASE_URL=https://api-inference.huggingface.co/v1/ \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
python inference.py

# Run with OpenAI (API_BASE_URL defaults to OpenAI when unset)
HF_TOKEN=sk-... MODEL_NAME=gpt-4o-mini python inference.py

# Override the environment server URL
HF_TOKEN=sk-... MODEL_NAME=gpt-4o-mini STORE_ENV_URL=http://localhost:8000 python inference.py
```

### Structured Log Format

```
[START] {"task": "easy", "seed": 42, "model": "gpt-4o-mini", "max_steps": 20, "profit_target": 120.0}
[STEP]  {"task": "easy", "step": 1, "action": {"action_type": "discount", "product_id": 2, "discount_pct": 20}, "reward": 0.88, "cumulative_profit": 12.5, "holding_cost": 8.2, "placement_cost": 0, "restock_cost": 0}
...
[END]   {"task": "easy", "cumulative_profit": 145.3, "score": 1.0, "profit_target": 120.0}
```

A single server instance handles all four tasks — the task config is passed in the `reset` message, so no server restart is needed between tasks.

---

## Action & Observation Reference

### Action: `StoremanagerAction`

| Field | Type | Default | Description |
|---|---|---|---|
| `action_type` | `"discount"` \| `"restock"` \| `"place"` | `"discount"` | Type of action |
| `product_id` | int | required | ID of the product to act on |
| `discount_pct` | `0` \| `10` \| `20` \| `50` | `0` | Discount tier (for `discount` actions) |
| `restock_quantity` | int | `None` | Units to order (for `restock` actions) |
| `target_zone` | `1` \| `2` \| `3` | `None` | Destination zone (for `place` actions) |

### Observation: `StoremanagerObservation`

| Field | Type | Description |
|---|---|---|
| `inventory` | `List[ProductState]` | Full state of all products (including inactive) |
| `current_step` | int | Current step number (0 at reset, increments after each valid step) |
| `max_steps` | int | Total steps in this episode |
| `steps_remaining` | int | Steps left in the episode |
| `cumulative_profit` | float | Total profit accumulated so far |
| `last_step_profit` | float | Net profit from the most recent step |
| `last_expired_products` | `List[str]` | Names of products that expired this step |
| `zone_capacity` | `Dict[int, int]` | Max products per zone: `{1: 2, 2: 4, 3: 99}` |
| `zone_occupancy` | `Dict[int, int]` | Active products currently in each zone |
| `zone_multipliers` | `Dict[int, float]` | Pick-prob multiplier per zone: `{1: 1.5, 2: 1.0, 3: 0.6}` |
| `holding_cost_this_step` | float | Total holding cost charged this step |
| `placement_cost_this_step` | float | `3.0` if a `place` action was taken, else `0.0` |
| `restock_cost_this_step` | float | Ordering cost if a `restock` was placed, else `0.0` |
| `task_name` | str | Active task: `easy` / `medium` / `hard` / `expert` / `custom` |
| `error` | str \| None | Error message if the action was invalid (step does **not** advance) |
| `metadata` | dict | Per-step breakdown: `step_revenue`, `step_cost`, `expiry_penalty`, `holding_cost`, `placement_cost`, `restock_cost`, `step_profit` |

### ProductState Fields (inside `inventory`)

| Field | Description |
|---|---|
| `product_id` | Unique integer ID |
| `name` | Product name (e.g. `"Milk"`) |
| `cost_price` | Per-unit cost to the store |
| `selling_price` | Current selling price (decreases each time a discount is applied) |
| `base_selling_price` | Original price — never modified, reference for total discount depth |
| `quantity` | Units currently in stock |
| `expiry_step` | Episode step at which this product expires |
| `zone` | Current shelf zone: `1` (premium), `2` (standard), `3` (back) |
| `effective_pick_prob` | Normalized pick probability this step (sums to 1.0 across all active products) |
| `current_discount_pct` | Discount tier applied **this step only** (0 if no discount action taken) |
| `holding_cost_rate` | Holding cost per unit per step as a fraction of `cost_price` |
| `restock_cost_per_unit` | Cost to order one additional unit |
| `restock_lead_time` | Steps until a placed order is delivered |
| `pending_restock_quantity` | Units currently on order (not yet delivered) |
| `pending_restock_arrives_at_step` | Step number when the pending order will arrive |
| `is_active` | `False` if the product is sold out or expired |

---

## Error Handling

When an action is invalid, the step does **not** advance. The returned observation contains an `error` string and reflects the pre-action state. Common error cases:

| Scenario | Error |
|---|---|
| Discount on inactive / sold-out product | `"Product '...' is no longer active"` |
| Restock with `quantity ≤ 0` | `"restock_quantity must be a positive integer"` |
| Restock when pending order exists | `"already has a pending restock order"` |
| Place to same zone product is already in | `"is already in zone N"` |
| Place to zone at capacity | `"Zone N is at capacity"` |
| Invalid `product_id` | `"Invalid product_id"` |

Always check `obs.get("error")` before processing the observation.

---

## Configuration

Control the server via environment variables:

| Variable | Default | Description |
|---|---|---|
| `STORE_TASK` | _(none)_ | Pre-load a task config at startup: `easy`, `medium`, `hard`, or `expert` |
| `STORE_NUM_PRODUCTS` | 8 | Number of products (overridden by `STORE_TASK`) |
| `STORE_MAX_STEPS` | 30 | Episode length in steps (overridden by `STORE_TASK`) |
| `STORE_NUM_CUSTOMERS` | 10 | Customers simulated per step (overridden by `STORE_TASK`) |
| `STORE_MAX_CONCURRENT_ENVS` | 4 | Max parallel WebSocket sessions |

**Task switching at runtime:** A single server instance can serve all four tasks. Pass `"task": "<name>"` in the WebSocket `reset` message and the environment reconfigures immediately without a server restart.

---

## Deploying to Hugging Face Spaces

```bash
openenv push
openenv push --repo-id my-org/store-manager --private
```

After deployment, your space includes:
- **Web Interface** at `/web`
- **API Documentation** at `/docs`
- **WebSocket** at `/ws` — persistent session for low-latency agent interactions

---

## Project Structure

```
StoreManager/
├── __init__.py                       # Module exports
├── models.py                         # Pydantic schemas: Action, Observation, State
├── client.py                         # StoremanagerEnv WebSocket client
├── tasks.py                          # Task configs (easy/medium/hard/expert) + grader
├── inference.py                      # LLM agent runner with structured logs
├── test_ws.py                        # WebSocket integration test suite (7 test groups)
├── openenv.yaml                      # OpenEnv manifest
├── pyproject.toml                    # Dependencies
└── server/
    ├── StoreManager_environment.py   # Core RL environment logic
    ├── app.py                        # FastAPI server (HTTP + WebSocket)
    └── Dockerfile                    # Container image
```
