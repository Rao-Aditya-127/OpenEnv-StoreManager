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

A retail store simulator where an LLM agent acts as the store manager. Each step the agent chooses one action — apply a discount, restock inventory, or move a product to a better shelf zone — to influence customer demand and maximize cumulative profit over a fixed episode horizon.

## Task Overview

| Component | Description |
|---|---|
| **Products** | N products, each with cost price, selling price, quantity, expiry step, shelf zone, and a base pick probability |
| **Customers** | Fixed number of customers per step pick products according to a normalized probability distribution |
| **Agent Action** | One of three action types per step: `discount`, `restock`, or `place` |
| **Reward** | `step_revenue − step_cost − expiry_penalty − holding_cost − placement_cost − restock_cost` per step |
| **Expiry penalty** | `remaining_quantity × cost_price` for each product that expires |
| **Episode end** | After `max_steps` steps, or when all products are inactive |

### Pick Probability Formula

Customer pick probability combines zone placement and discount:

```
effective_weight[i] = base_probability[i] × zone_multiplier[zone[i]] × (1 + discount_pct[i] / 100)
pick_prob[i]        = effective_weight[i] / sum(effective_weight for all active products)
```

---

## Actions

The agent submits exactly one action per step. Three action types are available:

### 1. Discount

Apply a temporary price reduction to one product. The discount is **ephemeral** — it resets to 0% every step.

```json
{ "action_type": "discount", "product_id": 2, "discount_pct": 20 }
```

| Field | Type | Values |
|---|---|---|
| `action_type` | string | `"discount"` |
| `product_id` | int | any valid product ID |
| `discount_pct` | int | `0`, `10`, `20`, or `50` |

**Effect:** Boosts pick probability relatively by the discount percentage. A 20% discount means the product's weight is multiplied by 1.20.

**When to use:** Clear near-expiry stock (avoid expiry penalty), or drive volume on high-margin products.

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

**Effect:** After `restock_lead_time` steps, `quantity` increases by `restock_quantity` and `pending_restock_quantity` resets to 0. Only one pending order per product is allowed at a time.

**Costs:**
- `restock_cost = restock_quantity × restock_cost_per_unit` (upfront, this step)
- `holding_cost = quantity × cost_price × holding_cost_rate` per step on all stock

**When to use:** Replenish a fast-selling product with enough steps remaining to sell through the new stock before expiry.

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
|------|----------|----------------------------|--------------|
| 1 | Premium / entrance / eye-level | **×1.5** | 2 |
| 2 | Standard shelves | **×1.0** | 4 |
| 3 | Back / low-traffic | **×0.6** | unlimited |

**When to use:** Move high-value or near-expiry products to zone 1 to maximize their pick probability. Demote slow-movers to zone 3 to free up premium space.

---

## Reward Breakdown

Every step the reward is computed as:

```
reward = step_revenue
       − cost_of_goods_sold
       − expiry_penalty       # remaining_qty × cost_price for expired products
       − holding_cost         # quantity × cost_price × holding_cost_rate (all products)
       − placement_cost       # $3.00 flat if a "place" action was taken
       − restock_cost         # restock_quantity × restock_cost_per_unit if a "restock" was taken
```

The `metadata` field in each observation breaks these down individually:
```json
{
  "step_revenue": 45.20,
  "step_cost": 18.40,
  "expiry_penalty": 0.0,
  "holding_cost": 12.35,
  "placement_cost": 3.0,
  "restock_cost": 0.0
}
```

---

## Difficulty Tasks

Three pre-configured tasks with increasing difficulty:

| Task | Products | Steps | Customers/step | Expiry range | Grade target |
|------|----------|-------|----------------|--------------|-------------|
| **easy** | 4 | 20 | 15 | 18–30 steps | $120 profit |
| **medium** | 8 | 30 | 10 | 8–20 steps | $280 profit |
| **hard** | 12 | 20 | 8 | 4–10 steps | $200 profit |

### Grading

```
score = clamp(cumulative_profit / profit_target, 0.0, 1.0)
```

---

## Quick Start

```python
from StoreManager import StoremanagerAction, StoremanagerEnv

with StoremanagerEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print([p.name for p in result.observation.inventory])

    # Discount action
    result = env.step(StoremanagerAction(action_type="discount", product_id=0, discount_pct=20))
    print(f"Reward: {result.reward:.2f}")

    # Restock action
    result = env.step(StoremanagerAction(action_type="restock", product_id=1, restock_quantity=10))
    print(f"Restock cost: {result.observation.restock_cost_this_step:.2f}")

    # Place action
    result = env.step(StoremanagerAction(action_type="place", product_id=2, target_zone=1))
    print(f"Placement cost: {result.observation.placement_cost_this_step:.2f}")
    print(f"Zone occupancy: {result.observation.zone_occupancy}")
```

---

## Running Locally

```bash
# Install dependencies
uv sync

# Run server (default task: custom config)
uvicorn server.app:app --reload

# Run with a specific difficulty task
STORE_TASK=hard uvicorn server.app:app --reload

# Test environment logic directly (no server needed)
python server/StoreManager_environment.py
```

## Running the Inference Script

The `inference.py` script runs an LLM agent through all three tasks and reports scores.

```bash
# Server must be running first
uvicorn server.app:app &

# Run inference
OPENAI_API_KEY=sk-... python inference.py

# Custom server URL or model
OPENAI_API_KEY=sk-... python inference.py --url http://localhost:8000 --model gpt-4o-mini
```

Output format:
```
[START] {"task": "easy", "seed": 42, "model": "gpt-4o-mini", "max_steps": 20, "profit_target": 120.0}
[STEP]  {"task": "easy", "step": 1, "action": {"action_type": "discount", "product_id": 2, "discount_pct": 20}, "reward": 12.5, "holding_cost": 8.2, "placement_cost": 0, "restock_cost": 0}
...
[END]   {"task": "easy", "cumulative_profit": 145.3, "score": 1.0, "profit_target": 120.0}
```

---

## Action & Observation Reference

### Action: `StoremanagerAction`

| Field | Type | Description |
|---|---|---|
| `action_type` | `"discount"` \| `"restock"` \| `"place"` | Type of action to take (default: `"discount"`) |
| `product_id` | int | ID of the product to act on |
| `discount_pct` | `0` \| `10` \| `20` \| `50` | Discount tier (used when `action_type="discount"`) |
| `restock_quantity` | int | Units to order (used when `action_type="restock"`) |
| `target_zone` | `1` \| `2` \| `3` | Destination zone (used when `action_type="place"`) |

### Observation: `StoremanagerObservation`

| Field | Type | Description |
|---|---|---|
| `inventory` | `List[ProductState]` | Full state of all products |
| `current_step` | int | Current step (1-indexed after first step) |
| `steps_remaining` | int | Steps left in the episode |
| `cumulative_profit` | float | Total profit so far |
| `last_step_profit` | float | Net profit from last step |
| `last_expired_products` | `List[str]` | Products that expired this step |
| `zone_capacity` | `Dict[int, int]` | Max products per zone: `{1: 2, 2: 4, 3: 99}` |
| `zone_occupancy` | `Dict[int, int]` | Current active products per zone |
| `zone_multipliers` | `Dict[int, float]` | Pick-prob multiplier per zone: `{1: 1.5, 2: 1.0, 3: 0.6}` |
| `holding_cost_this_step` | float | Total holding cost charged this step |
| `placement_cost_this_step` | float | Labor cost if a `place` action was taken ($3.00 or $0.00) |
| `restock_cost_this_step` | float | Ordering cost if a `restock` action was taken |
| `task_name` | str | Active task: `easy` / `medium` / `hard` / `custom` |
| `error` | str \| None | Error message if the action was invalid (step does not advance) |
| `metadata` | dict | Per-step cost breakdown: revenue, COGS, expiry, holding, placement, restock |

### ProductState fields (inside `inventory`)

| Field | Description |
|---|---|
| `product_id` | Unique integer ID |
| `name` | Product name (e.g. `"Milk"`) |
| `cost_price` / `selling_price` | Per-unit prices |
| `base_selling_price` | Original price before any discount |
| `quantity` | Units currently in stock |
| `expiry_step` | Episode step at which the product expires |
| `zone` | Current shelf zone: `1` (premium), `2` (standard), `3` (back) |
| `effective_pick_prob` | Normalized pick probability this step (sums to 1.0 across active) |
| `current_discount_pct` | Active discount this step (resets to 0 every step) |
| `holding_cost_rate` | Holding cost per unit per step as a fraction of `cost_price` |
| `restock_cost_per_unit` | Cost to order one additional unit |
| `restock_lead_time` | Steps until a restock order is delivered |
| `pending_restock_quantity` | Units currently on order (not yet delivered) |
| `pending_restock_arrives_at_step` | Step when the pending order will be delivered |

---

## Configuration

Control the server via environment variables:

| Variable | Default | Description |
|---|---|---|
| `STORE_TASK` | _(none)_ | Set to `easy`, `medium`, or `hard` to use a preset task |
| `STORE_NUM_PRODUCTS` | 8 | Number of products (overridden by STORE_TASK) |
| `STORE_MAX_STEPS` | 30 | Episode length (overridden by STORE_TASK) |
| `STORE_NUM_CUSTOMERS` | 10 | Customers per step (overridden by STORE_TASK) |
| `STORE_MAX_CONCURRENT_ENVS` | 4 | Max parallel WebSocket sessions |

## Deploying to Hugging Face Spaces

```bash
openenv push
openenv push --repo-id my-org/store-manager --private
```

After deployment, your space includes:
- **Web Interface** at `/web`
- **API Documentation** at `/docs`
- **WebSocket** at `/ws` — persistent session for low-latency agent interactions

## Project Structure

```
StoreManager/
├── __init__.py              # Module exports
├── models.py                # Action, Observation, State Pydantic models
├── client.py                # StoremanagerEnv WebSocket client
├── tasks.py                 # Task configs (easy/medium/hard) + grader
├── inference.py             # OpenAI agent runner with structured logs
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
└── server/
    ├── StoreManager_environment.py  # Core RL environment logic
    ├── app.py               # FastAPI server (HTTP + WebSocket)
    └── Dockerfile           # Container image
```
