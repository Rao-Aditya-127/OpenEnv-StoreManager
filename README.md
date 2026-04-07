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

A retail store simulator where an LLM agent acts as the store manager. Each step the agent applies a discount to one product to influence customer demand and maximize cumulative profit over a fixed episode horizon.

## Task Overview

| Component | Description |
|---|---|
| **Products** | N products, each with cost price, selling price, quantity, expiry step, and a base pick probability |
| **Customers** | Fixed number of customers per step pick products according to a normalized probability distribution |
| **Agent Action** | Choose one product and a discount tier: 0%, 10%, 20%, or 50% |
| **Reward** | `step_revenue − step_cost − expiry_penalty` per step |
| **Expiry penalty** | `remaining_quantity × cost_price` for each product that expires |
| **Episode end** | After `max_steps` steps, or when all products are inactive |

### Discount → Probability Formula

Discounting a product multiplicatively boosts its pick probability:

```
effective_weight[i] = base_probability[i] × (1 + discount_pct[i] / 100)
pick_prob[i]        = effective_weight[i] / sum(effective_weight for all active products)
```

## Difficulty Tasks

Three pre-configured tasks with increasing difficulty:

| Task | Products | Steps | Customers/step | Expiry range | Grade target |
|------|----------|-------|---------------|-------------|-------------|
| **easy** | 4 | 20 | 15 | 18–30 steps | $120 profit |
| **medium** | 8 | 30 | 10 | 8–20 steps | $280 profit |
| **hard** | 12 | 20 | 8 | 4–10 steps | $200 profit |

### Grading

Each task is graded deterministically (fixed seed per task):

```
score = clamp(cumulative_profit / profit_target, 0.0, 1.0)
```

A score of 1.0 means the agent reached or exceeded the profit target. Partial credit is awarded proportionally.

## Quick Start

```python
from StoreManager import StoremanagerAction, StoremanagerEnv

with StoremanagerEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print([p.name for p in result.observation.inventory])

    # Apply 20% discount to product 0
    result = env.step(StoremanagerAction(product_id=0, discount_pct=20))
    print(f"Reward: {result.reward:.2f}")
    print(f"Step profit: {result.observation.last_step_profit:.2f}")
    print(f"Cumulative: {result.observation.cumulative_profit:.2f}")
```

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
[STEP]  {"task": "easy", "step": 1, "action": {"product_id": 2, "discount_pct": 20}, "reward": 12.5, "cumulative_profit": 12.5}
...
[END]   {"task": "easy", "cumulative_profit": 145.3, "score": 1.0, "profit_target": 120.0}
```

## Action & Observation

### Action: `StoremanagerAction`
| Field | Type | Description |
|---|---|---|
| `product_id` | int | ID of the product to discount |
| `discount_pct` | 0 \| 10 \| 20 \| 50 | Discount tier to apply |

### Observation: `StoremanagerObservation`
| Field | Type | Description |
|---|---|---|
| `inventory` | List[ProductState] | Full state of all products |
| `current_step` | int | Current step (0-indexed) |
| `steps_remaining` | int | Steps left in the episode |
| `cumulative_profit` | float | Total profit so far |
| `last_step_profit` | float | Net profit from last step |
| `last_expired_products` | List[str] | Products that expired this step |
| `task_name` | str | Active task: easy / medium / hard / custom |

### ProductState fields (inside inventory)
| Field | Description |
|---|---|
| `product_id` | Unique integer ID |
| `name` | Product name (e.g. "Milk") |
| `cost_price` / `selling_price` | Per-unit prices |
| `quantity` | Units in stock |
| `expiry_step` | Step number at which product expires |
| `effective_pick_prob` | Normalized pick probability this step (sums to 1.0) |
| `current_discount_pct` | Active discount this step (resets each step) |

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
# Deploy to your namespace
openenv push

# Deploy to a specific repo
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
