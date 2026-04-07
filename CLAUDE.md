# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This is an **OpenEnv** reinforcement learning environment simulating a retail store manager. The actual RL task (see [docs/PROBLEM.md](docs/PROBLEM.md)) involves an LLM agent that manages store inventory by applying discounts to products to maximize profit, handling customer simulation, product expiry, and dynamic pricing. The current codebase is a **scaffold/template** — the core environment in [server/StoreManager_environment.py](server/StoreManager_environment.py) currently echoes messages back rather than implementing the full store simulation.

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

**Key extension points when implementing the real environment:**
- `StoremanagerEnvironment.reset()` — initialize product inventory, prices, expiry dates
- `StoremanagerEnvironment.step(action)` — apply discounts, simulate customer purchases, calculate profit/penalty rewards, handle expiry
- `StoremanagerAction` in `models.py` — replace `message` field with `product_id` and `discount_choice`
- `StoremanagerObservation` in `models.py` — replace echo fields with full inventory state, step number, cumulative profit

**Concurrency:** `SUPPORTS_CONCURRENT_SESSIONS = True` in the environment class; `max_concurrent_envs=1` in `server/app.py` (increase for parallel sessions).

**Package layout:** The root directory is both the Python package root (`StoreManager/`) and the Docker build context. `PYTHONPATH=/app/env` is set in the container so imports resolve correctly.

## OpenEnv Framework

- Environments inherit from `openenv.core.env_server.interfaces.Environment`
- Actions/Observations inherit from `openenv.core.env_server.types.Action` / `Observation`
- `openenv.yaml` declares the runtime config; `create_app()` auto-generates `/reset`, `/step`, `/state`, `/schema` HTTP endpoints and a `/ws` WebSocket endpoint
- The client base class `EnvClient` (from `openenv.core`) requires implementing `_step_payload()`, `_parse_result()`, and `_parse_state()`

## Reference Implementation: Wordle

[Examples/wordle/](Examples/wordle/) is a complete, working OpenEnv environment wrapping the Wordle game from the `textarena` library. Use it as the primary reference when implementing the store environment.

Key things to learn from it:

- **Richer models** ([Examples/wordle/models.py](Examples/wordle/models.py)): Shows how to add a `State` class (extending `openenv.core.env_server.types.State`) with custom fields like `turn`, `last_reward`, `raw_state`, and how to add nested sub-models (e.g. `TextArenaMessage`) inside an `Observation`.

- **Non-trivial environment logic** ([Examples/wordle/server/environment.py](Examples/wordle/server/environment.py)): Shows how to manage external library state, properly clear accumulated state on `reset()`, compute rewards with helper providers, and snapshot internal state for the `state` property.

- **Reward providers** ([Examples/wordle/rewards.py](Examples/wordle/rewards.py)): A clean pattern for separating reward computation into pluggable `RewardProvider` classes with `reset()` and `compute()` methods. Useful model for implementing the store's expiry penalty and discount-profit reward signals.

- **Factory mode for concurrency** ([Examples/wordle/server/app.py](Examples/wordle/server/app.py)): Passes a factory function (`create_textarena_environment`) instead of a class to `create_app()` when environment construction needs arguments — relevant if the store environment needs initialization parameters.

- **OpenEnv core source** ([Examples/wordle/src/core/](Examples/wordle/src/core/)): The full source of `openenv-core` is vendored here. Check [Examples/wordle/src/core/env_server/interfaces.py](Examples/wordle/src/core/env_server/interfaces.py) and [Examples/wordle/src/core/env_client.py](Examples/wordle/src/core/env_client.py) to understand exactly what the base classes expect.

Also see [.claude/docs/PATTERNS.md](.claude/docs/PATTERNS.md) for the canonical OpenEnv coding conventions used across all environments in this project.
