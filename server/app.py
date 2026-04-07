# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Store Manager Environment.

This module creates an HTTP + WebSocket server that exposes the
StoremanagerEnvironment over the OpenEnv API.

Endpoints:
    POST /reset  — Reset the environment, return initial observation
    POST /step   — Execute an action, return next observation + reward
    GET  /state  — Get current environment state
    GET  /schema — Action / Observation / State JSON schemas
    WS   /ws     — Persistent WebSocket session (lower latency)

Configuration (via environment variables):
    STORE_NUM_PRODUCTS       Number of products per episode  (default 8)
    STORE_MAX_STEPS          Episode length in steps         (default 30)
    STORE_NUM_CUSTOMERS      Customers simulated per step    (default 10)
    STORE_MAX_CONCURRENT_ENVS Max parallel WebSocket sessions (default 4)

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with:\n    uv sync\n"
    ) from e

try:
    from models import StoremanagerAction, StoremanagerObservation
    from StoreManager_environment import StoremanagerEnvironment
    from tasks import TASKS
except ModuleNotFoundError:
    from models import StoremanagerAction, StoremanagerObservation
    from server.StoreManager_environment import StoremanagerEnvironment
    from tasks import TASKS


def create_store_environment() -> StoremanagerEnvironment:
    """Factory: create a fresh StoremanagerEnvironment from env-var config.

    If STORE_TASK is set to a valid task name (easy/medium/hard), the task
    config takes precedence over individual STORE_* vars.
    """
    task_name = os.getenv("STORE_TASK", "")
    task_config = TASKS.get(task_name) if task_name else None
    return StoremanagerEnvironment(
        task_config=task_config,
        num_products=int(os.getenv("STORE_NUM_PRODUCTS", "8")),
        max_steps=int(os.getenv("STORE_MAX_STEPS", "30")),
        num_customers=int(os.getenv("STORE_NUM_CUSTOMERS", "10")),
    )


app = create_app(
    create_store_environment,
    StoremanagerAction,
    StoremanagerObservation,
    env_name="StoreManager",
    max_concurrent_envs=int(os.getenv("STORE_MAX_CONCURRENT_ENVS", "4")),
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Entry point for direct execution via uv run or python -m.

        uv run --project . server
        uv run --project . server --port 8001
        python -m StoreManager.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
