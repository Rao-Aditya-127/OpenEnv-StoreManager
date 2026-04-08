# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Store Manager Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    ProductState,
    StoremanagerAction,
    StoremanagerObservation,
    StoremanagerState,
)


class StoremanagerEnv(
    EnvClient[StoremanagerAction, StoremanagerObservation, StoremanagerState]
):
    """
    Client for the Store Manager Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with StoremanagerEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print([p.name for p in result.observation.inventory])
        ...
        ...     action = StoremanagerAction(product_id=0, discount_pct=20)
        ...     result = env.step(action)
        ...     print(f"Reward: {result.reward:.2f}")

    Example with Docker:
        >>> client = StoremanagerEnv.from_docker_image("StoreManager-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(StoremanagerAction(product_id=0, discount_pct=10))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: StoremanagerAction) -> Dict:
        """Convert StoremanagerAction to JSON payload for the step message."""
        payload: Dict = {
            "action_type": action.action_type,
            "product_id": action.product_id,
            "discount_pct": action.discount_pct,
        }
        if action.restock_quantity is not None:
            payload["restock_quantity"] = action.restock_quantity
        if action.target_zone is not None:
            payload["target_zone"] = action.target_zone
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[StoremanagerObservation]:
        """Parse server response into StepResult[StoremanagerObservation]."""
        obs_data = payload.get("observation", {})

        raw_inventory = obs_data.get("inventory", [])
        inventory = [ProductState(**p) for p in raw_inventory if isinstance(p, dict)]

        observation = StoremanagerObservation(
            inventory=inventory,
            current_step=obs_data.get("current_step", 0),
            max_steps=obs_data.get("max_steps", 30),
            steps_remaining=obs_data.get("steps_remaining", 0),
            cumulative_profit=obs_data.get("cumulative_profit", 0.0),
            last_step_profit=obs_data.get("last_step_profit", 0.0),
            last_step_units_sold=obs_data.get("last_step_units_sold", 0),
            last_expired_products=obs_data.get("last_expired_products", []),
            num_customers_per_step=obs_data.get("num_customers_per_step", 10),
            error=obs_data.get("error"),
            zone_capacity=obs_data.get("zone_capacity", {1: 2, 2: 4, 3: 99}),
            zone_occupancy=obs_data.get("zone_occupancy", {1: 0, 2: 0, 3: 0}),
            zone_multipliers=obs_data.get("zone_multipliers", {1: 1.5, 2: 1.0, 3: 0.6}),
            holding_cost_this_step=obs_data.get("holding_cost_this_step", 0.0),
            placement_cost_this_step=obs_data.get("placement_cost_this_step", 0.0),
            restock_cost_this_step=obs_data.get("restock_cost_this_step", 0.0),
            unjustified_discount_penalty=obs_data.get("unjustified_discount_penalty", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> StoremanagerState:
        """Parse state endpoint response into StoremanagerState."""
        return StoremanagerState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            num_products=payload.get("num_products", 8),
            max_steps=payload.get("max_steps", 30),
            cumulative_profit=payload.get("cumulative_profit", 0.0),
            last_reward=payload.get("last_reward", 0.0),
            active_product_count=payload.get("active_product_count", 0),
            inventory_snapshot=payload.get("inventory_snapshot", []),
        )
