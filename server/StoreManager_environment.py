# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Store Manager Environment Implementation.

A retail store simulator where an LLM agent acts as the store manager.
The agent can apply discounts, restock inventory, or move products between
shelf zones each step to maximise cumulative profit over a fixed episode horizon.

Episode dynamics:
  - N products, each with a cost price, selling price, stock quantity,
    expiry step, a base pick-probability weight, a shelf zone, and restock params.
  - Each step the agent chooses one action:
      - "discount": apply a discount (0/10/20/50%) to one product
      - "restock":  place a reorder for a product (delivered after lead_time steps)
      - "place":    move a product to a different shelf zone (1/2/3)
  - Discounts are ephemeral — they apply only for the current step.
  - Zone multiplier scales pick probability: zone 1 = 1.5x, 2 = 1.0x, 3 = 0.6x.
  - 10 customers sample products per step; probabilities incorporate zone + discount.
  - Holding cost is charged every step based on quantity on hand.
  - Restock cost is charged upfront; stock arrives after lead_time steps.
  - Placement cost is a flat labor fee per move.
  - Reward = step_revenue - step_cost - expiry_penalty - holding_cost
             - placement_cost - restock_cost.
  - Expiry penalty = remaining_quantity * cost_price for each expired product.
  - Episode ends after max_steps steps, or when all products are inactive.
"""

from __future__ import annotations

import sys
import os

# Ensure the project root (parent of server/) is on sys.path so that
# models.py and tasks.py are importable whether this file is run directly
# or imported as part of the package.
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment

from models import (
    ProductState,
    StoremanagerAction,
    StoremanagerObservation,
    StoremanagerState,
)

try:
    from tasks import TaskConfig
except ImportError:
    TaskConfig = None  # type: ignore[assignment,misc]

# ── Curated product catalogue ──────────────────────────────────────────────────
_PRODUCT_NAMES: List[str] = [
    "Milk", "Bread", "Cheese", "Yogurt", "Juice", "Butter",
    "Eggs", "Cream", "Honey", "Jam", "Soda", "Water",
    "Tea", "Coffee", "Flour", "Rice",
]

# ── Zone constants ─────────────────────────────────────────────────────────────
ZONE_MULTIPLIERS: Dict[int, float] = {1: 1.5, 2: 1.0, 3: 0.6}
ZONE_CAPACITY: Dict[int, int] = {1: 2, 2: 4, 3: 99}
PLACEMENT_COST: float = 3.0  # flat labor fee per zone move


class StoremanagerEnvironment(Environment):
    """
    Retail store manager RL environment.

    The agent observes full inventory state and chooses one action per step:
    apply a discount, place a restock order, or move a product to a better zone.

    Args:
        num_products: Number of distinct products in the store (default 8).
        max_steps: Episode length in steps (default 30).
        num_customers: Customers simulated per step (default 10).
        seed: Optional base random seed for reproducible episodes.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        task_config: Optional["TaskConfig"] = None,  # type: ignore[name-defined]
        num_products: int = 8,
        max_steps: int = 30,
        num_customers: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if task_config is not None:
            num_products = task_config.num_products
            max_steps = task_config.max_steps
            num_customers = task_config.num_customers
            self._expiry_offset_min = task_config.expiry_offset_min
            self._expiry_offset_max = task_config.expiry_offset_max
            self._task_name = task_config.name
            self._profit_target = task_config.profit_target
        else:
            self._expiry_offset_min = 8
            self._expiry_offset_max = 25
            self._task_name = "custom"
            # Default: assume ~$8 average profit per step for custom configs
            self._profit_target = 8.0 * max_steps

        self._num_products = min(num_products, len(_PRODUCT_NAMES))
        self._max_steps = max_steps
        self._num_customers = num_customers
        self._base_seed = seed

        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._inventory: List[ProductState] = []
        self._pending_orders: List[Dict[str, Any]] = []
        self._current_step: int = 0
        self._cumulative_profit: float = 0.0

        self._state = StoremanagerState(
            num_products=self._num_products,
            max_steps=self._max_steps,
        )

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StoremanagerObservation:
        """Reset the store and generate a fresh inventory.

        Accepts an optional ``task`` keyword argument to dynamically switch the
        task configuration without restarting the server.  This allows a single
        server instance to serve all difficulty levels sequentially.
        """
        # ── Dynamic task reconfiguration ────────────────────────────────────
        task_name = kwargs.get("task")
        if task_name:
            try:
                from tasks import TASKS  # local import to avoid circular deps
                task_config = TASKS.get(task_name)
                if task_config is not None:
                    self._num_products = min(task_config.num_products, len(_PRODUCT_NAMES))
                    self._max_steps = task_config.max_steps
                    self._num_customers = task_config.num_customers
                    self._expiry_offset_min = task_config.expiry_offset_min
                    self._expiry_offset_max = task_config.expiry_offset_max
                    self._task_name = task_config.name
                    self._profit_target = task_config.profit_target
            except ImportError:
                pass  # tasks module unavailable; keep current config

        self._reset_rubric()

        effective_seed = seed if seed is not None else self._base_seed
        self._rng = np.random.default_rng(effective_seed)

        self._current_step = 0
        self._cumulative_profit = 0.0
        self._pending_orders = []
        self._inventory = self._generate_inventory()
        self._recompute_pick_probs()

        self._update_state(episode_id=episode_id or str(uuid4()), last_reward=0.0)

        return StoremanagerObservation(
            inventory=list(self._inventory),
            current_step=0,
            max_steps=self._max_steps,
            steps_remaining=self._max_steps,
            cumulative_profit=0.0,
            last_step_profit=0.0,
            last_step_units_sold=0,
            last_expired_products=[],
            num_customers_per_step=self._num_customers,
            task_name=self._task_name,
            zone_capacity=dict(ZONE_CAPACITY),
            zone_occupancy=self._zone_occupancy(),
            zone_multipliers=dict(ZONE_MULTIPLIERS),
            holding_cost_this_step=0.0,
            placement_cost_this_step=0.0,
            restock_cost_this_step=0.0,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: StoremanagerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> StoremanagerObservation:
        """Execute one step: process deliveries, apply action, simulate customers, handle expiry."""
        # ── 1. Validate product_id ──────────────────────────────────────────
        valid_ids = {p.product_id for p in self._inventory}
        if action.product_id not in valid_ids:
            return self._error_observation(
                f"Invalid product_id {action.product_id}. "
                f"Valid IDs: {sorted(valid_ids)}"
            )

        # ── 2. Process pending deliveries ───────────────────────────────────
        delivered = [o for o in self._pending_orders if o["arrive_at_step"] <= self._current_step]
        for order in delivered:
            product = next((p for p in self._inventory if p.product_id == order["product_id"]), None)
            if product is not None:
                product.quantity += order["quantity"]
                product.is_active = True
                product.pending_restock_quantity = 0
                product.pending_restock_arrives_at_step = None
        self._pending_orders = [o for o in self._pending_orders if o["arrive_at_step"] > self._current_step]

        # ── 3. Compute holding cost (charged before action) ─────────────────
        holding_cost = sum(
            p.quantity * p.cost_price * p.holding_cost_rate
            for p in self._inventory
            if p.is_active and p.quantity > 0
        )

        # ── 4. Reset per-step discount tracker (selling_price persists across steps) ──
        # current_discount_pct tracks only this step's discount for the pick-prob formula.
        # selling_price is NOT reset — discounts compound: each action cuts the current price.
        for product in self._inventory:
            product.current_discount_pct = 0.0

        # ── 5. Dispatch action ──────────────────────────────────────────────
        placement_cost = 0.0
        restock_cost = 0.0
        target = next(p for p in self._inventory if p.product_id == action.product_id)

        if action.action_type == "discount":
            if not target.is_active:
                return self._error_observation(
                    f"Product '{target.name}' (id={target.product_id}) is no longer active "
                    f"(sold out or expired). Choose an active product to discount."
                )
            if target.quantity <= 0:
                return self._error_observation(
                    f"Product '{target.name}' (id={target.product_id}) has no stock remaining."
                )
            target.current_discount_pct = float(action.discount_pct)
            # selling_price compounds down: each discount cuts the current price
            target.selling_price = round(
                target.selling_price * (1.0 - action.discount_pct / 100.0), 2
            )
            # popularity_multiplier compounds up: repeated discounting makes the
            # product progressively more visible/attractive to customers
            if action.discount_pct > 0:
                target.popularity_multiplier = round(
                    target.popularity_multiplier * (1.0 + action.discount_pct / 100.0), 6
                )

        elif action.action_type == "restock":
            qty = action.restock_quantity
            if qty is None or qty <= 0:
                return self._error_observation(
                    f"restock_quantity must be a positive integer, got {qty!r}."
                )
            if target.pending_restock_quantity > 0:
                return self._error_observation(
                    f"Product '{target.name}' already has a pending restock order "
                    f"for {target.pending_restock_quantity} units arriving at step "
                    f"{target.pending_restock_arrives_at_step}. Wait for delivery first."
                )
            arrive_at = self._current_step + target.restock_lead_time
            self._pending_orders.append({
                "product_id": target.product_id,
                "quantity": qty,
                "arrive_at_step": arrive_at,
            })
            target.pending_restock_quantity = qty
            target.pending_restock_arrives_at_step = arrive_at
            restock_cost = qty * target.restock_cost_per_unit

        elif action.action_type == "place":
            zone = action.target_zone
            if zone is None:
                return self._error_observation("target_zone must be specified for 'place' actions.")
            if zone == target.zone:
                return self._error_observation(
                    f"Product '{target.name}' is already in zone {zone}."
                )
            # Count active products currently in the target zone (excluding this product)
            occupants = sum(
                1 for p in self._inventory
                if p.is_active and p.zone == zone and p.product_id != target.product_id
            )
            if occupants >= ZONE_CAPACITY[zone]:
                return self._error_observation(
                    f"Zone {zone} is at capacity ({ZONE_CAPACITY[zone]} products). "
                    f"Move another product out first."
                )
            target.zone = zone
            placement_cost = PLACEMENT_COST

        # ── 6. Gather sellable products ─────────────────────────────────────
        active = [p for p in self._inventory if p.is_active and p.quantity > 0]
        if not active:
            done = True
            self._update_state(last_reward=0.0)
            return StoremanagerObservation(
                inventory=list(self._inventory),
                current_step=self._current_step,
                max_steps=self._max_steps,
                steps_remaining=max(0, self._max_steps - self._current_step),
                cumulative_profit=self._cumulative_profit,
                last_step_profit=0.0,
                last_step_units_sold=0,
                last_expired_products=[],
                num_customers_per_step=self._num_customers,
                task_name=self._task_name,
                zone_capacity=dict(ZONE_CAPACITY),
                zone_occupancy=self._zone_occupancy(),
                zone_multipliers=dict(ZONE_MULTIPLIERS),
                holding_cost_this_step=round(holding_cost, 4),
                placement_cost_this_step=placement_cost,
                restock_cost_this_step=round(restock_cost, 4),
                done=done,
                reward=self._normalize_reward(0.0),
            )

        # ── 7. Compute pick probabilities (zone + discount + popularity) ────
        effective_weights = np.array([
            p.base_probability
            * ZONE_MULTIPLIERS[p.zone]
            * (1.0 + p.current_discount_pct / 100.0)
            * p.popularity_multiplier
            for p in active
        ])
        pick_probs = effective_weights / effective_weights.sum()

        for product, prob in zip(active, pick_probs):
            product.effective_pick_prob = round(float(prob), 6)

        # ── 8. Simulate customers ───────────────────────────────────────────
        chosen_ids = self._rng.choice(
            a=[p.product_id for p in active],
            size=self._num_customers,
            p=pick_probs,
            replace=True,
        )
        sales_counts: dict[int, int] = {}
        for pid in chosen_ids:
            sales_counts[pid] = sales_counts.get(int(pid), 0) + 1

        # ── 9. Apply sales (clamp by available quantity) ────────────────────
        step_revenue = 0.0
        step_cost = 0.0
        units_sold_total = 0

        for product in active:
            wanted = sales_counts.get(product.product_id, 0)
            sold = min(wanted, product.quantity)
            if sold > 0:
                step_revenue += sold * product.selling_price
                step_cost += sold * product.cost_price
                product.quantity -= sold
                units_sold_total += sold
                product.sales_history.append(
                    {
                        "step": self._current_step,
                        "discount_pct": product.current_discount_pct,
                        "units_sold": sold,
                    }
                )
            if product.quantity == 0:
                product.is_active = False

        # ── 10. Advance step counter ────────────────────────────────────────
        self._current_step += 1

        # ── 11. Check expiry ────────────────────────────────────────────────
        expiry_penalty = 0.0
        expired_names: List[str] = []
        for product in self._inventory:
            if product.is_active and product.expiry_step <= self._current_step:
                expiry_penalty += product.quantity * product.cost_price
                expired_names.append(product.name)
                product.is_active = False
                product.quantity = 0

        # ── 12. Reward and cumulative profit ────────────────────────────────
        step_profit = (
            step_revenue - step_cost - expiry_penalty
            - holding_cost - placement_cost - restock_cost
        )
        self._cumulative_profit += step_profit

        # ── 13. Terminal condition ───────────────────────────────────────────
        done = self._current_step >= self._max_steps or all(
            not p.is_active for p in self._inventory
        )

        # ── 14. State update and return ─────────────────────────────────────
        normalized_reward = self._normalize_reward(step_profit)
        self._update_state(last_reward=normalized_reward)

        return StoremanagerObservation(
            inventory=list(self._inventory),
            current_step=self._current_step,
            max_steps=self._max_steps,
            steps_remaining=max(0, self._max_steps - self._current_step),
            cumulative_profit=self._cumulative_profit,
            last_step_profit=step_profit,
            last_step_units_sold=units_sold_total,
            last_expired_products=expired_names,
            num_customers_per_step=self._num_customers,
            task_name=self._task_name,
            zone_capacity=dict(ZONE_CAPACITY),
            zone_occupancy=self._zone_occupancy(),
            zone_multipliers=dict(ZONE_MULTIPLIERS),
            holding_cost_this_step=round(holding_cost, 4),
            placement_cost_this_step=placement_cost,
            restock_cost_this_step=round(restock_cost, 4),
            done=done,
            reward=normalized_reward,
            metadata={
                "step_revenue": round(step_revenue, 4),
                "step_cost": round(step_cost, 4),
                "expiry_penalty": round(expiry_penalty, 4),
                "holding_cost": round(holding_cost, 4),
                "placement_cost": placement_cost,
                "restock_cost": round(restock_cost, 4),
                "step_profit": round(step_profit, 4),
            },
        )

    @property
    def state(self) -> StoremanagerState:
        return self._state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _normalize_reward(self, step_profit: float) -> float:
        """
        Map raw step profit to a meaningful reward in (0.0, 1.0) using a
        sigmoid centered at break-even (step_profit = 0).

        Formula:
            x = step_profit / per_step_target   (normalise by average needed per step)
            reward = sigmoid(k * x) = 1 / (1 + exp(-k * x))

        Where k=2 gives a useful spread:
            step_profit =  0            -> 0.50  (neutral, broke even)
            step_profit =  per_step_tgt -> 0.88  (on-track for target)
            step_profit = 2*per_step_tgt -> 0.98 (excellent step)
            step_profit = -per_step_tgt -> 0.12  (below break-even)
            step_profit = -2*per_step_tgt -> 0.02 (large penalty step)

        This gives the model a continuous, differentiable gradient signal
        across the full [0, 1] range — much more informative than a hard clamp.
        Raw dollar profit is preserved in last_step_profit and metadata.
        """
        per_step_target = self._profit_target / self._max_steps
        x = step_profit / per_step_target if per_step_target != 0 else 0.0
        return round(1.0 / (1.0 + math.exp(-2.0 * x)), 6)

    def _zone_occupancy(self) -> Dict[int, int]:
        """Count active products in each zone."""
        occupancy: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        for p in self._inventory:
            if p.is_active and p.zone in occupancy:
                occupancy[p.zone] += 1
        return occupancy

    def _recompute_pick_probs(self) -> None:
        """Recompute normalised pick probabilities for all active products (zone + popularity)."""
        active = [p for p in self._inventory if p.is_active and p.quantity > 0]
        if not active:
            return
        effs = np.array([
            p.base_probability
            * ZONE_MULTIPLIERS[p.zone]
            * (1.0 + p.current_discount_pct / 100.0)
            * p.popularity_multiplier
            for p in active
        ])
        probs = effs / effs.sum()
        for product, prob in zip(active, probs):
            product.effective_pick_prob = round(float(prob), 6)

    def _generate_inventory(self) -> List[ProductState]:
        """Generate a randomised product inventory for a new episode."""
        chosen_names = self._rng.choice(
            _PRODUCT_NAMES, size=self._num_products, replace=False
        )

        # Draw base probability weights via Dirichlet, then rescale to [0.05, 0.4]
        raw_probs = self._rng.dirichlet(alpha=[1.0] * self._num_products)
        max_raw = raw_probs.max()
        base_probs = 0.05 + raw_probs * (0.40 - 0.05) / max_raw

        # Assign initial zones: fill zone 1 first (up to capacity), then zone 2, rest to zone 3
        zone_slots = {z: ZONE_CAPACITY[z] for z in [1, 2, 3]}
        product_zones: List[int] = []
        for _ in range(self._num_products):
            for zone in [1, 2, 3]:
                if zone_slots[zone] > 0:
                    product_zones.append(zone)
                    zone_slots[zone] -= 1
                    break

        inventory: List[ProductState] = []
        for i in range(self._num_products):
            cost = round(float(self._rng.uniform(1.0, 8.0)), 2)
            margin = float(self._rng.uniform(1.3, 2.5))
            selling = round(cost * margin, 2)
            qty = int(self._rng.integers(15, 41))
            expiry_offset = int(self._rng.integers(self._expiry_offset_min, self._expiry_offset_max + 1))
            holding_rate = round(float(self._rng.uniform(0.01, 0.04)), 4)
            restock_cost_per_unit = round(cost * float(self._rng.uniform(1.0, 1.25)), 2)
            lead_time = int(self._rng.integers(1, 4))  # 1–3 steps

            inventory.append(
                ProductState(
                    product_id=i,
                    name=str(chosen_names[i]),
                    cost_price=cost,
                    selling_price=selling,
                    base_selling_price=selling,
                    quantity=qty,
                    expiry_step=self._current_step + expiry_offset,
                    base_probability=float(base_probs[i]),
                    current_discount_pct=0.0,
                    is_active=True,
                    sales_history=[],
                    zone=product_zones[i],
                    holding_cost_rate=holding_rate,
                    restock_cost_per_unit=restock_cost_per_unit,
                    restock_lead_time=lead_time,
                    pending_restock_quantity=0,
                    pending_restock_arrives_at_step=None,
                    popularity_multiplier=1.0,
                )
            )
        return inventory

    def _update_state(
        self,
        *,
        episode_id: Optional[str] = None,
        last_reward: float = 0.0,
    ) -> None:
        """Sync the internal StoremanagerState with current environment data."""
        if episode_id is not None:
            self._state.episode_id = episode_id
        self._state.step_count = self._current_step
        self._state.num_products = self._num_products
        self._state.max_steps = self._max_steps
        self._state.cumulative_profit = self._cumulative_profit
        self._state.last_reward = last_reward
        self._state.active_product_count = sum(
            1 for p in self._inventory if p.is_active
        )
        self._state.inventory_snapshot = [p.model_dump() for p in self._inventory]

    def _error_observation(self, message: str) -> StoremanagerObservation:
        """Return an observation flagging an invalid action (step does NOT advance)."""
        return StoremanagerObservation(
            inventory=list(self._inventory),
            current_step=self._current_step,
            max_steps=self._max_steps,
            steps_remaining=max(0, self._max_steps - self._current_step),
            cumulative_profit=self._cumulative_profit,
            last_step_profit=0.0,
            last_step_units_sold=0,
            last_expired_products=[],
            num_customers_per_step=self._num_customers,
            task_name=self._task_name,
            zone_capacity=dict(ZONE_CAPACITY),
            zone_occupancy=self._zone_occupancy(),
            zone_multipliers=dict(ZONE_MULTIPLIERS),
            holding_cost_this_step=0.0,
            placement_cost_this_step=0.0,
            restock_cost_this_step=0.0,
            done=False,
            reward=0.0,
            error=message,
        )


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    env = StoremanagerEnvironment(num_products=4, max_steps=6, seed=42)
    obs = env.reset()
    print("=== RESET ===")
    for p in obs.inventory:
        print(f"  [{p.product_id}] {p.name:10s}  zone={p.zone}  qty={p.quantity}  "
              f"lead={p.restock_lead_time}  hold_rate={p.holding_cost_rate:.3f}")
    print(f"  Zone occupancy: {obs.zone_occupancy}")

    actions = [
        StoremanagerAction(action_type="discount", product_id=0, discount_pct=20),
        StoremanagerAction(action_type="restock", product_id=1, restock_quantity=10),
        StoremanagerAction(action_type="place", product_id=2, target_zone=1),
        StoremanagerAction(action_type="place", product_id=3, target_zone=1),  # should fail — zone 1 at capacity
        StoremanagerAction(action_type="discount", product_id=0, discount_pct=10),
        StoremanagerAction(action_type="discount", product_id=1, discount_pct=0),
    ]

    for step_num, action in enumerate(actions):
        obs = env.step(action)
        print(f"\n=== STEP {step_num + 1} ({action.action_type.upper()}) ===")
        if obs.error:
            print(f"  ERROR: {obs.error}")
        else:
            print(f"  Reward:          {obs.reward:.2f}")
            print(f"  Holding cost:    {obs.holding_cost_this_step:.4f}")
            print(f"  Placement cost:  {obs.placement_cost_this_step:.2f}")
            print(f"  Restock cost:    {obs.restock_cost_this_step:.4f}")
            print(f"  Cumulative:      {obs.cumulative_profit:.2f}")
            print(f"  Zone occupancy:  {obs.zone_occupancy}")
            if obs.last_expired_products:
                print(f"  Expired:         {obs.last_expired_products}")
        if obs.done:
            print("  Episode done.")
            break
