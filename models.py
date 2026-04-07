# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Store Manager Environment.

The Store Manager environment simulates a retail store where an LLM agent
acts as the store manager, applying discounts to products to maximize profit
while managing inventory and handling product expiry.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ProductState(BaseModel):
    """State of a single product in the store inventory."""

    product_id: int = Field(..., description="Unique integer identifier for the product")
    name: str = Field(..., description="Human-readable product name")
    cost_price: float = Field(..., description="Cost to the store per unit")
    selling_price: float = Field(..., description="Current selling price (after discount)")
    base_selling_price: float = Field(..., description="Original selling price before any discount")
    quantity: int = Field(..., description="Units currently in stock")
    expiry_step: int = Field(
        ..., description="Episode step number at which this product expires"
    )
    base_probability: float = Field(
        ..., description="Base pick-probability weight before discount"
    )
    current_discount_pct: float = Field(
        default=0.0, description="Active discount percentage this step (0/10/20/50)"
    )
    effective_pick_prob: float = Field(
        default=0.0,
        description="Normalised pick probability this step after applying discount (sums to 1 across all active products)",
    )
    is_active: bool = Field(
        default=True, description="False if the product is sold out or expired"
    )
    sales_history: List[Dict] = Field(
        default_factory=list,
        description="List of {step, discount_pct, units_sold} records",
    )
    # Zone / Shelf Placement fields
    zone: int = Field(
        default=2, description="Current shelf zone (1=premium/high-traffic, 2=standard, 3=back/low-traffic)"
    )
    # Restocking / Procurement fields
    holding_cost_rate: float = Field(
        default=0.02, description="Holding cost per unit per step as a fraction of cost_price"
    )
    restock_cost_per_unit: float = Field(
        default=0.0, description="Cost per unit when reordering (may exceed cost_price due to rush ordering)"
    )
    restock_lead_time: int = Field(
        default=2, description="Number of steps until a restock order is delivered"
    )
    pending_restock_quantity: int = Field(
        default=0, description="Units currently on order (not yet delivered)"
    )
    pending_restock_arrives_at_step: Optional[int] = Field(
        default=None, description="Step number when the pending restock order will arrive"
    )


class StoremanagerAction(Action):
    """Agent action: discount a product, restock inventory, or move a product to a different zone."""

    action_type: Literal["discount", "restock", "place"] = Field(
        default="discount",
        description="Type of action: 'discount' to apply a price discount, 'restock' to order more inventory, 'place' to move product to a different shelf zone",
    )
    product_id: int = Field(..., description="ID of the product to act on")
    discount_pct: Literal[0, 10, 20, 50] = Field(
        default=0, description="Discount percentage (used when action_type='discount'): 0, 10, 20, or 50"
    )
    restock_quantity: Optional[int] = Field(
        default=None, description="Number of units to order (used when action_type='restock'); must be > 0"
    )
    target_zone: Optional[Literal[1, 2, 3]] = Field(
        default=None, description="Destination zone (used when action_type='place'): 1=premium, 2=standard, 3=back"
    )


class StoremanagerObservation(Observation):
    """Full observation returned to the agent after each step."""

    inventory: List[ProductState] = Field(
        default_factory=list,
        description="Current state of all products (including inactive ones)",
    )
    current_step: int = Field(default=0, description="Current step number (0-indexed)")
    max_steps: int = Field(default=30, description="Total steps in this episode")
    steps_remaining: int = Field(
        default=30, description="Steps remaining in the episode"
    )
    cumulative_profit: float = Field(
        default=0.0, description="Total profit accumulated so far in the episode"
    )
    last_step_profit: float = Field(
        default=0.0, description="Net profit from the most recent step"
    )
    last_step_units_sold: int = Field(
        default=0, description="Total units sold across all products in the last step"
    )
    last_expired_products: List[str] = Field(
        default_factory=list,
        description="Names of products that expired at the end of the last step",
    )
    num_customers_per_step: int = Field(
        default=10, description="Fixed number of customers simulated per step"
    )
    task_name: str = Field(
        default="custom",
        description="Active difficulty task: easy / medium / hard / custom",
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the last action was invalid"
    )
    # Zone information
    zone_capacity: Dict[int, int] = Field(
        default_factory=lambda: {1: 2, 2: 4, 3: 99},
        description="Maximum number of products allowed per zone",
    )
    zone_occupancy: Dict[int, int] = Field(
        default_factory=lambda: {1: 0, 2: 0, 3: 0},
        description="Current number of active products in each zone",
    )
    zone_multipliers: Dict[int, float] = Field(
        default_factory=lambda: {1: 1.5, 2: 1.0, 3: 0.6},
        description="Pick-probability multiplier per zone",
    )
    # Step cost breakdown
    holding_cost_this_step: float = Field(
        default=0.0, description="Total holding cost charged this step across all products"
    )
    placement_cost_this_step: float = Field(
        default=0.0, description="Labor cost charged this step if a zone placement was made"
    )
    restock_cost_this_step: float = Field(
        default=0.0, description="Ordering cost charged this step if a restock order was placed"
    )


class StoremanagerState(State):
    """Internal state snapshot for debugging and orchestration."""

    num_products: int = Field(default=8, description="Number of products in this episode")
    max_steps: int = Field(default=30, description="Episode length in steps")
    cumulative_profit: float = Field(default=0.0, description="Total episode profit so far")
    last_reward: float = Field(default=0.0, description="Reward from the last step")
    active_product_count: int = Field(
        default=0, description="Number of currently active (non-expired, in-stock) products"
    )
    inventory_snapshot: List[Dict] = Field(
        default_factory=list,
        description="Serialised inventory for state introspection",
    )
