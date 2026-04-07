# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task configurations and graders for the Store Manager environment.

Four difficulty levels are defined, in strict increasing order of complexity:

  - easy   : 4 products, 20 steps — learn discount mechanics, no expiry pressure
  - medium : 8 products, 28 steps — discount + zone placement (zone 1 is contested)
  - hard   : 10 products, 30 steps — discount + zone + expiry timing (most expire mid-episode)
  - expert : 12 products, 25 steps — all three actions required (restock essential, aggressive expiry)

Each task has a fixed seed so grading is deterministic: the same agent always
produces the same episode given the same seed, and the score is computed as:

    score = clamp(cumulative_profit / profit_target, 0.0, 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TaskConfig:
    """Configuration for one difficulty level of the Store Manager task."""

    name: str
    description: str
    num_products: int
    max_steps: int
    num_customers: int
    expiry_offset_min: int  # min steps before a product expires
    expiry_offset_max: int  # max steps before a product expires
    seed: int               # fixed seed for deterministic episode generation
    profit_target: float    # cumulative profit at which grade == 1.0


# ── Canonical task registry ────────────────────────────────────────────────────

TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        description=(
            "4 products, 20 steps, 15 customers/step. "
            "Products expire well after the episode ends — no expiry pressure. "
            "Focus on learning the discount mechanic to boost high-margin products. "
            "Zone placement is available but low-stakes; restocking is unnecessary."
        ),
        num_products=4,
        max_steps=20,
        num_customers=15,
        expiry_offset_min=18,
        expiry_offset_max=30,
        seed=42,
        profit_target=120.0,
    ),
    "medium": TaskConfig(
        name="medium",
        description=(
            "8 products, 28 steps, 12 customers/step. "
            "Several products expire mid-episode (12-22 steps). "
            "Zone 1 has only 2 slots for 8 competing products — "
            "combine targeted discounts with zone placement to maximize clearance "
            "before expiry. Restocking is rarely necessary."
        ),
        num_products=8,
        max_steps=28,
        num_customers=12,
        expiry_offset_min=12,
        expiry_offset_max=22,
        seed=200,
        profit_target=200.0,
    ),
    "hard": TaskConfig(
        name="hard",
        description=(
            "10 products, 30 steps, 10 customers/step. "
            "Most products expire in 8-18 steps — expiry management is critical. "
            "Zone strategy (rotate zone 1 for near-expiry items) combined with "
            "discounts drives clearance. Restock fast-sellers that run out early "
            "to maintain revenue through the full episode."
        ),
        num_products=10,
        max_steps=30,
        num_customers=10,
        expiry_offset_min=8,
        expiry_offset_max=18,
        seed=456,
        profit_target=240.0,
    ),
    "expert": TaskConfig(
        name="expert",
        description=(
            "12 products, 25 steps, 8 customers/step. "
            "Aggressive expiry (4-10 steps) with limited initial stock — "
            "all three actions must be coordinated: discount near-expiry items to "
            "clear them fast, restock high-margin products before they stock out, "
            "and continuously rotate zone 1 to prioritise the most urgent products."
        ),
        num_products=12,
        max_steps=25,
        num_customers=8,
        expiry_offset_min=4,
        expiry_offset_max=10,
        seed=789,
        profit_target=190.0,
    ),
}


# ── Grader ─────────────────────────────────────────────────────────────────────

def grade(task_name: str, cumulative_profit: float) -> float:
    """
    Return a deterministic score in [0.0, 1.0] for a completed episode.

    Formula:
        score = clamp(cumulative_profit / profit_target, 0.0, 1.0)

    Args:
        task_name: One of "easy", "medium", "hard", "expert" (or a custom string).
        cumulative_profit: Total profit accumulated over the episode.

    Returns:
        Score between 0.0 (no profit) and 1.0 (hit or exceeded target).
    """
    config = TASKS.get(task_name)
    if config is None or config.profit_target <= 0:
        return 1.0
    return min(1.0, max(0.0, cumulative_profit / config.profit_target))


__all__ = ["TaskConfig", "TASKS", "grade"]
