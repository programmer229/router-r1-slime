"""
Utilities for running Router-R1 style multi-turn routing flows with slime.

The modules in this package provide a custom generate function, reward
computation, and a lightweight routing pool client that mirror the original
Router-R1 implementation while integrating with slime's rollout APIs.
"""

__all__ = [
    "generate_with_router",
    "qa_em_format",
    "routing_service",
    "prompt_pool",
    "prompt_utils",
]
