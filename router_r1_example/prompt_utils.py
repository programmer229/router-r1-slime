"""Helper utilities for applying Router-R1 prompt templates."""

from __future__ import annotations

from typing import Literal

from router_r1_example import prompt_pool

PromptTemplateLiteral = Literal["qwen", "llama"]

PROMPT_MAP = {
    "qwen": prompt_pool.PROMPT_TEMPLATE_QWEN,
    "llama": prompt_pool.PROMPT_TEMPLATE_LLAMA,
}


def format_router_prompt(question: str, *, template: PromptTemplateLiteral = "qwen") -> str:
    """
    Format a question using one of the Router-R1 prompt templates.

    Args:
        question: Natural language question to embed inside the template.
        template: Either ``"qwen"`` or ``"llama"`` indicating which prompt to use.

    Returns:
        A string ready to feed to the base policy model.
    """
    template_key = template.lower()
    if template_key not in PROMPT_MAP:
        raise ValueError(f"Unsupported template '{template}'. Expected one of {list(PROMPT_MAP)}.")

    return PROMPT_MAP[template_key].format_map({"question": question})
