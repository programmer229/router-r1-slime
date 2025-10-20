# Port of the Router-R1 multi-turn generation loop adapted for slime.

from __future__ import annotations

import asyncio
import os
import re
from typing import List, Tuple

from router_r1_example.qa_em_format import compute_score_em
from router_r1_example.routing_service import access_routing_pool
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

ROUTER_R1_CONFIGS = {
    "max_turns": 4,
    "route_concurrency": 32,
    "api_base": os.getenv("ROUTER_R1_API_BASE", "https://openrouter.ai/api/v1"),
    "api_key": os.getenv("ROUTER_R1_API_KEY", ""),
    "structure_format_score": 0,
    "final_format_score": 0,
    "retrieval_score": 0.2,
}

SEMAPHORE = asyncio.Semaphore(ROUTER_R1_CONFIGS["route_concurrency"])
ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
INVALID_ROUTE_MESSAGE = (
    "\nMy previous routing request is invalid. I must use "
    "<search> LLM-Name:Question </search> with real values. Let me try again.\n"
)


async def _route_via_pool(query: str) -> Tuple[str, float]:
    """Call the routing pool in a background thread to avoid blocking asyncio."""
    async with SEMAPHORE:
        try:
            response = await asyncio.to_thread(
                access_routing_pool,
                queries=[query],
                api_base=ROUTER_R1_CONFIGS["api_base"],
                api_key=ROUTER_R1_CONFIGS["api_key"],
            )
        except Exception as exc:  # noqa: BLE001 - propagate context to caller
            print(f"[Routing] Failed to route query '{query}': {exc}")
            return "API Request Error", 0.0

    result = response.get("result", [""])
    completion_tokens = response.get("completion_tokens_list", [0.0])

    text = result[0] if result else ""
    tokens = float(completion_tokens[0]) if completion_tokens else 0.0
    return text, tokens


def postprocess_responses(resp: str) -> str:
    """Ensure the latest tag is properly closed so the parser stays stable."""
    if "</search>" in resp:
        return resp.split("</search>")[0] + "</search>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    return resp


def postprocess_predictions(prediction: str) -> List[Tuple[str, str]]:
    """Parse `<search>` and `<answer>` blocks, validating routing syntax."""
    if not isinstance(prediction, str):
        return [("noop", "")]

    actions: List[Tuple[str, str]] = []
    for match in ACTION_PATTERN.finditer(prediction):
        action = match.group(1)
        content = match.group(2).strip()

        if action == "search":
            lowered = content.lower()
            if "llm-name" in lowered or "your-query" in lowered:
                actions.append(("route invalid placeholder", content))
                continue
            if ":" not in content:
                actions.append(("route invalid missing-colon", content))
                continue
            prefix, query_body = content.split(":", 1)
            if not prefix.strip() or not query_body.strip():
                actions.append(("route invalid empty-fields", content))
                continue

        actions.append((action, content))

    if not actions:
        actions.append(("noop", ""))

    return actions


async def execute_predictions(prediction: str) -> Tuple[str, bool]:
    """
    Execute the Router-R1 action sequence.

    Multiple `<search>` blocks are processed sequentially; each successful call
    injects an `<information>` block into the next observation. Invalid routing
    attempts emit a reminder about the correct format. `<answer>` terminates the
    episode.
    """
    actions = postprocess_predictions(prediction)

    next_obs_parts: List[str] = []
    done = False

    for action, content in actions:
        if action == "search":
            route_result, _ = await _route_via_pool(content)
            lowered = route_result.strip().lower()
            if lowered in {"llm name error", "api request error", ""}:
                next_obs_parts.append("\n\n<information>None</information>\n\n")
            else:
                next_obs_parts.append(f"\n\n<information>{route_result.strip()}</information>\n\n")
        elif action == "answer":
            done = True
            break
        elif action.startswith("route invalid"):
            next_obs_parts.append(INVALID_ROUTE_MESSAGE)
        else:
            # Covers noop or any other unexpected output.
            next_obs_parts.append(INVALID_ROUTE_MESSAGE)

    return "".join(next_obs_parts), done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom `generate` function for slime rollouts."""
    if args.partial_rollout:
        raise AssertionError("Partial rollout is not supported for this function.")

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt = sample.prompt
    prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids: List[int] = []
    loss_mask: List[int] = []

    for turn in range(ROUTER_R1_CONFIGS["max_turns"]):
        payload = {"text": prompt + response, "sampling_params": sampling_params}
        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = postprocess_responses(output["text"])
        print(
            f"[RouterR1][sample={sample.index}][turn={turn}] response chunk: {cur_response}",
            flush=True,
        )
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        if next_obs:
            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response += next_obs
            response_token_ids += obs_tokens_ids
            loss_mask += [0] * len(obs_tokens_ids)

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    print(sample.response, "FULL OUTPUT", flush=True)
    finish_type = output["meta_info"]["finish_reason"]["type"]
    if finish_type == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_type == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample: Sample, **kwargs):
    """Reward function mirroring Router-R1's exact-match scoring."""
    del args, kwargs  # Unused
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    return compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        structure_format_score=ROUTER_R1_CONFIGS["structure_format_score"],
        final_format_score=ROUTER_R1_CONFIGS["final_format_score"],
        retrieval_score=ROUTER_R1_CONFIGS["retrieval_score"],
    )
