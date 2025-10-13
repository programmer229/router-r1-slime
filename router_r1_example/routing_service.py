"""
Lightweight client for the Router-R1 routing pool.

This module ports the original Router-R1 `route_service.py` so that we can call
specialized LLM APIs from within slime examples. The entry function is
`access_routing_pool`, which accepts a list of routing queries written in the
form ``ModelName: question`` and returns model responses along with the number
of completion tokens consumed.
"""

from __future__ import annotations

import time
from multiprocessing.dummy import Pool as ThreadPool
from typing import Iterable, Sequence

import openai
from tqdm import tqdm

_cached_client: openai.OpenAI | None = None


def get_client(
    base_url: str = "",
    api_key: str = "",
    max_retries: int = 2,
    timeout: int = 60,
) -> openai.OpenAI:
    """Return a cached OpenAI client instance."""
    global _cached_client
    if _cached_client is None:
        _cached_client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )
    return _cached_client


def get_llm_response_via_api(
    prompt: str,
    *,
    llm_model: str,
    base_url: str,
    api_key: str,
    tau: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = 42,
    max_trials: int = 500,
    time_gap: int = 5,
) -> tuple[str, int]:
    """
    Dispatch a single prompt to the routing pool and return its response.

    The implementation mirrors the upstream Router-R1 helper, retrying failed
    calls and capturing the number of completion tokens for cost accounting.
    """
    client = get_client(base_url=base_url, api_key=api_key)

    while max_trials:
        max_trials -= 1
        try:
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=tau,
                top_p=top_p,
                seed=seed,
                max_tokens=512,
            )
            break
        except Exception as exc:  # noqa: BLE001 - propagate context to caller
            print(exc)
            if "request timed out" in str(exc).strip().lower():
                break
            print("Retrying...")
            time.sleep(time_gap)
    else:
        raise RuntimeError("Reached maximum retry attempts while querying routing pool.")

    if completion is None:  # pragma: no cover - defensive
        raise RuntimeError("Routing pool call exhausted retries without a response.")

    message = completion.choices[0].message.content
    completion_tokens = completion.usage.completion_tokens
    return message, completion_tokens


OPENROUTER_MODEL_MAP = {
    "qwen/qwen2.5-7b-instruct": "qwen/Qwen2.5-7B-Instruct",
    "meta/llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/mixtral-8x22b-instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "google/gemma-2-27b-it": "google/Gemma-2-27B-It",
    "writer/palmyra-creative-122b": "writer/Palmyra-Creative-122B",
    "nvidia/llama3-chatqa-1.5-8b": "nvidia/Llama-3.1-ChatQA-1.5-8B",
    "nvidia/llama-3.1-nemotron-51b-instruct": "nvidia/Llama-3.1-Nemotron-51B-Instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1": "nvidia/Llama-3.3-Nemotron-Super-49B-v1",
    "ibm/granite-3.0-8b-instruct": "ibm/Granite-3.0-8B-Instruct",
}

API_PRICE_1M_TOKENS = {
    "qwen/Qwen2.5-7B-Instruct": 0.3,
    "meta-llama/llama-3.1-70b-instruct": 0.88,
    "meta-llama/llama-3.1-8b-instruct": 0.18,
    "mistralai/Mistral-7B-Instruct-v0.3": 0.2,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 1.2,
    "google/Gemma-2-27B-It": 0.8,
    "writer/Palmyra-Creative-122B": 1.8,
    "nvidia/Llama-3.1-ChatQA-1.5-8B": 0.18,
    "nvidia/Llama-3.1-Nemotron-51B-Instruct": 0.18,
    "nvidia/Llama-3.3-Nemotron-Super-49B-v1": 0.18,
    "ibm/Granite-3.0-8B-Instruct": 0.18,
}

AGENT_PROMPT = """
You are a helpful assistant participating in a multi-agent reasoning process,
where a base model routes sub-questions to specialized models like you.

Your task is to either:
  • Answer the question directly, providing a concise explanation, or
  • Offer relevant background, context, or insights that help answer it.

If you truly cannot assist, clearly state that and advise the base model to
consult another LLM.

Constraints:
  • Keep responses concise (under 512 tokens).
  • Stay on topic.

Sub-question:
{query}
"""


def request_task(data: tuple[int, str, float, str, str, str]) -> tuple[int, str, float]:
    q_id, query_text, tau, llm_name, api_base, api_key = data
    if not llm_name:
        return q_id, "LLM Name Error", 0.0

    llm_identifier = OPENROUTER_MODEL_MAP.get(llm_name, llm_name)
    input_prompt = AGENT_PROMPT.format_map({"query": query_text})
    try:
        response, completion_tokens = get_llm_response_via_api(
            input_prompt,
            llm_model=llm_identifier,
            base_url=api_base,
            api_key=api_key,
            tau=tau,
        )
    except Exception as exc:  # noqa: BLE001 - caller will inspect response text
        print(exc)
        response = "API Request Error"
        completion_tokens = 0

    price_per_million = API_PRICE_1M_TOKENS.get(llm_identifier, 0.0)
    return q_id, response, completion_tokens * price_per_million


def check_llm_name(target_llm: str) -> tuple[str, float]:
    """Map a free-form model name to the canonical OpenRouter identifier."""
    target_llm_lower = target_llm.lower()
    tau = 0.0
    llm_name = ""

    if "qwen" in target_llm_lower:
        llm_name = "qwen/qwen2.5-7b-instruct"
    elif "palmyra" in target_llm_lower or "creative" in target_llm_lower:
        llm_name = "writer/palmyra-creative-122b"
    elif "llama" in target_llm_lower:
        if "70b" in target_llm_lower:
            llm_name = "meta/llama-3.1-70b-instruct"
        elif "51b" in target_llm_lower:
            llm_name = "nvidia/llama-3.1-nemotron-51b-instruct"
        elif "49b" in target_llm_lower:
            llm_name = "nvidia/llama-3.3-nemotron-super-49b-v1"
        elif "8b" in target_llm_lower:
            if "chatqa" in target_llm_lower:
                llm_name = "nvidia/llama3-chatqa-1.5-8b"
            else:
                llm_name = "meta/llama-3.1-8b-instruct"
    elif "mistral" in target_llm_lower:
        llm_name = "mistralai/mistral-7b-instruct-v0.3"
    elif "mixtral" in target_llm_lower:
        llm_name = "mistralai/mixtral-8x22b-instruct-v0.1"
    elif "granite" in target_llm_lower:
        llm_name = "ibm/granite-3.0-8b-instruct"
    elif "gemma" in target_llm_lower:
        llm_name = "google/gemma-2-27b-it"
        tau = 0.1

    return OPENROUTER_MODEL_MAP.get(llm_name, llm_name), tau


def build_routing_tasks(
    queries: Sequence[str],
    api_base: str,
    api_key: str,
) -> list[tuple[int, str, float, str, str, str]]:
    """Construct routing job tuples from user-specified queries."""
    task_args = []
    for q_id, single_query in enumerate(queries):
        parts = single_query.split(":", 1)
        if len(parts) != 2:
            task_args.append((q_id, "", 0.0, "", api_base, api_key))
            continue
        target_llm, query_text = parts[0].strip(), parts[1]
        llm_name, tau = check_llm_name(target_llm=target_llm)
        task_args.append((q_id, query_text, tau, llm_name, api_base, api_key))
    return task_args


def access_routing_pool(
    *,
    queries: Iterable[str],
    api_base: str,
    api_key: str,
) -> dict[str, list]:
    """
    Query a set of routing requests concurrently.

    Returns a dictionary with keys:
      • ``result`` – list of model responses ordered by query id
      • ``completion_tokens_list`` – approximate token usage translated into cost
    """
    task_args = build_routing_tasks(list(queries), api_base=api_base, api_key=api_key)
    results: list[tuple[int, str, float]] = []

    with ThreadPool(10) as pool:
        for entry in tqdm(pool.imap_unordered(request_task, task_args), total=len(task_args), desc="Routing", ncols=100):
            results.append(entry)

    results.sort(key=lambda item: item[0])
    responses = [response for _, response, _ in results]
    completion_tokens = [tokens for _, _, tokens in results]

    return {"result": responses, "completion_tokens_list": completion_tokens}
