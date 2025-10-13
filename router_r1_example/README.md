# Router-R1 Lite (slime)

This directory contains a minimal Router-R1 integration on top of the
`slime` rollout framework. The code mirrors the original Router-R1 project but
keeps everything self-contained so you can swap in this custom generation and
reward logic when launching slime jobs.

## Components
- `generate_with_router.py` – multi-turn generation loop that parses
  `<search> LLM:query </search>` actions, dispatches them through the routing
  pool, and feeds `<information> ... </information>` back to the model.
- `routing_service.py` – a thin client for the Router-R1 routing pool (uses the
  OpenRouter-compatible OpenAI API).
- `qa_em_format.py` – reward computation based on exact match with optional
  structural bonuses.
- `prompt_pool.py` – prompt scaffolding for Router-R1-style reasoning.

## Preparation
1. Install slime (and its dependencies) following the upstream instructions.
2. Install the Router-R1 requirements needed by the routing pool client:
   ```bash
   pip install openai tqdm
   ```
3. Export your routing API endpoint and key as environment variables before
   launching slime:
   ```bash
   export ROUTER_R1_API_BASE="https://openrouter.ai/api/v1"
   export ROUTER_R1_API_KEY="sk-..."
   ```
4. Prepare routing data and the base model checkpoints exactly as you would for
   the Search-R1 example.
5. Build the QA datasets locally using the copied `data_process` scripts:
   ```bash
   ./router_r1_example/prepare_router_r1_data.sh
   ```
   Override `DATA_DIR`, `TRAIN_SOURCES`, `EVAL_SOURCES`, `TEST_SOURCES`, or
   `ROUTER_R1_MODEL` to customize paths and dataset composition. By default the
   parquet files are written to `./data/router_r1/`:
   - `train_nh_<model>.parquet`
   - `test_nh_<model>.parquet`
   - `test_<source>_<model>.parquet`

## Usage
When launching slime you only need to point the `CUSTOM_ARGS` to the new module:
```bash
CUSTOM_ARGS=(
  --custom-generate-function-path router_r1_example.generate_with_router.generate
  --custom-rm-path router_r1_example.generate_with_router.reward_func
)
```

For a full training script you can copy `slime/examples/search-r1/run_qwen2.5_3B.sh`
and replace the `CUSTOM_ARGS` block with the values above. The provided
`router_r1_example/run_router_r1_qwen2.5_3B.sh` already points
`--prompt-data` to `${DATA_DIR}/train_nh_${ROUTER_R1_MODEL}.parquet`
so it works seamlessly with the generated datasets.

If you need to format raw questions into prompts, import the helper:
```python
from router_r1_example.prompt_utils import format_router_prompt

prompt = format_router_prompt("Who discovered penicillin?", template="qwen")
```

## Notes
- The routing pool expects `<search> ModelName:question </search>` with real LLM
  names. Invalid formats echo back guidance and keep the episode running.
- Set `ROUTER_R1_CONFIGS["api_base"]` and `["api_key"]` before running.
- Reward scoring still uses exact-match with structural bonuses; adjust the
  weights in `ROUTER_R1_CONFIGS` if you need different incentives.
