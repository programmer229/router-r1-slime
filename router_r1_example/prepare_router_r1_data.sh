#!/bin/bash
# Generate Router-R1 training/validation/test datasets using the local data_process scripts.

set -euo pipefail

DATA_DIR=${DATA_DIR:-"./data/router_r1"}
MODEL=${ROUTER_R1_MODEL:-"qwen"}
TRAIN_SOURCES=${TRAIN_SOURCES:-"nq,hotpotqa"}
EVAL_SOURCES=${EVAL_SOURCES:-"nq,hotpotqa"}
TEST_SOURCES=${TEST_SOURCES:-"nq"}

echo "[Router-R1] Writing datasets to ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

python data_process/qa_train_merge.py \
  --local_dir "${DATA_DIR}" \
  --data_sources "${TRAIN_SOURCES}" \
  --model "${MODEL}"

python data_process/qa_test_merge.py \
  --local_dir "${DATA_DIR}" \
  --data_sources "${EVAL_SOURCES}" \
  --model "${MODEL}"

python data_process/qa_test_gen.py \
  --local_dir "${DATA_DIR}" \
  --data_sources "${TEST_SOURCES}" \
  --model "${MODEL}"

echo "[Router-R1] Finished building datasets in ${DATA_DIR}"
