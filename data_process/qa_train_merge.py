# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import random
import datasets
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs
import argparse

from prompt_pool import PROMPT_TEMPLATE_QWEN, PROMPT_TEMPLATE_LLAMA


MATH_DATASETS = {
    "openai/gsm8k": {
        "aliases": {"math", "gsm8k", "openai/gsm8k"},
        "config": "main",
    },
    "EleutherAI/hendrycks_math": {
        "aliases": {"hendrycks", "hendrycks_math", "eleutherai/hendrycks_math"},
        "configs": (
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "precalculus",
            "prealgebra",
        ),
    },
    "nlile/hendrycks-MATH-benchmark": {
        "aliases": {
            "hendrycks-math-benchmark",
            "hendrycks_math_benchmark",
            "math-benchmark",
            "nlile/hendrycks-math-benchmark",
        },
    },
}
MATH_ALIAS_TO_CANONICAL = {}
for _canonical_name, _spec in MATH_DATASETS.items():
    MATH_ALIAS_TO_CANONICAL[_canonical_name.lower()] = _canonical_name
    for _alias in _spec["aliases"]:
        MATH_ALIAS_TO_CANONICAL[_alias.lower()] = _canonical_name

MATH_FINAL_ANSWER_PATTERN = re.compile(r"####\s*(.*)")
MATH_ANSWER_STRIP_RE = re.compile(r"[\s\n]+")
MATH_INSTRUCTION = ''


random.seed(42)


def make_prefix(dp, llm):
    question = dp['question']
    if llm == "qwen":
        prefix = PROMPT_TEMPLATE_QWEN.format_map({"question": question})
    elif llm == "llama":
        prefix = PROMPT_TEMPLATE_LLAMA.format_map({"question": question})
    else:
        raise NotImplementedError

    return prefix


def _is_math_source(data_source: str) -> bool:
    return data_source.lower() in MATH_ALIAS_TO_CANONICAL


def _canonicalize_source(data_source: str) -> str:
    return MATH_ALIAS_TO_CANONICAL.get(data_source.lower(), data_source)


def _load_math_split(canonical_source: str, split_priority):
    spec = MATH_DATASETS.get(canonical_source)
    if spec is None:
        raise ValueError(f"Unsupported math dataset: {canonical_source}")

    if canonical_source == "openai/gsm8k":
        dataset_dict = datasets.load_dataset(canonical_source, spec["config"])
        for split_name in split_priority:
            if split_name in dataset_dict:
                return dataset_dict[split_name], split_name
        raise ValueError(f"None of the requested splits {split_priority} are available for {canonical_source}")

    if canonical_source == "EleutherAI/hendrycks_math":
        selected_splits = []
        for config in spec["configs"]:
            dataset_dict = datasets.load_dataset(canonical_source, config)
            chosen_split = None
            chosen_split_name = None
            for split_name in split_priority:
                if split_name in dataset_dict:
                    chosen_split = dataset_dict[split_name]
                    chosen_split_name = split_name
                    break
            if chosen_split is None:
                continue
            chosen_split = chosen_split.add_column("math_source_config", [config] * len(chosen_split))
            selected_splits.append((chosen_split, chosen_split_name))

        if not selected_splits:
            raise ValueError(f"None of the requested splits {split_priority} are available for {canonical_source}")

        if len(selected_splits) == 1:
            dataset_split, split_name = selected_splits[0]
            return dataset_split, split_name

        datasets_to_concat = [item[0] for item in selected_splits]
        # assume all splits share the same selected split name (first available within priority)
        selected_split_name = selected_splits[0][1]
        return datasets.concatenate_datasets(datasets_to_concat), selected_split_name

    if canonical_source == "nlile/hendrycks-MATH-benchmark":
        dataset_dict = datasets.load_dataset(canonical_source)
        for split_name in split_priority:
            if split_name in dataset_dict:
                return dataset_dict[split_name], split_name
        raise ValueError(
            f"None of the requested splits {split_priority} are available for {canonical_source}"
        )

    raise NotImplementedError(f"Unhandled math dataset: {canonical_source}")


def _format_question(raw_question: str, is_math: bool) -> str:
    question = raw_question.strip()
    if is_math:
        if '####' not in question:
            question = f"{question}\n\n{MATH_INSTRUCTION}"
        return question
    if question and question[-1] != '?':
        question += '?'
    return question


def _extract_question(example: dict, is_math: bool) -> str:
    if not is_math:
        return example['question']
    return example.get('question') or example.get('problem') or example.get('prompt') or ""


def _extract_math_ground_truth(answer_text: str) -> str:
    if not isinstance(answer_text, str):
        return ""
    match = MATH_FINAL_ANSWER_PATTERN.search(answer_text)
    if match:
        final_answer = match.group(1)
    else:
        final_answer = answer_text
    final_answer = final_answer.strip()
    final_answer = final_answer.split('\n')[0]
    # remove redundant whitespaces within the answer and trailing punctuation
    final_answer = MATH_ANSWER_STRIP_RE.sub(' ', final_answer).strip()
    final_answer = final_answer.replace(',', '').replace('$', '')
    if final_answer.endswith('.'):  # avoid trailing periods on numeric answers
        final_answer = final_answer[:-1]
    return final_answer


# python data_process/qa_train_merge.py --data_sources nq,hotpotqa --model qwen
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_sources', default='nq,hotpotqa')
    parser.add_argument('--model', type=str, default='qwen')

    args = parser.parse_args()
    model_name = args.model.lower()
    data_sources = [src.strip() for src in args.data_sources.split(',') if src.strip()]
    all_dataset = []

    folder_path = Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    for data_source in data_sources:
        is_math_source = _is_math_source(data_source)
        canonical_source = _canonicalize_source(data_source)

        if is_math_source:
            train_dataset, _ = _load_math_split(canonical_source, ("train",))
        else:
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', canonical_source)
            train_dataset = dataset['train']

        # random sample (cap at dataset size)
        sample_size = min(len(train_dataset), 7000)
        if sample_size < len(train_dataset):
            sampled_indices = random.sample(list(range(len(train_dataset))), sample_size)
            train_dataset = train_dataset.select(sampled_indices)

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question_text = _extract_question(example, is_math_source)
                question_text = _format_question(question_text, is_math_source)
                question = make_prefix({"question": question_text}, llm=model_name)

                if is_math_source:
                    raw_answer = example.get('answer', '') or example.get('solution', '')
                    solution = _extract_math_ground_truth(raw_answer)
                    solution = {
                        "target" : [solution],
                    }
                else:
                    solution = {
                        "target": example['golden_answers'],
                    }

                data = {
                    "data_source": canonical_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "math" if is_math_source else "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                if is_math_source:
                    math_subject = example.get('math_source_config') or example.get('subject')
                    if math_subject:
                        data["extra_info"]["math_subject"] = math_subject
                    math_level = example.get('level')
                    if math_level is not None:
                        data["extra_info"]["math_level"] = math_level
                    math_unique_id = example.get('unique_id') or example.get('id')
                    if math_unique_id:
                        data["extra_info"]["math_unique_id"] = math_unique_id
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train_nh_{}.parquet'.format(model_name)))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
