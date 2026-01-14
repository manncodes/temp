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
Preprocess enriched parquet (with hints) to VERL format for RL training.

Supports POPE/StepHint-style progressive hints in system prompts.

Usage:
    # Process enriched parquet with hint level 3
    python dolci_think_rl.py enriched.parquet --local_save_dir ~/data/dolci_rl \
        --hint_strategy level --hint_level 3 --partition_by_domain

    # Expand to all hint levels (creates 6x data: no hint + 5 levels)
    python dolci_think_rl.py enriched.parquet --local_save_dir ~/data/dolci_rl \
        --hint_strategy expand_all --include_no_hint --partition_by_domain

    # Random hint level per example
    python dolci_think_rl.py enriched.parquet --local_save_dir ~/data/dolci_rl \
        --hint_strategy random --partition_by_domain
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Optional, Set, List, Dict, Any

import pandas as pd
import datasets

from verl.utils.hdfs_io import copy, makedirs

# Valid abilities that can be filtered
VALID_ABILITIES = frozenset({"math", "code", "code_stdio", "instruction_following", "chat", "reasoning"})

# Hint strategies
HINT_STRATEGIES = frozenset({"none", "random", "expand_all", "level"})

# Difficulty buckets for partitioning
DIFFICULTY_BUCKETS = {
    "easy": (1, 3),      # L1-3
    "medium": (4, 6),    # L4-6
    "hard": (7, 10),     # L7-10
}


def get_difficulty_bucket(difficulty: Optional[int]) -> str:
    """Map difficulty score to bucket name."""
    if difficulty is None:
        return "unknown"
    for bucket, (low, high) in DIFFICULTY_BUCKETS.items():
        if low <= difficulty <= high:
            return bucket
    return "unknown"


def format_hint_system_prompt(hint: str, hint_level: int) -> str:
    """Format a hint into a system prompt."""
    return f"You are given a hint to help solve this problem.\n\nHint (Level {hint_level}/5): {hint}"


def extract_prompt_text(prompt) -> str:
    """Extract text content from prompt field."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            return first.get("content", "")
        return str(first)
    return str(prompt) if prompt else ""


def parse_hints(hints) -> Optional[List[str]]:
    """Parse hints field which may be JSON string or list."""
    if hints is None:
        return None
    if isinstance(hints, list):
        return hints
    if isinstance(hints, str):
        try:
            parsed = json.loads(hints)
            return parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            return None
    return None


def process_enriched_parquet(
    input_path: str,
    hint_strategy: str,
    hint_level: Optional[int],
    include_no_hint: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Process enriched parquet and apply hint strategy.

    Returns list of processed examples ready for VERL format.
    """
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Columns: {list(df.columns)}")

    # Filter to successful hint generations
    if "hint_generation_success" in df.columns:
        original_len = len(df)
        df = df[df["hint_generation_success"] == True]
        print(f"Filtered to {len(df)} rows with successful hint generation (was {original_len})")

    rng = random.Random(seed)
    processed = []

    stats = {"total": 0, "with_hints": 0, "by_level": defaultdict(int)}

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Extract fields from enriched parquet
        prompt_text = extract_prompt_text(row.get("prompt"))
        hints = parse_hints(row.get("hints"))
        difficulty = row.get("estimated_difficulty")
        ability = row.get("ability", "unknown")

        # Handle reward_model field
        reward_model = row.get("reward_model")
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except json.JSONDecodeError:
                reward_model = {}
        elif not isinstance(reward_model, dict):
            reward_model = {}

        # Get ground truth
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is None:
            gt_raw = row.get("ground_truth")
            if isinstance(gt_raw, list) and gt_raw:
                ground_truth = gt_raw[0]
            elif gt_raw is not None:
                ground_truth = gt_raw

        # Get dataset source for reward style
        dataset_source = row.get("dataset_source")
        if isinstance(dataset_source, list) and dataset_source:
            dataset_source = dataset_source[0]
        reward_style = "rule"

        # Determine which hint(s) to use based on strategy
        examples_to_create = []

        if hint_strategy == "none" or hints is None:
            examples_to_create.append((None, None))

        elif hint_strategy == "level" and hint_level is not None:
            level_idx = min(hint_level - 1, len(hints) - 1)
            examples_to_create.append((hints[level_idx], hint_level))

        elif hint_strategy == "random":
            level_idx = rng.randint(0, len(hints) - 1)
            examples_to_create.append((hints[level_idx], level_idx + 1))

        elif hint_strategy == "expand_all":
            if include_no_hint:
                examples_to_create.append((None, None))
            for lvl, hint in enumerate(hints, 1):
                examples_to_create.append((hint, lvl))

        # Create examples
        for hint_text, hint_lvl in examples_to_create:
            # Build prompt messages
            prompt_messages = []

            if hint_text:
                prompt_messages.append({
                    "role": "system",
                    "content": format_hint_system_prompt(hint_text, hint_lvl),
                })
                stats["with_hints"] += 1
                stats["by_level"][hint_lvl] += 1

            prompt_messages.append({
                "role": "user",
                "content": prompt_text,
            })

            # Build VERL format
            example = {
                "data_source": "dolci_think_rl_enriched",
                "prompt": prompt_messages,
                "ability": ability,
                "reward_model": {
                    "style": reward_style,
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "ability": ability,
                    "original_index": idx,
                    "hint_level": hint_lvl,
                    "has_hint": hint_text is not None,
                    "estimated_difficulty": int(difficulty) if pd.notna(difficulty) else None,
                    "dataset_source": dataset_source,
                },
            }

            processed.append(example)
            stats["total"] += 1

    print(f"\nProcessed {stats['total']} examples ({stats['with_hints']} with hints)")
    if stats["by_level"]:
        level_str = ", ".join(f"L{k}={v}" for k, v in sorted(stats["by_level"].items()))
        print(f"Hint levels: {level_str}")

    return processed


def split_by_ability(
    examples: List[Dict[str, Any]],
    test_per_ability: int,
    seed: int,
) -> tuple[List[Dict], List[Dict]]:
    """Split examples into train/test with uniform test distribution per ability."""
    rng = random.Random(seed)

    # Group by ability
    by_ability: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        ability = ex.get("ability", "unknown")
        by_ability[ability].append(ex)

    train, test = [], []

    for ability in sorted(by_ability.keys()):
        items = by_ability[ability]
        rng.shuffle(items)

        n_test = min(test_per_ability, len(items))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)

    print(f"\nSplit: {len(train)} train, {len(test)} test")
    for ability in sorted(by_ability.keys()):
        n_train = sum(1 for ex in train if ex.get("ability") == ability)
        n_test = sum(1 for ex in test if ex.get("ability") == ability)
        print(f"  {ability}: {n_train} train, {n_test} test")

    return train, test


def save_partitioned(
    examples: List[Dict[str, Any]],
    output_dir: str,
    split_name: str,
) -> Dict[str, Dict[str, int]]:
    """Save partitioned by ability and difficulty bucket."""
    partitions: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for ex in examples:
        ability = ex.get("ability", "unknown")
        difficulty = ex.get("extra_info", {}).get("estimated_difficulty")
        bucket = get_difficulty_bucket(difficulty)

        partitions[ability][bucket].append(ex)
        stats[ability][bucket] += 1

    for ability, buckets in partitions.items():
        ability_dir = os.path.join(output_dir, ability)
        os.makedirs(ability_dir, exist_ok=True)

        for bucket, items in buckets.items():
            if not items:
                continue

            filepath = os.path.join(ability_dir, f"{split_name}_{bucket}.parquet")
            ds = datasets.Dataset.from_list(items)
            ds.to_parquet(filepath)
            print(f"  {filepath} ({len(items)} examples)")

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess enriched parquet to VERL format with hints"
    )
    parser.add_argument(
        "input_file",
        help="Input enriched parquet file (from hint_generator.py)",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dolci_hints",
        help="Output directory",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory to copy to",
    )
    parser.add_argument(
        "--hint_strategy",
        choices=["none", "random", "expand_all", "level"],
        default="level",
        help="How to apply hints",
    )
    parser.add_argument(
        "--hint_level",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=3,
        help="Hint level for 'level' strategy (1=vague, 5=detailed)",
    )
    parser.add_argument(
        "--include_no_hint",
        action="store_true",
        help="For expand_all: also include version without hint",
    )
    parser.add_argument(
        "--partition_by_domain",
        action="store_true",
        help="Partition output by domain (ability) and difficulty",
    )
    parser.add_argument(
        "--test_per_ability",
        type=int,
        default=50,
        help="Test samples per ability",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--include_abilities",
        nargs="+",
        default=None,
        help=f"Only include these abilities: {sorted(VALID_ABILITIES)}",
    )
    parser.add_argument(
        "--exclude_abilities",
        nargs="+",
        default=None,
        help=f"Exclude these abilities: {sorted(VALID_ABILITIES)}",
    )

    args = parser.parse_args()

    # Validate
    if args.hint_strategy == "level" and args.hint_level is None:
        parser.error("--hint_level required for --hint_strategy=level")

    # Process enriched parquet
    examples = process_enriched_parquet(
        args.input_file,
        args.hint_strategy,
        args.hint_level,
        args.include_no_hint,
        args.seed,
    )

    # Filter by ability if requested
    if args.include_abilities:
        include_set = set(args.include_abilities)
        examples = [ex for ex in examples if ex.get("ability") in include_set]
        print(f"Filtered to {len(examples)} with abilities: {args.include_abilities}")

    if args.exclude_abilities:
        exclude_set = set(args.exclude_abilities)
        examples = [ex for ex in examples if ex.get("ability") not in exclude_set]
        print(f"Filtered to {len(examples)} excluding abilities: {args.exclude_abilities}")

    # Split train/test
    train_examples, test_examples = split_by_ability(
        examples, args.test_per_ability, args.seed
    )

    # Save
    output_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.partition_by_domain:
        print("\nSaving train partitioned by domain/difficulty...")
        train_stats = save_partitioned(train_examples, output_dir, "train")

        print("\nSaving test partitioned by domain/difficulty...")
        test_stats = save_partitioned(test_examples, output_dir, "test")

        # Print stats
        print("\nPartition statistics:")
        print("Train:")
        for ability in sorted(train_stats.keys()):
            buckets = train_stats[ability]
            bucket_str = ", ".join(f"{b}={c}" for b, c in sorted(buckets.items()))
            print(f"  {ability}: {bucket_str}")
        print("Test:")
        for ability in sorted(test_stats.keys()):
            buckets = test_stats[ability]
            bucket_str = ", ".join(f"{b}={c}" for b, c in sorted(buckets.items()))
            print(f"  {ability}: {bucket_str}")

    # Also save flat parquet
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    print(f"\nSaving flat parquet...")
    datasets.Dataset.from_list(train_examples).to_parquet(train_path)
    datasets.Dataset.from_list(test_examples).to_parquet(test_path)
    print(f"  {train_path} ({len(train_examples)} examples)")
    print(f"  {test_path} ({len(test_examples)} examples)")

    # Save example JSON
    if train_examples:
        example_path = os.path.join(output_dir, "train_example.json")
        with open(example_path, "w") as f:
            json.dump(train_examples[0], f, indent=2, default=str)
        print(f"  {example_path}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        print(f"\nCopying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
