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
Preprocess the allenai/Dolci-Think-RL dataset to parquet format for RL training.

Supports POPE/StepHint-style progressive hints in system prompts.

Usage:
    # Basic usage
    python dolci_think_rl.py --local_save_dir ~/data/dolci_think_rl

    # With hints from enriched parquet (single hint level)
    python dolci_think_rl.py --local_save_dir ~/data/dolci_think_rl \
        --hints_file enriched.parquet --hint_level 3

    # Expand to all hint levels (creates N copies per example)
    python dolci_think_rl.py --local_save_dir ~/data/dolci_think_rl \
        --hints_file enriched.parquet --hint_strategy expand_all

    # Random hint level per example
    python dolci_think_rl.py --local_save_dir ~/data/dolci_think_rl \
        --hints_file enriched.parquet --hint_strategy random
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Optional, Set, List, Dict, Any

import datasets

from verl.utils.hdfs_io import copy, makedirs

# Valid abilities that can be filtered
VALID_ABILITIES = frozenset({"math", "code", "instruction_following", "chat", "reasoning"})

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


def get_ability_from_source(dataset_source: str) -> str:
    """Map dataset_source to ability category."""
    source_to_ability = {
        "math": "math",
        "code_stdio": "code_stdio",
        "code": "code",
        "ifeval": "instruction_following",
        "general-quality": "chat",
        "general-quality_ref": "chat",
    }
    return source_to_ability.get(dataset_source, "reasoning")


def get_reward_style(dataset_source: str) -> str:
    """Determine reward style based on dataset source."""
    if dataset_source in ("math", "code"):
        return "rule"
    return "rule"


def extract_ground_truth(ground_truth_list):
    """Extract ground truth from the list format."""
    if ground_truth_list is None or len(ground_truth_list) == 0:
        return None
    return ground_truth_list[0]


def extract_dataset_info(dataset_list):
    """Extract dataset info from the list format."""
    if dataset_list is None or len(dataset_list) == 0:
        return None
    return dataset_list[0]


def validate_abilities(abilities: Optional[list], arg_name: str) -> Optional[Set[str]]:
    """Validate that provided abilities are recognized."""
    if abilities is None:
        return None
    abilities_set = set(abilities)
    invalid = abilities_set - VALID_ABILITIES
    if invalid:
        raise ValueError(
            f"Invalid abilities in --{arg_name}: {sorted(invalid)}. "
            f"Valid abilities are: {sorted(VALID_ABILITIES)}"
        )
    return abilities_set


def create_ability_filter(
    include_abilities: Optional[Set[str]],
    exclude_abilities: Optional[Set[str]],
):
    """Create a filter function for abilities."""
    def filter_fn(example) -> bool:
        dataset_source = example.get("dataset", [None])[0]
        ability = get_ability_from_source(dataset_source)
        if include_abilities is not None and ability not in include_abilities:
            return False
        if exclude_abilities is not None and ability in exclude_abilities:
            return False
        return True
    return filter_fn


def split_uniform_test_from_dataset(
    dataset: datasets.Dataset,
    samples_per_ability: Optional[int] = 50,
    seed: int = 42,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Split dataset into train and test with uniform distribution across abilities."""
    if len(dataset) == 0:
        return dataset, dataset

    ability_to_indices: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        example = dataset[idx]
        dataset_source = example.get("dataset", [None])[0]
        ability = get_ability_from_source(dataset_source)
        if ability not in ability_to_indices:
            ability_to_indices[ability] = []
        ability_to_indices[ability].append(idx)

    min_count = min(len(indices) for indices in ability_to_indices.values())
    if samples_per_ability is None:
        samples_per_ability = min_count
    else:
        samples_per_ability = min(samples_per_ability, min_count)

    if samples_per_ability == 0:
        print("Warning: At least one ability has zero examples")
        return dataset, dataset.select([])

    rng = random.Random(seed)
    test_indices_set: set[int] = set()
    for ability in sorted(ability_to_indices.keys()):
        indices = ability_to_indices[ability]
        sampled = rng.sample(indices, samples_per_ability)
        test_indices_set.update(sampled)

    all_indices = set(range(len(dataset)))
    train_indices = sorted(all_indices - test_indices_set)
    test_indices = sorted(test_indices_set)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    print(f"Split dataset with uniform test distribution:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Test samples per ability: {samples_per_ability}")
    print(f"  Abilities found: {sorted(ability_to_indices.keys())}")
    for ability in sorted(ability_to_indices.keys()):
        original_count = len(ability_to_indices[ability])
        print(f"    {ability}: {original_count} total, {samples_per_ability} in test")
    print(f"  Train size: {len(train_indices)}")
    print(f"  Test size: {len(test_indices)}")

    return dataset.select(train_indices), dataset.select(test_indices)


# --- Hint Integration ---

def load_hints_mapping(hints_file: str) -> Dict[str, Dict[str, Any]]:
    """Load hints from enriched parquet/jsonl, keyed by prompt content."""
    import pandas as pd

    if hints_file.endswith(".parquet"):
        df = pd.read_parquet(hints_file)
    else:
        with open(hints_file) as f:
            data = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(data)

    hints_map = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        # Extract prompt text as key
        prompt = row.get("prompt")
        if isinstance(prompt, list) and prompt:
            if isinstance(prompt[0], dict):
                key = prompt[0].get("content", "")
            else:
                key = str(prompt[0])
        else:
            key = str(prompt) if prompt else ""

        if not key:
            continue

        # Parse hints if stored as JSON string
        hints = row.get("hints")
        if isinstance(hints, str):
            try:
                hints = json.loads(hints)
            except json.JSONDecodeError:
                hints = None

        hints_map[key] = {
            "hints": hints,
            "estimated_difficulty": row.get("estimated_difficulty"),
            "hint_generation_success": row.get("hint_generation_success", False),
        }

    print(f"Loaded hints for {len(hints_map)} examples from {hints_file}")
    return hints_map


def format_hint_system_prompt(hint: str, hint_level: int) -> str:
    """Format a hint into a system prompt."""
    return f"You are given a hint to help solve this problem.\n\nHint (Level {hint_level}/5): {hint}"


def get_hint_for_example(
    prompt_text: str,
    hints_map: Dict[str, Dict[str, Any]],
    hint_level: Optional[int],
    hint_strategy: str,
    rng: random.Random,
) -> Optional[tuple[str, int]]:
    """
    Get hint for an example based on strategy.

    Returns:
        Tuple of (hint_text, hint_level) or None if no hint.
    """
    if hint_strategy == "none":
        return None

    hint_data = hints_map.get(prompt_text)
    if not hint_data or not hint_data.get("hints"):
        return None

    hints = hint_data["hints"]
    if not isinstance(hints, list) or len(hints) == 0:
        return None

    if hint_strategy == "level" and hint_level is not None:
        # Use specific level (1-indexed)
        idx = min(hint_level - 1, len(hints) - 1)
        return hints[idx], hint_level

    if hint_strategy == "random":
        # Random level
        idx = rng.randint(0, len(hints) - 1)
        return hints[idx], idx + 1

    return None


def expand_examples_with_all_hints(
    dataset: datasets.Dataset,
    hints_map: Dict[str, Dict[str, Any]],
    include_no_hint: bool = True,
) -> List[Dict[str, Any]]:
    """
    Expand dataset to include all hint levels as separate examples.

    For each example with hints, creates up to 6 versions:
    - 1 with no hint (if include_no_hint)
    - 5 with each hint level (L1-L5)
    """
    expanded = []

    for idx in range(len(dataset)):
        example = dataset[idx]
        prompt = example.get("prompt", "")

        # Get prompt text for hint lookup
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            prompt_text = prompt

        hint_data = hints_map.get(prompt_text)
        hints = hint_data.get("hints") if hint_data else None

        if not hints or not isinstance(hints, list):
            # No hints available, just add the original
            expanded.append(dict(example))
            continue

        # Add version without hint
        if include_no_hint:
            expanded.append(dict(example))

        # Add version for each hint level
        for level, hint in enumerate(hints, 1):
            ex_copy = dict(example)
            ex_copy["_hint_level"] = level
            ex_copy["_hint_text"] = hint
            expanded.append(ex_copy)

    return expanded


def save_partitioned_by_ability_and_difficulty(
    dataset: datasets.Dataset,
    output_dir: str,
    split_name: str,
) -> Dict[str, Dict[str, int]]:
    """
    Save dataset partitioned by ability (domain) and difficulty bucket.

    Creates directory structure:
        output_dir/
            math/
                train_easy.parquet
                train_medium.parquet
                train_hard.parquet
            code/
                train_easy.parquet
                ...
            reasoning/
                ...

    Returns:
        Dict mapping ability -> bucket -> count
    """
    # Group examples by ability and difficulty
    partitions: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for idx in range(len(dataset)):
        example = dataset[idx]

        # Get ability
        ability = example.get("ability", "unknown")

        # Get difficulty from extra_info
        extra_info = example.get("extra_info", {})
        difficulty = extra_info.get("estimated_difficulty")
        bucket = get_difficulty_bucket(difficulty)

        partitions[ability][bucket].append(dict(example))
        stats[ability][bucket] += 1

    # Save each partition
    for ability, buckets in partitions.items():
        ability_dir = os.path.join(output_dir, ability)
        os.makedirs(ability_dir, exist_ok=True)

        for bucket, examples in buckets.items():
            if not examples:
                continue

            filename = f"{split_name}_{bucket}.parquet"
            filepath = os.path.join(ability_dir, filename)

            partition_dataset = datasets.Dataset.from_list(examples)
            partition_dataset.to_parquet(filepath)

            print(f"  Saved {filepath} ({len(examples)} examples)")

    return dict(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess allenai/Dolci-Think-RL dataset")
    parser.add_argument("--local_dir", default=None, help="Deprecated. Use --local_save_dir instead.")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy processed data to.")
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local path to the raw dataset if already downloaded.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dolci_think_rl",
        help="Directory to save the preprocessed dataset.",
    )
    parser.add_argument(
        "--test_samples_per_ability",
        type=int,
        default=50,
        help="Number of test samples per ability for uniform distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split. Default: 42",
    )
    parser.add_argument(
        "--filter_null_ground_truth",
        action="store_true",
        help="Filter out examples with null ground_truth values.",
    )
    parser.add_argument(
        "--exclude_abilities",
        nargs="+",
        default=None,
        metavar="ABILITY",
        help=f"Abilities to exclude. Valid: {', '.join(sorted(VALID_ABILITIES))}",
    )
    parser.add_argument(
        "--include_abilities",
        nargs="+",
        default=None,
        metavar="ABILITY",
        help=f"Only include these abilities. Valid: {', '.join(sorted(VALID_ABILITIES))}",
    )
    # Hint arguments
    parser.add_argument(
        "--hints_file",
        default=None,
        help="Path to enriched parquet/jsonl with hints (from hint_generator.py)",
    )
    parser.add_argument(
        "--hint_strategy",
        choices=["none", "random", "expand_all", "level"],
        default="none",
        help=(
            "How to apply hints: "
            "'none' = no hints, "
            "'random' = random hint level per example, "
            "'expand_all' = create N copies with each hint level, "
            "'level' = use specific level (requires --hint_level)"
        ),
    )
    parser.add_argument(
        "--hint_level",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Hint level to use (1-5, where 1=vague, 5=detailed). Required if --hint_strategy=level",
    )
    parser.add_argument(
        "--include_no_hint",
        action="store_true",
        help="When using expand_all, also include version without hint",
    )
    parser.add_argument(
        "--partition_by_domain",
        action="store_true",
        help="Partition output by domain (ability) and difficulty bucket",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.include_abilities is not None and args.exclude_abilities is not None:
        parser.error("--include_abilities and --exclude_abilities are mutually exclusive")

    if args.hint_strategy == "level" and args.hint_level is None:
        parser.error("--hint_level is required when --hint_strategy=level")

    if args.hint_strategy != "none" and args.hints_file is None:
        parser.error(f"--hints_file is required when --hint_strategy={args.hint_strategy}")

    include_abilities = validate_abilities(args.include_abilities, "include_abilities")
    exclude_abilities = validate_abilities(args.exclude_abilities, "exclude_abilities")

    data_source = "allenai/Dolci-Think-RL"

    print(f"Loading the {data_source} dataset from HuggingFace...", flush=True)

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    if "train" in dataset:
        full_dataset = dataset["train"]
        full_dataset = full_dataset.shuffle(42)
    else:
        full_dataset = datasets.concatenate_datasets(list(dataset.values()))

    print(f"Loaded {len(full_dataset)} examples from {data_source}", flush=True)

    if args.filter_null_ground_truth:
        original_size = len(full_dataset)
        full_dataset = full_dataset.filter(
            lambda x: x.get("ground_truth") is not None and len(x.get("ground_truth", [])) > 0
        )
        print(f"Filtered out {original_size - len(full_dataset)} examples with null ground_truth", flush=True)

    if include_abilities is not None or exclude_abilities is not None:
        original_size = len(full_dataset)
        ability_filter = create_ability_filter(include_abilities, exclude_abilities)
        full_dataset = full_dataset.filter(ability_filter)
        filter_desc = (
            f"included abilities: {sorted(include_abilities)}" if include_abilities
            else f"excluded abilities: {sorted(exclude_abilities)}"
        )
        print(f"Filtered by abilities ({filter_desc}): {original_size} -> {len(full_dataset)} examples", flush=True)

    train_dataset, test_dataset = split_uniform_test_from_dataset(
        full_dataset,
        samples_per_ability=args.test_samples_per_ability,
        seed=args.seed,
    )

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}", flush=True)

    # Load hints if specified
    hints_map = {}
    if args.hints_file:
        hints_map = load_hints_mapping(args.hints_file)

    # For expand_all strategy, expand the datasets
    if args.hint_strategy == "expand_all":
        print(f"Expanding datasets with all hint levels (include_no_hint={args.include_no_hint})...")
        train_expanded = expand_examples_with_all_hints(train_dataset, hints_map, args.include_no_hint)
        test_expanded = expand_examples_with_all_hints(test_dataset, hints_map, args.include_no_hint)
        print(f"Expanded train: {len(train_dataset)} -> {len(train_expanded)}")
        print(f"Expanded test: {len(test_dataset)} -> {len(test_expanded)}")
        # Convert back to dataset
        train_dataset = datasets.Dataset.from_list(train_expanded)
        test_dataset = datasets.Dataset.from_list(test_expanded)

    # RNG for random hint selection
    hint_rng = random.Random(args.seed)

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_text = example.get("prompt", "")

            # Check for pre-expanded hint (from expand_all)
            hint_level = example.get("_hint_level")
            hint_text = example.get("_hint_text")

            # If not pre-expanded, get hint based on strategy
            if hint_text is None and args.hint_strategy not in ("none", "expand_all"):
                hint_result = get_hint_for_example(
                    prompt_text, hints_map, args.hint_level, args.hint_strategy, hint_rng
                )
                if hint_result:
                    hint_text, hint_level = hint_result

            # Build prompt with optional system message for hint
            prompt_messages = []

            if hint_text:
                # Add hint as system message
                prompt_messages.append({
                    "role": "system",
                    "content": format_hint_system_prompt(hint_text, hint_level),
                })

            prompt_messages.append({
                "role": "user",
                "content": prompt_text,
            })

            # Extract ground truth
            ground_truth = extract_ground_truth(example.get("ground_truth"))

            # Get dataset source category
            dataset_source = example.get("dataset", [None])[0]
            ability = get_ability_from_source(dataset_source)
            reward_style = get_reward_style(dataset_source)

            # Get difficulty from hints_map or example
            difficulty = None
            if prompt_text in hints_map:
                difficulty = hints_map[prompt_text].get("estimated_difficulty")
            if difficulty is None:
                difficulty = example.get("estimated_difficulty")

            data = {
                "data_source": data_source,
                "prompt": prompt_messages,
                "ability": ability,
                "reward_model": {
                    "style": reward_style,
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "ability": ability,
                    "split": split,
                    "index": idx,
                    "original_prompt": prompt_text,
                    "original_dataset": example.get("original_dataset"),
                    "dataset_source": dataset_source,
                    "custom_id": example.get("custom_id"),
                    "constraint_type": example.get("constraint_type"),
                    "constraint": example.get("constraint"),
                    # Hint metadata
                    "hint_level": hint_level,
                    "has_hint": hint_text is not None,
                    "estimated_difficulty": difficulty,
                },
            }
            return data

        return process_fn

    print("Processing train dataset...", flush=True)
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    print("Processing test dataset...", flush=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Handle save directory
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    hdfs_dir = args.hdfs_dir

    if args.partition_by_domain:
        # Save partitioned by ability and difficulty
        print("Saving train dataset partitioned by domain and difficulty...", flush=True)
        train_stats = save_partitioned_by_ability_and_difficulty(train_dataset, local_dir, "train")

        print("Saving test dataset partitioned by domain and difficulty...", flush=True)
        test_stats = save_partitioned_by_ability_and_difficulty(test_dataset, local_dir, "test")

        # Also save flat versions for convenience
        train_path = os.path.join(local_dir, "train.parquet")
        test_path = os.path.join(local_dir, "test.parquet")
        train_dataset.to_parquet(train_path)
        test_dataset.to_parquet(test_path)
        print(f"Also saved flat: {train_path}, {test_path}")

        # Print partition statistics
        print("\nPartition statistics:")
        print("  Train:")
        for ability in sorted(train_stats.keys()):
            buckets = train_stats[ability]
            bucket_str = ", ".join(f"{b}={c}" for b, c in sorted(buckets.items()))
            print(f"    {ability}: {bucket_str}")
        print("  Test:")
        for ability in sorted(test_stats.keys()):
            buckets = test_stats[ability]
            bucket_str = ", ".join(f"{b}={c}" for b, c in sorted(buckets.items()))
            print(f"    {ability}: {bucket_str}")
    else:
        train_path = os.path.join(local_dir, "train.parquet")
        test_path = os.path.join(local_dir, "test.parquet")

        print(f"Saving train dataset to {train_path}...", flush=True)
        train_dataset.to_parquet(train_path)

        print(f"Saving test dataset to {test_path}...", flush=True)
        test_dataset.to_parquet(test_path)

    # Save example JSONs
    if len(train_dataset) > 0:
        example = train_dataset[0]
        example_path = os.path.join(local_dir, "train_example.json")
        with open(example_path, "w") as f:
            json.dump(example, f, indent=2, default=str)
        print(f"Saved train example to {example_path}", flush=True)

    if len(test_dataset) > 0:
        example = test_dataset[0]
        example_path = os.path.join(local_dir, "test_example.json")
        with open(example_path, "w") as f:
            json.dump(example, f, indent=2, default=str)
        print(f"Saved test example to {example_path}", flush=True)

    if hdfs_dir is not None:
        print(f"Copying to HDFS: {hdfs_dir}...", flush=True)
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print("HDFS copy complete.", flush=True)

    print("\nPreprocessing complete!", flush=True)
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    print(f"  Output: {local_dir}")
    if args.partition_by_domain:
        print("  Structure: <domain>/<split>_<difficulty>.parquet")

    # Print hint statistics
    if args.hint_strategy != "none":
        train_with_hints = sum(1 for i in range(len(train_dataset)) if train_dataset[i].get("extra_info", {}).get("has_hint"))
        test_with_hints = sum(1 for i in range(len(test_dataset)) if test_dataset[i].get("extra_info", {}).get("has_hint"))
        print(f"  Train with hints: {train_with_hints}/{len(train_dataset)}")
        print(f"  Test with hints: {test_with_hints}/{len(test_dataset)}")
