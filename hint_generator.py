"""
Progressive Hint Generator for RL Training Data Enrichment

Generates difficulty estimates and 5-level progressive hints for each problem
to enable POPE-style guided exploration during RL training.

Based on research:
- POPE (Prefix-Guided Exploration) from CMU ML Blog
- Hint-before-Solving Prompting (HSP)
- Socratic scaffolding from Intelligent Tutoring Systems

Uses same async patterns as StructuredJudge for high-throughput batch processing.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
import asyncio
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

###############################################################
import os
if "KUBERNETES_SERVICE_HOST" in os.environ and os.getenv("KUBERNETES_SERVICE_HOST") != "":
    if os.getenv("NO_PROXY"):
        os.environ["NO_PROXY"] += ",.svc.cluster.local"
    if os.getenv("no_proxy"):
        os.environ["no_proxy"] += ",.svc.cluster.local"
###############################################################


# =============================================================================
# Schema Definitions
# =============================================================================

class ProgressiveHints(BaseModel):
    """Pre-computed difficulty and hints for POPE-style RL training."""
    estimated_difficulty: int = Field(
        ge=1, le=10,
        description="1=trivial, 10=competition-level"
    )
    hints: List[str] = Field(
        min_length=5,
        max_length=5,
        description="5 progressive hints: orientation → approach → insight → structure → near-complete"
    )


# =============================================================================
# System Prompt
# =============================================================================

HINT_GENERATION_SYSTEM_PROMPT = '''You are an expert problem analyst creating training data for AI reasoning models.

Your task: Given a problem and its solution, generate:
1. A difficulty rating (1-10)
2. Exactly 5 progressive hints following the "Socratic scaffolding" pattern

## Difficulty Scale
- 1-2: Single-step, direct application of one concept
- 3-4: Multi-step but straightforward, standard techniques
- 5-6: Requires insight or combining 2-3 concepts
- 7-8: Non-obvious approach, multiple insights needed
- 9-10: Competition-level, requires creative leaps or deep expertise

## Hint Progression (Critical)
Each hint must be STRICTLY more revealing than the previous:

- **Hint 1 (Orientation)**: Identify the problem type or relevant domain. No method revealed.
  → "This is a dynamic programming problem" or "Consider the properties of prime numbers"

- **Hint 2 (Approach)**: Point toward the general technique without specifics.
  → "Think about how to break this into overlapping subproblems"

- **Hint 3 (Key Insight)**: Reveal the critical insight or "aha moment" needed.
  → "The optimal substructure is: dp[i] = max(dp[i-1], dp[i-2] + val[i])"

- **Hint 4 (Structured Solution)**: Provide the solution skeleton/outline.
  → "1) Initialize base cases dp[0]=0, dp[1]=val[1]. 2) Iterate i=2 to n..."

- **Hint 5 (Near-Complete)**: Give almost everything except final execution.
  → "Apply the recurrence dp[i] = max(dp[i-1], dp[i-2]+val[i]) for i=2..n. The answer is dp[n]. For this input: dp[2]=max(1, 0+2)=2, dp[3]=max(2, 1+3)=4..."

## Rules
- Hints must form a strict monotonic progression in information revealed
- Hint 1 alone should NOT enable solving (even for easy problems)
- Hint 5 should make the problem trivial (only mechanical steps remain)
- For code: hints should guide algorithm design, not give code directly
- Be concise. Each hint should be 1-3 sentences max.

Respond in JSON format only.'''


def create_hint_generation_prompt(problem: str, solution: str) -> str:
    """Create the user prompt for hint generation."""
    return f'''## Problem
{problem}

## Reference Solution
{solution}

Generate difficulty and 5 progressive hints.'''


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class HintGenerationResult:
    """Container for hint generation result with metadata."""
    index: int
    hints: Optional[ProgressiveHints]
    elapsed_ms: float
    retries: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.hints is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        if self.hints is None:
            return {
                "index": self.index,
                "success": False,
                "error": self.error,
            }
        return {
            "index": self.index,
            "success": True,
            "estimated_difficulty": self.hints.estimated_difficulty,
            "hints": self.hints.hints,
        }


# =============================================================================
# HintGenerator Implementation
# =============================================================================

class HintGenerator:
    """
    High-throughput hint generator for RL training data enrichment.

    Generates difficulty estimates and progressive hints for each problem
    to enable POPE-style guided exploration during training.
    """

    DEFAULT_BATCH_TIMEOUT = 300.0  # 5 minutes for hint generation (more complex than judging)

    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8000/v1",
        api_key: str = "dummy",
        model: Optional[str] = None,
        max_concurrent: int = 64,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        batch_timeout: float = DEFAULT_BATCH_TIMEOUT,
    ):
        """
        Initialize the hint generator.

        Args:
            base_url: OpenAI-compatible API endpoint
            api_key: API key (can be dummy for local vLLM)
            model: Model name (auto-detected if None)
            max_concurrent: Maximum concurrent requests
            timeout: Timeout per request in seconds
            max_retries: Maximum retry attempts
            retry_base_delay: Base delay for exponential backoff
            batch_timeout: Global timeout for entire batch
        """
        self.base_url = base_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.batch_timeout = batch_timeout

        # Sync client for model detection and single generations
        self._sync_client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model or self._detect_model()

        # Async client
        self._async_client: Optional[AsyncOpenAI] = None
        self._async_client_owned = False

        # Statistics
        self._total_generations = 0
        self._total_failures = 0
        self._total_retries = 0
        self._total_timeouts = 0

        print(f"HintGenerator initialized")
        print(f"  Model: {self.model}")
        print(f"  Max concurrent: {self.max_concurrent}")
        print(f"  Per-request timeout: {self.timeout}s")
        print(f"  Batch timeout: {self.batch_timeout}s")
        print(f"  Max retries: {self.max_retries}")

    def _detect_model(self) -> str:
        """Auto-detect model from API."""
        try:
            models = self._sync_client.models.list()
            return models.data[0].id
        except Exception as e:
            print(f"Warning: Could not detect model: {e}")
            return "unknown"

    def _get_async_client(self) -> AsyncOpenAI:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
            self._async_client_owned = True
        return self._async_client

    # =========================================================================
    # Core Generation Logic
    # =========================================================================

    async def _generate_single_async(
        self,
        problem: str,
        solution: str,
        index: int,
        semaphore: asyncio.Semaphore,
        temperature: float = 0.7,
    ) -> HintGenerationResult:
        """Generate hints for a single problem with retry logic."""
        client = self._get_async_client()
        user_message = create_hint_generation_prompt(problem, solution)

        start_time = time.perf_counter()
        last_error: Optional[str] = None
        retries_used = 0

        for attempt in range(self.max_retries + 1):
            try:
                async with semaphore:
                    completion = await asyncio.wait_for(
                        client.beta.chat.completions.parse(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": HINT_GENERATION_SYSTEM_PROMPT},
                                {"role": "user", "content": user_message},
                            ],
                            response_format=ProgressiveHints,
                            temperature=temperature,
                        ),
                        timeout=self.timeout,
                    )

                    parsed = completion.choices[0].message.parsed
                    if parsed is None:
                        raise ValueError("Failed to parse structured output")

                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    return HintGenerationResult(
                        index=index,
                        hints=parsed,
                        elapsed_ms=elapsed_ms,
                        retries=retries_used,
                    )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                retries_used += 1

            except asyncio.CancelledError:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return HintGenerationResult(
                    index=index,
                    hints=None,
                    elapsed_ms=elapsed_ms,
                    retries=retries_used,
                    error="Cancelled due to batch timeout",
                )

            except Exception as e:
                last_error = str(e)
                retries_used += 1

            # Exponential backoff before retry
            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._total_failures += 1
        self._total_retries += retries_used

        return HintGenerationResult(
            index=index,
            hints=None,
            elapsed_ms=elapsed_ms,
            retries=retries_used,
            error=last_error,
        )

    # =========================================================================
    # Public API: Batch Generation
    # =========================================================================

    async def generate_batch_async(
        self,
        problems: List[str],
        solutions: List[str],
        temperature: float = 0.7,
        show_progress: bool = True,
        desc: str = "Generating hints",
        batch_timeout: Optional[float] = None,
    ) -> List[HintGenerationResult]:
        """
        Generate hints for a batch of problem-solution pairs asynchronously.

        Args:
            problems: List of problem statements
            solutions: List of reference solutions
            temperature: Sampling temperature (higher for more varied hints)
            show_progress: Show progress bar
            desc: Progress bar description
            batch_timeout: Override default batch timeout

        Returns:
            List of HintGenerationResult in original order
        """
        if len(problems) != len(solutions):
            raise ValueError(f"Length mismatch: {len(problems)} problems vs {len(solutions)} solutions")

        if len(problems) == 0:
            return []

        effective_timeout = batch_timeout if batch_timeout is not None else self.batch_timeout
        batch_size = len(problems)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        results_dict: Dict[int, HintGenerationResult] = {}
        completed_count = 0

        # Create tasks
        tasks = []
        for i, (problem, solution) in enumerate(zip(problems, solutions)):
            task = asyncio.create_task(
                self._generate_single_async(problem, solution, i, semaphore, temperature),
                name=f"hint_{i}"
            )
            tasks.append(task)

        pbar = tqdm(
            total=batch_size,
            desc=desc,
            unit="prob",
            disable=not show_progress,
            smoothing=0.1,
        )

        start_time = time.perf_counter()
        timed_out = False

        try:
            pending = set(tasks)

            while pending:
                elapsed = time.perf_counter() - start_time
                remaining_timeout = max(0.1, effective_timeout - elapsed)

                if elapsed >= effective_timeout:
                    timed_out = True
                    break

                done, pending = await asyncio.wait(
                    pending,
                    timeout=remaining_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    try:
                        result = task.result()
                        results_dict[result.index] = result
                        completed_count += 1
                        pbar.update(1)
                    except Exception as e:
                        task_name = task.get_name()
                        idx = int(task_name.split("_")[1]) if "_" in task_name else -1
                        if idx >= 0:
                            results_dict[idx] = HintGenerationResult(
                                index=idx,
                                hints=None,
                                elapsed_ms=0,
                                retries=0,
                                error=f"Task error: {e}",
                            )
                            completed_count += 1
                            pbar.update(1)

                if time.perf_counter() - start_time >= effective_timeout:
                    timed_out = True
                    break

        finally:
            pbar.close()

        # Handle timeout
        if timed_out or pending:
            self._total_timeouts += len(pending) if pending else 0

            for task in pending:
                task.cancel()

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            timeout_msg = f"Batch timeout after {effective_timeout}s"
            for i in range(batch_size):
                if i not in results_dict:
                    results_dict[i] = HintGenerationResult(
                        index=i,
                        hints=None,
                        elapsed_ms=0,
                        retries=0,
                        error=timeout_msg,
                    )

            elapsed_total = time.perf_counter() - start_time
            print(f"\nWarning: Batch timed out after {elapsed_total:.1f}s. "
                  f"Completed {completed_count}/{batch_size} generations.")

        results = [results_dict[i] for i in range(batch_size)]
        self._total_generations += len(results)

        return results

    def generate_batch(
        self,
        problems: List[str],
        solutions: List[str],
        temperature: float = 0.7,
        show_progress: bool = True,
        desc: str = "Generating hints",
        batch_timeout: Optional[float] = None,
    ) -> List[HintGenerationResult]:
        """Synchronous wrapper for batch generation."""
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - use thread pool
            return self._generate_batch_threaded(
                problems, solutions, temperature, show_progress, desc, batch_timeout
            )
        except RuntimeError:
            return asyncio.run(
                self.generate_batch_async(
                    problems, solutions, temperature, show_progress, desc, batch_timeout
                )
            )

    def _generate_batch_threaded(
        self,
        problems: List[str],
        solutions: List[str],
        temperature: float,
        show_progress: bool,
        desc: str,
        batch_timeout: Optional[float] = None,
    ) -> List[HintGenerationResult]:
        """Thread-based batch generation with global timeout."""
        effective_timeout = batch_timeout if batch_timeout is not None else self.batch_timeout
        batch_size = len(problems)
        results_dict: Dict[int, HintGenerationResult] = {}

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            def generate_single(idx: int) -> HintGenerationResult:
                start = time.perf_counter()
                user_message = create_hint_generation_prompt(problems[idx], solutions[idx])

                for attempt in range(self.max_retries + 1):
                    try:
                        completion = self._sync_client.beta.chat.completions.parse(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": HINT_GENERATION_SYSTEM_PROMPT},
                                {"role": "user", "content": user_message},
                            ],
                            response_format=ProgressiveHints,
                            temperature=temperature,
                        )

                        parsed = completion.choices[0].message.parsed
                        if parsed is None:
                            raise ValueError("Parse failed")

                        elapsed_ms = (time.perf_counter() - start) * 1000
                        return HintGenerationResult(
                            index=idx,
                            hints=parsed,
                            elapsed_ms=elapsed_ms,
                            retries=attempt,
                        )

                    except Exception as e:
                        if attempt < self.max_retries:
                            time.sleep(self.retry_base_delay * (2 ** attempt))
                        else:
                            elapsed_ms = (time.perf_counter() - start) * 1000
                            return HintGenerationResult(
                                index=idx,
                                hints=None,
                                elapsed_ms=elapsed_ms,
                                retries=attempt,
                                error=str(e),
                            )

                return HintGenerationResult(idx, None, 0, 0, "Unknown error")

            futures = {pool.submit(generate_single, i): i for i in range(batch_size)}

            pbar = tqdm(
                total=batch_size,
                desc=desc,
                disable=not show_progress,
                unit="prob",
                smoothing=0.1,
            )

            start_time = time.perf_counter()
            completed_count = 0

            try:
                for future in as_completed(futures, timeout=effective_timeout):
                    try:
                        result = future.result(timeout=1.0)
                        results_dict[result.index] = result
                        completed_count += 1
                        pbar.update(1)
                    except Exception as e:
                        idx = futures[future]
                        results_dict[idx] = HintGenerationResult(
                            index=idx,
                            hints=None,
                            elapsed_ms=0,
                            retries=0,
                            error=str(e),
                        )
                        completed_count += 1
                        pbar.update(1)

            except FuturesTimeoutError:
                elapsed_total = time.perf_counter() - start_time
                print(f"\nWarning: Batch timed out after {elapsed_total:.1f}s. "
                      f"Completed {completed_count}/{batch_size} generations.")

            finally:
                pbar.close()

        # Fill missing results
        for i in range(batch_size):
            if i not in results_dict:
                results_dict[i] = HintGenerationResult(
                    index=i,
                    hints=None,
                    elapsed_ms=0,
                    retries=0,
                    error=f"Batch timeout after {effective_timeout}s",
                )

        self._total_generations += batch_size
        self._total_timeouts += batch_size - completed_count

        return [results_dict[i] for i in range(batch_size)]

    # =========================================================================
    # Public API: Single Generation
    # =========================================================================

    def generate(
        self,
        problem: str,
        solution: str,
        temperature: float = 0.7,
    ) -> Optional[ProgressiveHints]:
        """Generate hints for a single problem synchronously."""
        user_message = create_hint_generation_prompt(problem, solution)

        for attempt in range(self.max_retries + 1):
            try:
                completion = self._sync_client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": HINT_GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=ProgressiveHints,
                    temperature=temperature,
                )

                return completion.choices[0].message.parsed

            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** attempt))
                else:
                    print(f"Generation failed after {self.max_retries + 1} attempts: {e}")
                    return None

        return None

    # =========================================================================
    # Dataset Enrichment
    # =========================================================================

    def enrich_dataset(
        self,
        problems: List[str],
        solutions: List[str],
        temperature: float = 0.7,
        show_progress: bool = True,
        batch_timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enrich a dataset with difficulty and hints.

        Returns list of dicts with:
        - problem: original problem
        - solution: original solution
        - estimated_difficulty: 1-10 score
        - hints: list of 5 progressive hints
        - success: whether generation succeeded

        Args:
            problems: List of problem statements
            solutions: List of reference solutions
            temperature: Sampling temperature
            show_progress: Show progress bar
            batch_timeout: Override default batch timeout

        Returns:
            List of enriched problem dictionaries
        """
        results = self.generate_batch(
            problems=problems,
            solutions=solutions,
            temperature=temperature,
            show_progress=show_progress,
            desc="[HintGenerator] Enriching dataset",
            batch_timeout=batch_timeout,
        )

        enriched = []
        for i, result in enumerate(results):
            entry = {
                "problem": problems[i],
                "solution": solutions[i],
                "success": result.success,
            }
            if result.success:
                entry["estimated_difficulty"] = result.hints.estimated_difficulty
                entry["hints"] = result.hints.hints
            else:
                entry["error"] = result.error
                entry["estimated_difficulty"] = None
                entry["hints"] = None
            enriched.append(entry)

        return enriched

    # =========================================================================
    # Statistics and Cleanup
    # =========================================================================

    def get_statistics(self) -> dict:
        """Get generation statistics."""
        return {
            "total_generations": self._total_generations,
            "total_failures": self._total_failures,
            "total_retries": self._total_retries,
            "total_timeouts": self._total_timeouts,
            "failure_rate": self._total_failures / max(1, self._total_generations),
            "timeout_rate": self._total_timeouts / max(1, self._total_generations),
        }

    def reset_statistics(self):
        """Reset generation statistics."""
        self._total_generations = 0
        self._total_failures = 0
        self._total_retries = 0
        self._total_timeouts = 0

    async def close_async(self):
        """Close async client."""
        if self._async_client is not None and self._async_client_owned:
            await self._async_client.close()
        self._async_client = None

    def close(self):
        """Close all resources."""
        if self._sync_client is not None:
            self._sync_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()
        self.close()
        return False


# =============================================================================
# Factory Function
# =============================================================================

def create_hint_generator(
    base_url: str = "http://0.0.0.0:8000/v1",
    api_key: str = "dummy",
    model: Optional[str] = None,
    max_concurrent: int = 64,
    timeout: float = 60.0,
    batch_timeout: float = 300.0,
) -> HintGenerator:
    """
    Create a hint generator for dataset enrichment.

    Usage:
        generator = create_hint_generator()
        enriched = generator.enrich_dataset(problems, solutions)
    """
    return HintGenerator(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_concurrent=max_concurrent,
        timeout=timeout,
        max_retries=2,
        retry_base_delay=0.5,
        batch_timeout=batch_timeout,
    )


# =============================================================================
# Parquet Dataset Processing
# =============================================================================

def extract_problem_from_row(row: Dict[str, Any]) -> str:
    """
    Extract problem text from a parquet row.

    Handles the Dolci-Think-RL schema:
    - prompt: [{"role": "user", "content": "..."}]
    """
    prompt = row.get("prompt")
    if prompt is None:
        return ""

    # Handle list of message dicts
    if isinstance(prompt, list) and len(prompt) > 0:
        first_msg = prompt[0]
        if isinstance(first_msg, dict):
            return first_msg.get("content", "")

    # Fallback: treat as string
    if isinstance(prompt, str):
        return prompt

    return str(prompt)


def extract_solution_from_row(row: Dict[str, Any]) -> str:
    """
    Extract solution/ground_truth from a parquet row.

    Handles the Dolci-Think-RL schema:
    - reward_model: {"style": "...", "ground_truth": "JSON string"}
    """
    reward_model = row.get("reward_model")
    if reward_model is None:
        return ""

    # Handle dict with ground_truth field
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth", "")
        # ground_truth is already a JSON string, return as-is for the LLM
        if isinstance(ground_truth, str):
            return ground_truth
        return json.dumps(ground_truth)

    return str(reward_model)


def enrich_parquet(
    input_path: str,
    output_path: str,
    generator: HintGenerator,
    temperature: float = 0.7,
    batch_size: int = 1000,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Enrich a parquet file with difficulty estimates and progressive hints.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        generator: HintGenerator instance
        temperature: Sampling temperature
        batch_size: Process in batches of this size
        show_progress: Show progress bars

    Returns:
        Statistics dict with success/failure counts
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows")

    # Extract problems and solutions
    print("Extracting problems and solutions...")
    problems = []
    solutions = []

    for idx in tqdm(range(total_rows), desc="Parsing rows", disable=not show_progress):
        row = df.iloc[idx].to_dict()
        problems.append(extract_problem_from_row(row))
        solutions.append(extract_solution_from_row(row))

    # Process in batches to manage memory and allow checkpointing
    all_difficulties = []
    all_hints = []
    all_successes = []
    all_errors = []

    total_success = 0
    num_batches = (total_rows + batch_size - 1) // batch_size

    print(f"Processing {total_rows} rows in {num_batches} batches...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_rows)

        batch_problems = problems[start_idx:end_idx]
        batch_solutions = solutions[start_idx:end_idx]

        results = generator.generate_batch(
            problems=batch_problems,
            solutions=batch_solutions,
            temperature=temperature,
            show_progress=show_progress,
            desc=f"[Batch {batch_idx + 1}/{num_batches}]",
        )

        for result in results:
            if result.success:
                all_difficulties.append(result.hints.estimated_difficulty)
                all_hints.append(json.dumps(result.hints.hints))
                all_successes.append(True)
                all_errors.append(None)
                total_success += 1
            else:
                all_difficulties.append(None)
                all_hints.append(None)
                all_successes.append(False)
                all_errors.append(result.error)

    # Add new columns to dataframe
    print("Adding enriched columns...")
    df["estimated_difficulty"] = all_difficulties
    df["hints"] = all_hints  # JSON string of list
    df["hint_generation_success"] = all_successes
    df["hint_generation_error"] = all_errors

    # Save to parquet
    print(f"Writing {output_path}...")
    df.to_parquet(output_path, index=False)

    stats = {
        "total_rows": total_rows,
        "successful": total_success,
        "failed": total_rows - total_success,
        "success_rate": total_success / max(1, total_rows),
        **generator.get_statistics(),
    }

    print(f"Done! {total_success}/{total_rows} successful generations ({stats['success_rate']:.1%})")

    return stats


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    """CLI entrypoint for processing parquet/JSONL files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate difficulty estimates and progressive hints for RL training data"
    )
    parser.add_argument(
        "input_file",
        help="Input parquet or JSONL file"
    )
    parser.add_argument(
        "output_file",
        help="Output parquet or JSONL file (same format as input)"
    )
    parser.add_argument(
        "--base-url",
        default="http://0.0.0.0:8000/v1",
        help="OpenAI-compatible API endpoint"
    )
    parser.add_argument(
        "--api-key",
        default="dummy",
        help="API key"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=64,
        help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=300.0,
        help="Batch timeout in seconds"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Process parquet in batches of this size"
    )
    parser.add_argument(
        "--format",
        choices=["auto", "parquet", "jsonl"],
        default="auto",
        help="Input/output format (auto-detected from extension by default)"
    )

    args = parser.parse_args()

    # Detect format
    fmt = args.format
    if fmt == "auto":
        if args.input_file.endswith(".parquet"):
            fmt = "parquet"
        else:
            fmt = "jsonl"

    # Create generator
    generator = create_hint_generator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
        batch_timeout=args.batch_timeout,
    )

    if fmt == "parquet":
        # Process parquet file
        stats = enrich_parquet(
            input_path=args.input_file,
            output_path=args.output_file,
            generator=generator,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
        print(f"Statistics: {stats}")

    else:
        # Process JSONL file (legacy support)
        print(f"Loading {args.input_file}...")
        data = []
        with open(args.input_file, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        print(f"Loaded {len(data)} examples")

        # Extract using same logic as parquet
        problems = [extract_problem_from_row(d) for d in data]
        solutions = [extract_solution_from_row(d) for d in data]

        results = generator.generate_batch(
            problems=problems,
            solutions=solutions,
            temperature=args.temperature,
            desc="[HintGenerator] Processing JSONL",
        )

        # Merge with original data and write output
        print(f"Writing {args.output_file}...")
        success_count = 0
        with open(args.output_file, "w") as f:
            for orig, result in zip(data, results):
                output = {**orig}
                if result.success:
                    output["estimated_difficulty"] = result.hints.estimated_difficulty
                    output["hints"] = result.hints.hints
                    output["hint_generation_success"] = True
                    success_count += 1
                else:
                    output["estimated_difficulty"] = None
                    output["hints"] = None
                    output["hint_generation_success"] = False
                    output["hint_generation_error"] = result.error
                f.write(json.dumps(output) + "\n")

        print(f"Done! {success_count}/{len(data)} successful generations")
        print(f"Statistics: {generator.get_statistics()}")


if __name__ == "__main__":
    main()
