"""
Generate difficulty + progressive hints for POPE-style RL training.
Usage: python hint_generator.py train.parquet train_enriched.parquet --base-url http://0.0.0.0:8000/v1
"""

import asyncio
import json
import time
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

if "KUBERNETES_SERVICE_HOST" in os.environ:
    for var in ("NO_PROXY", "no_proxy"):
        if os.getenv(var):
            os.environ[var] += ",.svc.cluster.local"


class ProgressiveHints(BaseModel):
    estimated_difficulty: int = Field(ge=1, le=10)
    hints: List[str] = Field(min_length=5, max_length=5)


SYSTEM_PROMPT = '''Generate difficulty (1-10) and 5 progressive hints for this problem.

Difficulty: 1-2 trivial, 3-4 standard, 5-6 needs insight, 7-8 multiple insights, 9-10 competition-level.

Hints must be strictly progressive:
1. Orientation: Problem type only ("This is a DP problem")
2. Approach: General technique ("Use memoization")
3. Key Insight: The "aha" moment ("dp[i] = max(dp[i-1], dp[i-2] + val[i])")
4. Structure: Solution outline with steps
5. Near-Complete: Everything except final computation

Rules: Hint 1 must not enable solving. Hint 5 makes it trivial. 1-3 sentences each. JSON only.'''


@dataclass
class Result:
    index: int
    hints: Optional[ProgressiveHints]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.hints is not None


class HintGenerator:
    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8000/v1",
        api_key: str = "dummy",
        model: Optional[str] = None,
        max_concurrent: int = 64,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent

        self._sync = OpenAI(base_url=base_url, api_key=api_key)
        self._async: Optional[AsyncOpenAI] = None
        self.model = model or self._detect_model()

    def _detect_model(self) -> str:
        try:
            return self._sync.models.list().data[0].id
        except Exception:
            return "unknown"

    def _get_async(self) -> AsyncOpenAI:
        if not self._async:
            self._async = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._async

    def _make_prompt(self, problem: str, solution: str) -> str:
        return f"## Problem\n{problem}\n\n## Reference Solution\n{solution}"

    async def _generate_one(self, problem: str, solution: str, idx: int, sem: asyncio.Semaphore) -> Result:
        for attempt in range(self.max_retries + 1):
            try:
                async with sem:
                    resp = await asyncio.wait_for(
                        self._get_async().beta.chat.completions.parse(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": self._make_prompt(problem, solution)},
                            ],
                            response_format=ProgressiveHints,
                            temperature=0.7,
                        ),
                        timeout=self.timeout,
                    )
                    parsed = resp.choices[0].message.parsed
                    if parsed:
                        return Result(idx, parsed)
                    raise ValueError("Parse failed")
            except asyncio.CancelledError:
                return Result(idx, None, "cancelled")
            except Exception as e:
                if attempt == self.max_retries:
                    return Result(idx, None, str(e))
                await asyncio.sleep(0.5 * (2 ** attempt))
        return Result(idx, None, "exhausted retries")

    async def generate_batch_async(
        self, problems: List[str], solutions: List[str], show_progress: bool = True
    ) -> List[Result]:
        n = len(problems)
        sem = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            asyncio.create_task(self._generate_one(p, s, i, sem))
            for i, (p, s) in enumerate(zip(problems, solutions))
        ]

        results: Dict[int, Result] = {}
        pbar = tqdm(total=n, desc="Generating hints", disable=not show_progress)

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                r = t.result()
                results[r.index] = r
                pbar.update(1)
        pbar.close()

        return [results[i] for i in range(n)]

    def generate_batch(self, problems: List[str], solutions: List[str], **kw) -> List[Result]:
        try:
            asyncio.get_running_loop()
            return self._generate_batch_sync(problems, solutions, **kw)
        except RuntimeError:
            return asyncio.run(self.generate_batch_async(problems, solutions, **kw))

    def _generate_batch_sync(self, problems: List[str], solutions: List[str], show_progress: bool = True) -> List[Result]:
        n = len(problems)
        results: Dict[int, Result] = {}

        def worker(idx: int) -> Result:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = self._sync.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": self._make_prompt(problems[idx], solutions[idx])},
                        ],
                        response_format=ProgressiveHints,
                        temperature=0.7,
                    )
                    parsed = resp.choices[0].message.parsed
                    if parsed:
                        return Result(idx, parsed)
                except Exception as e:
                    if attempt == self.max_retries:
                        return Result(idx, None, str(e))
                    time.sleep(0.5 * (2 ** attempt))
            return Result(idx, None, "exhausted")

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(worker, i): i for i in range(n)}
            pbar = tqdm(total=n, desc="Generating hints", disable=not show_progress)
            for f in as_completed(futures):
                r = f.result()
                results[r.index] = r
                pbar.update(1)
            pbar.close()

        return [results[i] for i in range(n)]


def extract_problem(row: Dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        msg = prompt[0]
        if isinstance(msg, dict):
            return msg.get("content", "")
    return str(prompt) if prompt else ""


def extract_solution(row: Dict[str, Any]) -> str:
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth", "")
        return gt if isinstance(gt, str) else json.dumps(gt)
    return str(rm) if rm else ""


def enrich_parquet(input_path: str, output_path: str, generator: HintGenerator, batch_size: int = 1000):
    import pandas as pd

    df = pd.read_parquet(input_path)
    n = len(df)
    print(f"Loaded {n} rows from {input_path}")

    problems = [extract_problem(df.iloc[i].to_dict()) for i in range(n)]
    solutions = [extract_solution(df.iloc[i].to_dict()) for i in range(n)]

    all_diff, all_hints, all_ok, all_err = [], [], [], []
    success = 0

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        results = generator.generate_batch(
            problems[batch_start:batch_end],
            solutions[batch_start:batch_end],
        )
        for r in results:
            if r.success:
                all_diff.append(r.hints.estimated_difficulty)
                all_hints.append(json.dumps(r.hints.hints))
                all_ok.append(True)
                all_err.append(None)
                success += 1
            else:
                all_diff.append(None)
                all_hints.append(None)
                all_ok.append(False)
                all_err.append(r.error)

    df["estimated_difficulty"] = all_diff
    df["hints"] = all_hints
    df["hint_generation_success"] = all_ok
    df["hint_generation_error"] = all_err

    df.to_parquet(output_path, index=False)
    print(f"Done: {success}/{n} ({100*success/n:.1f}%) -> {output_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_file")
    p.add_argument("output_file")
    p.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--model", default=None)
    p.add_argument("--max-concurrent", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1000)
    args = p.parse_args()

    gen = HintGenerator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
    )

    if args.input_file.endswith(".parquet"):
        enrich_parquet(args.input_file, args.output_file, gen, args.batch_size)
    else:
        # JSONL
        with open(args.input_file) as f:
            data = [json.loads(line) for line in f if line.strip()]

        problems = [extract_problem(d) for d in data]
        solutions = [extract_solution(d) for d in data]
        results = gen.generate_batch(problems, solutions)

        with open(args.output_file, "w") as f:
            for orig, r in zip(data, results):
                orig["estimated_difficulty"] = r.hints.estimated_difficulty if r.success else None
                orig["hints"] = r.hints.hints if r.success else None
                orig["hint_generation_success"] = r.success
                if not r.success:
                    orig["hint_generation_error"] = r.error
                f.write(json.dumps(orig) + "\n")


if __name__ == "__main__":
    main()
