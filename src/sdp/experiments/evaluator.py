"""
Asynchronous experiment evaluation for parallel model testing.
"""
import asyncio
import logging
import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from ..llm.wrapper import OpenAIWrapper
from ..llm.experts import ExpertDebateSystem
from ..analysis.verdict_parser import parse_judge_verdict, parse_confidence
from ..analysis.hunk import Hunk

logger = logging.getLogger(__name__)


async def async_run_model_single(
    skeptic_name: str,
    work_items: List[Tuple],
    llm_client: OpenAIWrapper,
    analyzer_model: str,
    proposer_model: str,
    judge_model: str,
    debate_rounds: int,
    use_rag: bool,
    max_context_lines: int,
    save_path: str,
    csv_lock: Optional[asyncio.Lock] = None,
    progress_callback: Optional[callable] = None,
    subset_stats_init: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    """
    Run evaluation for one skeptic model asynchronously.
    
    Args:
        skeptic_name: Name of skeptic model to use
        work_items: List of (file_path, hunk, subset_name) tuples
        llm_client: OpenAIWrapper instance
        analyzer_model: Analyzer model name
        proposer_model: Proposer model name
        judge_model: Judge model name
        debate_rounds: Number of debate rounds
        use_rag: Whether to use RAG (retrieval-augmented generation)
        max_context_lines: Maximum context lines to include
        save_path: Path to save CSV results
        progress_callback: Optional callback for progress updates
        subset_stats_init: Initial statistics dictionary
    
    Returns:
        Tuple of (skeptic_name, subset_stats)
    """
    from copy import deepcopy
    
    subset_stats = {k: v.copy() for k, v in (subset_stats_init or {}).items()}
    
    # Create debate system with all models
    debate_system = ExpertDebateSystem(
        llm_client=llm_client,
        analyzer_model=analyzer_model,
        proposer_model=proposer_model,
        skeptic_model=skeptic_name,
        judge_model=judge_model,
        max_rounds=debate_rounds,
    )
    
    for fp, hunk, subset_name in work_items:
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "analyzer_model": analyzer_model,
            "proposer_model": proposer_model,
            "skeptic_model": skeptic_name,
            "judge_model": judge_model,
            "subset": subset_name,
            "file_path": fp,
            "old_label": getattr(hunk, "old_label", None),
            "new_label": getattr(hunk, "label", None),
            "verdict_str": None,
            "verdict_int": None,
            "confidence": None,
            "correct": None,
            "unparsed": None,
            "row_runtime_sec": None,
            "judge_decision": None,
            "analyzer_response": None,
            "proposer_response": None,
            "skeptic_response": None,
            "error": None,
        }

        try:
            start_t = time.perf_counter()
            
            # Run debate
            prediction, confidence, history = await asyncio.to_thread(
                debate_system.run_debate,
                hunk.unified_diff or "",
                hunk.src1[:500],
                hunk.src2[:500],
                hunk.relevant_context or "",
                "BENIGN" if hunk.old_label == 0 else "DEFECTIVE",
            )
            
            end_t = time.perf_counter()
            row_runtime = end_t - start_t

            # Extract judge response from history
            judge_response = None
            for round_obj in reversed(history):
                if round_obj.judge_response:
                    judge_response = round_obj.judge_response
                    break

            analyzer_response = None
            proposer_response = None
            skeptic_response = None
            for round_obj in history:
                if analyzer_response is None and round_obj.analyzer_response:
                    analyzer_response = round_obj.analyzer_response
                if round_obj.proposer_response:
                    proposer_response = round_obj.proposer_response
                if round_obj.skeptic_response:
                    skeptic_response = round_obj.skeptic_response

            # Parse verdict
            verdict_str, verdict_int = parse_judge_verdict(judge_response or "")
            if verdict_int is None:
                correct = 0
                unparsed = True
            else:
                correct = int(verdict_int == hunk.label)
                unparsed = False

            row.update({
                "verdict_str": verdict_str,
                "verdict_int": verdict_int,
                "confidence": confidence,
                "correct": correct,
                "unparsed": unparsed,
                "judge_decision": judge_response,
                "analyzer_response": analyzer_response,
                "proposer_response": proposer_response,
                "skeptic_response": skeptic_response,
                "row_runtime_sec": round(row_runtime, 3),
            })

        except Exception as e:
            logger.error(f"Error processing {fp}: {e}")
            row["error"] = str(e)

        # Thread-safe CSV append
        try:
            lock = csv_lock or asyncio.Lock()
            async with lock:
                if os.path.exists(save_path):
                    existing_cols = pd.read_csv(save_path, nrows=0).columns.tolist()
                else:
                    existing_cols = list(row.keys())

                df_row = pd.DataFrame([row]).reindex(columns=existing_cols)
                await asyncio.to_thread(
                    lambda: df_row.to_csv(save_path, mode="a", header=False, index=False)
                )
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")

        # Update stats
        subset_stats.setdefault(subset_name, {"correct": 0, "total": 0, "unparsed": 0})
        subset_stats[subset_name]["total"] += 1
        if row["correct"]:
            subset_stats[subset_name]["correct"] += 1
        if row["unparsed"]:
            subset_stats[subset_name]["unparsed"] += 1

        if progress_callback:
            progress_callback(subset_name, subset_stats)

    return skeptic_name, subset_stats


async def test_skeptic_variants_async(
    hunks: List[Hunk],
    skeptic_models: List[str],
    llm_client: OpenAIWrapper,
    analyzer_model: str = "deepseek-v4-flash",
    proposer_model: str = "deepseek-v4-flash",
    judge_model: str = "deepseek-v4-flash",
    debate_rounds: int = 1,
    use_rag: bool = False,
    max_context_lines: int = 400,
    sample_size: int = 100,
    seed: int = 42,
    save_path: str = "skeptic_results.csv",
    per_model_concurrency: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run multiple Skeptic models concurrently with partial result recovery.
    
    Args:
        hunks: List of Hunk objects to evaluate
        skeptic_models: List of skeptic model names
        llm_client: OpenAIWrapper instance
        analyzer_model: Analyzer model name
        proposer_model: Proposer model name
        judge_model: Judge model name
        debate_rounds: Number of debate rounds
        use_rag: Whether to use RAG
        max_context_lines: Maximum context lines
        sample_size: Sample size per subset
        seed: Random seed
        save_path: Path to save results CSV
        per_model_concurrency: Concurrency limit per model
    
    Returns:
        Tuple of (results_df, summary_df)
    """
    import random
    
    random.seed(seed)
    logger.info(f"Starting multi-skeptic evaluation with {len(skeptic_models)} models")
    
    # Build subsets
    subsets = {
        "Defective_01": [h for h in hunks if h.old_label == 0 and h.label == 1],
        "Benign_00": [h for h in hunks if h.old_label == 0 and h.label == 0],
        "Benign_10": [h for h in hunks if h.old_label == 1 and h.label == 0],
        "Defective_11": [h for h in hunks if h.old_label == 1 and h.label == 1],
    }
    
    for k in subsets:
        subsets[k] = random.sample(subsets[k], min(len(subsets[k]), sample_size))

    all_hunks = [(h.file_path, h, subset) for subset, lst in subsets.items() for h in lst]
    random.shuffle(all_hunks)

    # Setup CSV
    columns = [
        "timestamp", "analyzer_model", "proposer_model", "skeptic_model", "judge_model",
        "subset", "file_path", "old_label", "new_label", "verdict_str", "verdict_int",
        "confidence", "correct", "unparsed", "row_runtime_sec", "judge_decision",
        "analyzer_response", "proposer_response", "skeptic_response", "error",
    ]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(save_path):
        df_existing = pd.read_csv(save_path)
        missing = [c for c in columns if c not in df_existing.columns]
        if missing:
            for c in missing:
                df_existing[c] = pd.NA
            df_existing = df_existing.reindex(columns=columns)
            df_existing.to_csv(save_path, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(save_path, index=False)

    processed_keys = set()
    if os.path.exists(save_path):
        df_existing = pd.read_csv(save_path)
        df_completed = df_existing[df_existing["verdict_int"].notna()].copy()
        processed_keys = set(
            zip(df_completed["skeptic_model"].astype(str), df_completed["file_path"].astype(str))
        )

    # Filter new work
    work_per_model = {
        model: [(fp, h, subset) for (fp, h, subset) in all_hunks if (model, str(fp)) not in processed_keys]
        for model in skeptic_models
    }

    logger.info(f"Total work items: {sum(len(v) for v in work_per_model.values())}")

    # Run with per-model concurrency
    csv_lock = asyncio.Lock()

    async def run_model_with_limit(model, tasks_for_model):
        sem = asyncio.Semaphore(per_model_concurrency)

        async def run_one(fp, h, subset):
            async with sem:
                return await async_run_model_single(
                    skeptic_name=model,
                    work_items=[(fp, h, subset)],
                    llm_client=llm_client,
                    analyzer_model=analyzer_model,
                    proposer_model=proposer_model,
                    judge_model=judge_model,
                    debate_rounds=debate_rounds,
                    use_rag=use_rag,
                    max_context_lines=max_context_lines,
                    save_path=save_path,
                    csv_lock=csv_lock,
                )

        return await asyncio.gather(*(run_one(fp, h, subset) for fp, h, subset in tasks_for_model))

    try:
        tasks = []
        for model in skeptic_models:
            if work_per_model[model]:
                coro = run_model_with_limit(model, work_per_model[model])
                tasks.append(coro)

        await asyncio.gather(*tasks)
        logger.info("All evaluations completed successfully")

    except asyncio.CancelledError:
        logger.warning("Evaluation cancelled")
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")

    # Load results and compute summary
    df_results = pd.read_csv(save_path)
    df_results = df_results.dropna(subset=["verdict_int"])

    summary = (
        df_results.groupby(["skeptic_model", "subset"])
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            unparsed=("unparsed", "sum"),
        )
        .reset_index()
    )
    summary["accuracy_pct"] = (summary["correct"] / summary["total"] * 100).round(1)

    return df_results, summary
