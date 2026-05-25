"""
Experiment orchestration for running all model combinations.
"""
import asyncio
import logging
import itertools
from pathlib import Path
from typing import List, Optional

from ..llm.wrapper import OpenAIWrapper
from .evaluator import test_skeptic_variants_async
from ..analysis.hunk import Hunk

logger = logging.getLogger(__name__)


async def run_all_model_combinations(
    hunks: List[Hunk],
    selected_models: List[str],
    dataset_name: str,
    base_output_path: str = "results",
    n_debates: int = 1,
    sample_size: int = 100,
    per_model_concurrency: int = 4,
    seed: int = 42,
    use_rag: bool = False,
    max_context_lines: int = 400,
    llm_client: Optional[OpenAIWrapper] = None,
) -> None:
    """
    Run test_skeptic_variants_async for all combinations of (Analyzer, Proposer, Skeptic, Judge).
    
    Generates three types of combinations:
    1. All roles use the same model (4 combinations)
    2. Skeptic = Proposer, Analyzer != Judge (24 combinations)
    3. All roles different (24 combinations)
    
    Total: 52 combinations per debate round
    
    Folder layout:
        base_output_path/
            dataset_name/
                analyzer/
                    proposer/
                        skeptic/
                            judge/
                                debate_{n}/
                                    skeptics_results.csv
    
    Args:
        hunks: List of Hunk objects to evaluate
        selected_models: List of model names to test
        dataset_name: Name of dataset being tested
        base_output_path: Base path for results
        n_debates: Number of debate rounds to run
        sample_size: Sample size per model combination
        per_model_concurrency: Concurrent tasks per model
        seed: Random seed
        use_rag: Whether to use RAG
        max_context_lines: Maximum context lines
        llm_client: OpenAIWrapper instance (created if None)
    """
    if llm_client is None:
        llm_client = OpenAIWrapper()
    
    base_path = Path(base_output_path) / dataset_name
    base_path.mkdir(parents=True, exist_ok=True)

    models = list(selected_models)
    combinations = []
    
    # Case 1: All roles use the same model (4)
    for m in models:
        combinations.append((m, m, m, m))
    
    # Case 2: Skeptic = Proposer, Analyzer != Judge (24)
    for sp in models:
        remaining = [m for m in models if m != sp]
        for analyzer, judge in itertools.permutations(remaining, 2):
            combinations.append((analyzer, sp, sp, judge))
    
    # Case 3: All roles different (24)
    for analyzer, proposer, skeptic, judge in itertools.permutations(models, 4):
        combinations.append((analyzer, proposer, skeptic, judge))
    
    logger.info(f"Total restricted combinations: {len(combinations)}")
    
    count = 1
    for analyzer_model, proposer_model, skeptic_model, judge_model in combinations:
        for debate_round in range(1, n_debates + 1):
            logger.info(f"Combination {count}/{len(combinations)} with {n_debates} debate rounds")
            
            run_dir = (
                base_path
                / analyzer_model
                / proposer_model
                / skeptic_model
                / judge_model
                / f"debate_{debate_round}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            save_path = run_dir / "skeptics_results.csv"

            logger.info(
                f"Running:\n"
                f"  Analyzer={analyzer_model}\n"
                f"  Proposer={proposer_model}\n"
                f"  Skeptic={skeptic_model}\n"
                f"  Judge={judge_model}\n"
                f"  Debate round={debate_round}\n"
                f"  Save path={save_path}\n"
            )

            try:
                await test_skeptic_variants_async(
                    hunks=hunks,
                    skeptic_models=[skeptic_model],
                    llm_client=llm_client,
                    analyzer_model=analyzer_model,
                    proposer_model=proposer_model,
                    judge_model=judge_model,
                    debate_rounds=debate_round,
                    use_rag=use_rag,
                    max_context_lines=max_context_lines,
                    sample_size=sample_size,
                    seed=seed,
                    save_path=str(save_path),
                    per_model_concurrency=per_model_concurrency,
                )
            except Exception as e:
                logger.error(f"Error in combination {count}: {e}")
                continue

            count += 1

    logger.info("All model combinations completed")


async def run_debate_complexity_experiment(
    hunks: List[Hunk],
    analyzer_model: str,
    proposer_model: str,
    skeptic_model: str,
    judge_model: str,
    dataset_name: str,
    base_output_path: str = "results",
    debate_rounds_list: Optional[List[int]] = None,
    sample_size: int = 100,
    per_model_concurrency: int = 4,
    seed: int = 42,
    use_rag: bool = False,
    max_context_lines: int = 400,
    llm_client: Optional[OpenAIWrapper] = None,
) -> None:
    """
    Run experiments with varying number of debate rounds to measure complexity.
    
    Args:
        hunks: List of Hunk objects
        analyzer_model: Analyzer model to use
        proposer_model: Proposer model to use
        skeptic_model: Skeptic model to use
        judge_model: Judge model to use
        dataset_name: Dataset name
        base_output_path: Results directory
        debate_rounds_list: List of debate round counts (default: [1, 2, 3])
        sample_size: Sample size per experiment
        per_model_concurrency: Concurrent tasks
        seed: Random seed
        use_rag: Whether to use RAG
        max_context_lines: Maximum context lines
        llm_client: OpenAIWrapper instance
    """
    if llm_client is None:
        llm_client = OpenAIWrapper()
    
    if debate_rounds_list is None:
        debate_rounds_list = [1, 2, 3]
    
    for n_debate in debate_rounds_list:
        logger.info(f"Running debate complexity experiment with {n_debate} rounds")
        
        run_dir = Path(base_output_path) / dataset_name / f"n_debates_experiments_v{n_debate}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = run_dir / f"n_debates{n_debate}.csv"
        
        try:
            await test_skeptic_variants_async(
                hunks=hunks,
                skeptic_models=[skeptic_model],
                llm_client=llm_client,
                analyzer_model=analyzer_model,
                proposer_model=proposer_model,
                judge_model=judge_model,
                debate_rounds=n_debate,
                use_rag=use_rag,
                max_context_lines=max_context_lines,
                sample_size=sample_size,
                seed=seed,
                save_path=str(save_path),
                per_model_concurrency=per_model_concurrency,
            )
        except Exception as e:
            logger.error(f"Error in debate complexity experiment ({n_debate} rounds): {e}")


async def run_context_complexity_experiment(
    hunks: List[Hunk],
    analyzer_model: str,
    proposer_model: str,
    skeptic_model: str,
    judge_model: str,
    dataset_name: str,
    base_output_path: str = "results",
    max_lines_values: Optional[List[int]] = None,
    num_runs: int = 3,
    sample_size: int = 100,
    per_model_concurrency: int = 4,
    seed: int = 42,
    use_rag: bool = False,
    debate_rounds: int = 1,
    llm_client: Optional[OpenAIWrapper] = None,
) -> None:
    """
    Run experiments with varying max_context_lines to measure context complexity.
    
    Args:
        hunks: List of Hunk objects
        analyzer_model: Analyzer model to use
        proposer_model: Proposer model to use
        skeptic_model: Skeptic model to use
        judge_model: Judge model to use
        dataset_name: Dataset name
        base_output_path: Results directory
        max_lines_values: List of max_context_lines values (default: [0, 100, 200, 400, 600, 800, 1000])
        num_runs: Number of independent runs per value
        sample_size: Sample size per experiment
        per_model_concurrency: Concurrent tasks
        seed: Random seed
        use_rag: Whether to use RAG
        debate_rounds: Number of debate rounds
        llm_client: OpenAIWrapper instance
    """
    if llm_client is None:
        llm_client = OpenAIWrapper()
    
    if max_lines_values is None:
        max_lines_values = [0, 100, 200, 400, 600, 800, 1000]
    
    for run_num in range(1, num_runs + 1):
        for max_lines in max_lines_values:
            logger.info(f"Running context complexity experiment: max_lines={max_lines}, run={run_num}")
            
            run_dir = Path(base_output_path) / f"maxlines_experiments_v{run_num}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = run_dir / f"m{max_lines}.csv"
            
            try:
                await test_skeptic_variants_async(
                    hunks=hunks,
                    skeptic_models=[skeptic_model],
                    llm_client=llm_client,
                    analyzer_model=analyzer_model,
                    proposer_model=proposer_model,
                    judge_model=judge_model,
                    debate_rounds=debate_rounds,
                    use_rag=use_rag,
                    max_context_lines=max_lines,
                    sample_size=sample_size,
                    seed=seed,
                    save_path=str(save_path),
                    per_model_concurrency=per_model_concurrency,
                )
            except Exception as e:
                logger.error(f"Error in context complexity experiment (max_lines={max_lines}): {e}")
