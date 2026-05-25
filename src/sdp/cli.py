"""
Command-line interface for running defect prediction experiments.
"""
import argparse
import logging
import sys
from typing import Optional

from .config import DATASET_NAME_RUN, llm7_all_models
from .data.loader import load_dataset_pair
from .analysis.diff import compute_diffs
from .llm.wrapper import OpenAIWrapper
from .llm.experts import ExpertDebateSystem


def setup_logger(level=logging.INFO):
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_defect_prediction(
    dataset_name: str = DATASET_NAME_RUN,
    old_version: Optional[str] = None,
    new_version: Optional[str] = None,
    max_samples: Optional[int] = None,
    model: str = "deepseek-v4-flash",
    debate_rounds: int = 3,
    prepare_only: bool = False,
) -> None:
    """
    Run defect prediction on a dataset using the expert debate system.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'camel', 'lucene')
        old_version: Optional explicit old version
        new_version: Optional explicit new version
        max_samples: Maximum number of samples to process (None = all)
        model: Model name to use for all experts
        debate_rounds: Number of debate rounds
        prepare_only: Skip API calls and only validate data/diff preparation
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        # Load dataset pair
        past_version, next_version = load_dataset_pair(
            dataset_name,
            old_version=old_version,
            new_version=new_version,
        )
        logger.info(
            "Loaded dataset pair %s -> %s with %d/%d rows",
            past_version.attrs.get("dataset_version"),
            next_version.attrs.get("dataset_version"),
            len(past_version),
            len(next_version),
        )

        debate_system = None
        if not prepare_only:
            llm_client = OpenAIWrapper()
            debate_system = ExpertDebateSystem(
                llm_client=llm_client,
                analyzer_model=model,
                proposer_model=model,
                skeptic_model=model,
                judge_model=model,
                max_rounds=debate_rounds,
            )
        
        # Process samples
        sample_count = 0
        exact_matches = 0
        changed_matches = 0
        for idx, row in past_version.iterrows():
            if max_samples and sample_count >= max_samples:
                break
            
            file_name = row.get("File")
            bug_label = row.get("Bug")
            src = row.get("SRC")
            
            if not all([file_name, src]):
                logger.warning(f"Skipping sample {idx}: missing File or SRC")
                continue
            
            # Find corresponding file in new version
            new_row = next_version[next_version["File"] == file_name]
            if new_row.empty:
                logger.debug(f"File {file_name} not found in new version")
                continue
            exact_matches += 1
            
            new_src = new_row.iloc[0].get("SRC")
            if not new_src:
                logger.warning(f"Skipping {file_name}: missing SRC in new version")
                continue
            
            # Compute diff
            try:
                diffs = compute_diffs(src, new_src)
                if not diffs or not any(len(v) > 0 for k, v in diffs.items() if k != "unified"):
                    logger.debug(f"No diffs found for {file_name}")
                    continue
                changed_matches += 1
            except Exception as e:
                logger.error(f"Error computing diff for {file_name}: {e}")
                continue

            if prepare_only:
                logger.info(
                    "Prepared %s (sample %d): old_label=%s new_label=%s diff_lines=%d",
                    file_name,
                    sample_count + 1,
                    int(bug_label),
                    int(new_row.iloc[0].get("Bug", 0)),
                    len(diffs.get("unified", [])),
                )
                sample_count += 1
                continue
            
            # Run debate
            logger.info(f"Processing {file_name} (sample {sample_count + 1})")
            try:
                prediction, confidence, history = debate_system.run_debate(
                    diff_text="\n".join(diffs.get("unified", [])),
                    src1_context=src[:500],  # Truncate for context
                    src2_context=new_src[:500],
                    context=new_src[:1000],
                    previous_status="BENIGN" if bug_label == 0 else "DEFECTIVE",
                )
                
                logger.info(f"  → Prediction: {prediction} (confidence: {confidence}%)")
                sample_count += 1
            except Exception as e:
                logger.error(f"Error running debate for {file_name}: {e}")
                continue
        
        logger.info(
            "Completed processing %d samples (matched=%d, changed=%d, prepare_only=%s)",
            sample_count,
            exact_matches,
            changed_matches,
            prepare_only,
        )
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run defect prediction experiments with expert debate system."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME_RUN,
        help=f"Dataset name (default: {DATASET_NAME_RUN})",
    )
    parser.add_argument(
        "--old-version",
        type=str,
        default=None,
        help="Optional old dataset version (for example: 2.9.0)",
    )
    parser.add_argument(
        "--new-version",
        type=str,
        default=None,
        help="Optional new dataset version (for example: 3.0.0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v4-flash",
        help="Model to use for all experts (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of debate rounds (default: 3)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate dataset loading and diff preparation without calling the API",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    # List models if requested
    if args.list_models:
        logger.info("Available models:")
        for model in llm7_all_models:
            print(f"  - {model}")
        sys.exit(0)
    
    # Run defect prediction
    run_defect_prediction(
        dataset_name=args.dataset,
        old_version=args.old_version,
        new_version=args.new_version,
        max_samples=args.max_samples,
        model=args.model,
        debate_rounds=args.rounds,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
