"""CLI for the CPDP change-aware project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .baselines.direct_transfer import DirectTransferBaseline, save_baseline_outputs
from .baselines.static_llm import StaticTargetLLMBaseline
from .baselines.source_change_static import SourceChangeToStaticBaseline
from .data.builders import (
    build_adjacent_change_pairs_from_directory,
    build_change_pairs_from_version_csv,
    build_static_file_samples_from_version_csv,
)
from .data.io import load_change_pairs, load_static_file_samples
from .debate.static_system import StaticTargetDebateSystem
from .pipeline import CrossProjectChangePredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-project, change-aware defect prediction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-pairs", help="Build change-pair JSONL from two version CSV files.")
    prepare.add_argument("--project", required=True)
    prepare.add_argument("--old-csv", required=True)
    prepare.add_argument("--new-csv", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--old-version")
    prepare.add_argument("--new-version")
    prepare.add_argument("--language", default="java")

    prepare_static = subparsers.add_parser(
        "prepare-static",
        help="Build static single-version JSONL from one File-level CSV.",
    )
    prepare_static.add_argument("--project", required=True)
    prepare_static.add_argument("--csv", required=True)
    prepare_static.add_argument("--output", required=True)
    prepare_static.add_argument("--version")
    prepare_static.add_argument("--language", default="java")

    prepare_bulk = subparsers.add_parser(
        "prepare-bulk",
        help="Batch-build adjacent change-pair JSONL files from a File-level dataset directory.",
    )
    prepare_bulk.add_argument("--dataset-dir", required=True)
    prepare_bulk.add_argument("--output-dir", required=True)
    prepare_bulk.add_argument("--projects", nargs="*")
    prepare_bulk.add_argument("--language", default="java")

    fit = subparsers.add_parser("fit", help="Fit a source-project retrieval index.")
    fit.add_argument("--sources", nargs="+", required=True)
    fit.add_argument("--artifact-path", required=True)

    predict = subparsers.add_parser("predict", help="Predict defect labels for target-project change pairs.")
    predict.add_argument("--artifact-path", required=True)
    predict.add_argument("--target", nargs="+", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--backend", choices=["heuristic", "llm"], default="heuristic")
    predict.add_argument("--model", default=None)
    predict.add_argument("--top-k", type=int, default=5)

    baseline = subparsers.add_parser(
        "baseline-direct",
        help="Run a direct source-to-target CPDP baseline with TF-IDF + logistic regression.",
    )
    baseline.add_argument("--sources", nargs="+", required=True)
    baseline.add_argument("--target", nargs="+", required=True)
    baseline.add_argument("--predictions-output", required=True)
    baseline.add_argument("--summary-output", required=True)
    baseline.add_argument("--source-name", required=True)
    baseline.add_argument("--target-name", required=True)

    baseline_static = subparsers.add_parser(
        "baseline-static",
        help="Train on source change pairs and predict a single-version target project.",
    )
    baseline_static.add_argument("--sources", nargs="+", required=True)
    baseline_static.add_argument("--target-static", nargs="+", required=True)
    baseline_static.add_argument("--predictions-output", required=True)
    baseline_static.add_argument("--summary-output", required=True)
    baseline_static.add_argument("--source-name", required=True)
    baseline_static.add_argument("--target-name", required=True)

    baseline_static_llm = subparsers.add_parser(
        "baseline-static-llm",
        help="Run an LLM baseline for static target-project prediction without TF-IDF.",
    )
    baseline_static_llm.add_argument("--sources", nargs="+", required=True)
    baseline_static_llm.add_argument("--target-static", nargs="+", required=True)
    baseline_static_llm.add_argument("--predictions-output", required=True)
    baseline_static_llm.add_argument("--summary-output", required=True)
    baseline_static_llm.add_argument("--source-name", required=True)
    baseline_static_llm.add_argument("--target-name", required=True)
    baseline_static_llm.add_argument("--model", default=None)
    baseline_static_llm.add_argument("--top-k", type=int, default=3)

    baseline_static_debate = subparsers.add_parser(
        "baseline-static-debate",
        help="Run a static-target multi-agent debate baseline without TF-IDF.",
    )
    baseline_static_debate.add_argument("--sources", nargs="+", required=True)
    baseline_static_debate.add_argument("--target-static", nargs="+", required=True)
    baseline_static_debate.add_argument("--predictions-output", required=True)
    baseline_static_debate.add_argument("--summary-output", required=True)
    baseline_static_debate.add_argument("--source-name", required=True)
    baseline_static_debate.add_argument("--target-name", required=True)
    baseline_static_debate.add_argument("--model", default=None)
    baseline_static_debate.add_argument("--top-k", type=int, default=3)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-pairs":
        pairs = build_change_pairs_from_version_csv(
            project=args.project,
            old_csv=args.old_csv,
            new_csv=args.new_csv,
            output_path=args.output,
            old_version=args.old_version,
            new_version=args.new_version,
            language=args.language,
        )
        print(f"Built {len(pairs)} change pairs at {Path(args.output).resolve()}")
        return

    if args.command == "prepare-static":
        samples = build_static_file_samples_from_version_csv(
            project=args.project,
            csv_path=args.csv,
            output_path=args.output,
            version=args.version,
            language=args.language,
        )
        print(f"Built {len(samples)} static file samples at {Path(args.output).resolve()}")
        return

    if args.command == "prepare-bulk":
        manifest = build_adjacent_change_pairs_from_directory(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            language=args.language,
            projects=args.projects,
        )
        print(
            f"Built {len(manifest)} adjacent-pair files at {Path(args.output_dir).resolve()} "
            f"and wrote manifest.csv"
        )
        return

    if args.command == "fit":
        source_pairs = load_change_pairs(args.sources)
        predictor = CrossProjectChangePredictor(backend="heuristic")
        predictor.fit(source_pairs)
        predictor.save(args.artifact_path)
        print(f"Saved retrieval artifact to {Path(args.artifact_path).resolve()}")
        return

    if args.command == "predict":
        predictor = CrossProjectChangePredictor(
            backend=args.backend,
            model=args.model,
            top_k=args.top_k,
        )
        predictor.load(args.artifact_path)
        target_pairs = load_change_pairs(args.target)
        results = predictor.predict_many(target_pairs)
        frame = predictor.to_frame(results)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Saved {len(frame)} predictions to {Path(args.output).resolve()}")
        return

    if args.command == "baseline-direct":
        source_pairs = load_change_pairs(args.sources)
        target_pairs = load_change_pairs(args.target)
        baseline = DirectTransferBaseline()
        prediction_frame, summary = baseline.run(
            source_pairs=source_pairs,
            target_pairs=target_pairs,
            source_name=args.source_name,
            target_name=args.target_name,
        )
        save_baseline_outputs(
            prediction_frame,
            summary,
            prediction_path=args.predictions_output,
            summary_path=args.summary_output,
        )
        print(
            f"Saved {len(prediction_frame)} baseline predictions to "
            f"{Path(args.predictions_output).resolve()} and summary to {Path(args.summary_output).resolve()}"
        )
        return

    if args.command == "baseline-static":
        source_pairs = load_change_pairs(args.sources)
        target_samples = load_static_file_samples(args.target_static)
        baseline = SourceChangeToStaticBaseline()
        prediction_frame, summary = baseline.run(
            source_pairs=source_pairs,
            target_samples=target_samples,
            source_name=args.source_name,
            target_name=args.target_name,
        )
        save_baseline_outputs(
            prediction_frame,
            summary,
            prediction_path=args.predictions_output,
            summary_path=args.summary_output,
        )
        print(
            f"Saved {len(prediction_frame)} static-target baseline predictions to "
            f"{Path(args.predictions_output).resolve()} and summary to {Path(args.summary_output).resolve()}"
        )
        return

    if args.command == "baseline-static-llm":
        source_pairs = load_change_pairs(args.sources)
        target_samples = load_static_file_samples(args.target_static)
        baseline = StaticTargetLLMBaseline(model=args.model, top_k=args.top_k)
        prediction_frame, summary = baseline.run(
            source_pairs=source_pairs,
            target_samples=target_samples,
            source_name=args.source_name,
            target_name=args.target_name,
        )
        save_baseline_outputs(
            prediction_frame,
            summary,
            prediction_path=args.predictions_output,
            summary_path=args.summary_output,
        )
        print(
            f"Saved {len(prediction_frame)} static-target LLM predictions to "
            f"{Path(args.predictions_output).resolve()} and summary to {Path(args.summary_output).resolve()}"
        )
        return

    if args.command == "baseline-static-debate":
        source_pairs = load_change_pairs(args.sources)
        target_samples = load_static_file_samples(args.target_static)
        baseline = StaticTargetDebateSystem(model=args.model, top_k=args.top_k)
        prediction_frame, summary = baseline.run(
            source_pairs=source_pairs,
            target_samples=target_samples,
            source_name=args.source_name,
            target_name=args.target_name,
        )
        save_baseline_outputs(
            prediction_frame,
            summary,
            prediction_path=args.predictions_output,
            summary_path=args.summary_output,
        )
        print(
            f"Saved {len(prediction_frame)} static-target debate predictions to "
            f"{Path(args.predictions_output).resolve()} and summary to {Path(args.summary_output).resolve()}"
        )
        return


if __name__ == "__main__":
    main()
