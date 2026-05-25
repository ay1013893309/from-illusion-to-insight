#!/usr/bin/env python
"""
Verification script to ensure all modules import correctly and key functionality
is available.
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_imports():
    """Test that all critical modules can be imported."""
    tests_passed = 0
    tests_failed = 0

    test_cases = [
        ("Core config", lambda: __import__("sdp.config", fromlist=["DATASET_NAME_RUN"])),
        ("Data loader", lambda: __import__("sdp.data.loader", fromlist=["load_dataset_pair"])),
        ("Diff utilities", lambda: __import__("sdp.analysis.diff", fromlist=["compute_diff"])),
        ("Java parsing", lambda: __import__("sdp.analysis.java_parse", fromlist=["find_java_classes"])),
        ("Hunk dataclass", lambda: __import__("sdp.analysis.hunk", fromlist=["Hunk"])),
        ("Verdict parser", lambda: __import__("sdp.analysis.verdict_parser", fromlist=["parse_judge_verdict"])),
        ("Metrics", lambda: __import__("sdp.analysis.metrics", fromlist=["compute_metrics"])),
        ("LLM wrapper", lambda: __import__("sdp.llm.wrapper", fromlist=["OpenAIWrapper"])),
        ("Expert system", lambda: __import__("sdp.llm.experts", fromlist=["ExpertDebateSystem"])),
        ("Prompts loader", lambda: __import__("sdp.prompts.loader", fromlist=["load_all_prompts"])),
        ("Prompt: Analyzer", lambda: __import__("sdp.prompts.analyzer", fromlist=["get_analyzer_prompt"])),
        ("Prompt: Proposer", lambda: __import__("sdp.prompts.proposer", fromlist=["get_proposer_prompt"])),
        ("Prompt: Skeptic", lambda: __import__("sdp.prompts.skeptic", fromlist=["get_skeptic_prompt"])),
        ("Prompt: Judge", lambda: __import__("sdp.prompts.judge", fromlist=["get_judge_prompt"])),
        ("Experiment: Orchestrator", lambda: __import__("sdp.experiments.orchestrator", fromlist=["run_all_model_combinations"])),
        ("Experiment: Evaluator", lambda: __import__("sdp.experiments.evaluator", fromlist=["test_skeptic_variants_async"])),
        ("Experiment: Visualization", lambda: __import__("sdp.experiments.visualization", fromlist=["plot_time_vs_debate_rounds"])),
        ("CLI", lambda: __import__("sdp.cli", fromlist=["main"])),
    ]

    print("=" * 70)
    print("IMPORT VERIFICATION TEST")
    print("=" * 70)

    for test_name, test_fn in test_cases:
        try:
            test_fn()
            print(f"[PASS] {test_name:.<50} PASS")
            tests_passed += 1
        except Exception as exc:
            print(f"[FAIL] {test_name:.<50} FAIL")
            print(f"   Error: {exc}")
            traceback.print_exc()
            tests_failed += 1

    print("=" * 70)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)

    return tests_failed == 0


def test_key_classes():
    """Test that key classes can be instantiated."""
    print("\n" + "=" * 70)
    print("CLASS INSTANTIATION TEST")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    try:
        from sdp.analysis.hunk import Hunk

        Hunk(
            file_path="test.java",
            src1="old code",
            src2="new code",
            unified_diff="diff",
            changes_dict={},
            relevant_context="context",
            label=1,
            old_label=0,
        )
        print(f"[PASS] {'Hunk dataclass creation':.<50} PASS")
        tests_passed += 1
    except Exception as exc:
        print(f"[FAIL] {'Hunk dataclass creation':.<50} FAIL")
        print(f"   Error: {exc}")
        tests_failed += 1

    try:
        from sdp.llm.wrapper import OpenAIWrapper

        OpenAIWrapper(api_keys=["test-key"], base_url="https://api.example.com/v1")
        print(f"[PASS] {'OpenAIWrapper instantiation':.<50} PASS")
        tests_passed += 1
    except Exception as exc:
        print(f"[FAIL] {'OpenAIWrapper instantiation':.<50} FAIL")
        print(f"   Error: {exc}")
        tests_failed += 1

    try:
        from sdp.llm.wrapper import OpenAIWrapper
        from sdp.llm.experts import ExpertDebateSystem

        wrapper = OpenAIWrapper(api_keys=["test-key"], base_url="https://api.example.com/v1")
        ExpertDebateSystem(llm_client=wrapper, max_rounds=1)
        print(f"[PASS] {'ExpertDebateSystem instantiation':.<50} PASS")
        tests_passed += 1
    except Exception as exc:
        print(f"[FAIL] {'ExpertDebateSystem instantiation':.<50} FAIL")
        print(f"   Error: {exc}")
        tests_failed += 1

    try:
        from sdp.analysis.verdict_parser import parse_confidence, parse_judge_verdict

        verdict, int_val = parse_judge_verdict("### Final Prediction: BENIGN")
        assert verdict == "BENIGN" and int_val == 0, "Verdict parsing failed"
        conf = parse_confidence("### Confidence: 85")
        assert conf == 85, "Confidence parsing failed"
        print(f"[PASS] {'Verdict parser functions':.<50} PASS")
        tests_passed += 1
    except Exception as exc:
        print(f"[FAIL] {'Verdict parser functions':.<50} FAIL")
        print(f"   Error: {exc}")
        tests_failed += 1

    try:
        import numpy as np
        from sdp.analysis.metrics import harmonic_mean, normalize_subset

        norm = normalize_subset("Benign_00")
        assert norm == "B00", "Subset normalization failed"
        hm = harmonic_mean(0.8, 0.6)
        assert not np.isnan(hm), "Harmonic mean computation failed"
        print(f"[PASS] {'Metrics computation':.<50} PASS")
        tests_passed += 1
    except Exception as exc:
        print(f"[FAIL] {'Metrics computation':.<50} FAIL")
        print(f"   Error: {exc}")
        tests_failed += 1

    try:
        from sdp.prompts.loader import load_all_prompts

        loader = load_all_prompts()
        prompt = loader.get_analyzer("diff", "src1", "src2", "BENIGN")
        assert "system" in prompt and "user" in prompt, "Prompt structure invalid"
        print(f"[PASS] {'Prompt loader':.<50} PASS")
        tests_passed += 1
    except Exception as exc:
        print(f"[FAIL] {'Prompt loader':.<50} FAIL")
        print(f"   Error: {exc}")
        tests_failed += 1

    print("=" * 70)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)

    return tests_failed == 0


def test_java_parsing():
    """Test Java parsing functionality."""
    print("\n" + "=" * 70)
    print("JAVA PARSING TEST")
    print("=" * 70)

    try:
        from sdp.analysis.java_parse import find_java_classes, find_java_methods

        sample_java = """
        public class TestClass {
            public void testMethod() {
                int x = 5;
            }
        }
        """

        classes = find_java_classes(sample_java)
        methods = find_java_methods(sample_java)

        assert len(classes) > 0, "No classes found"
        assert len(methods) > 0, "No methods found"

        print(f"[PASS] {'Java class detection':.<50} PASS")
        print(f"[PASS] {'Java method detection':.<50} PASS")
        print(f"   Found {len(classes)} class(es) and {len(methods)} method(s)")
        return True
    except Exception as exc:
        print(f"[FAIL] {'Java parsing':.<50} FAIL")
        print(f"   Error: {exc}")
        return False


if __name__ == "__main__":
    print("\n")
    print("=" * 72)
    print(" COMPLEX MODEL SDP - MODULE VERIFICATION ".center(72))
    print("=" * 72)
    print("\n")

    results = [
        ("Imports", test_imports()),
        ("Classes", test_key_classes()),
        ("Java Parsing", test_java_parsing()),
    ]

    print("\n" + "=" * 70)
    print("OVERALL VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = all(result[1] for result in results)

    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<50} {status}")

    print("=" * 70)

    if all_passed:
        print("\nALL VERIFICATION TESTS PASSED!")
        print("\nThe modular refactoring is complete and functional.")
        print("Ready to run experiments!\n")
        sys.exit(0)

    print("\nSOME TESTS FAILED - Please fix import errors above.")
    sys.exit(1)
