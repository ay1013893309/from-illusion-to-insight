"""
Dataset loading and hunk construction helpers.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..analysis.diff import compute_diff, unified_diff
from ..analysis.file_matching import exact_path_matches, match_files_full
from ..analysis.hunk import Hunk
from ..analysis.java_parse import extract_relevant_context

FILE_LEVEL_DIR = Path(__file__).resolve().parent.parent / "Dataset" / "File-level"
DATASET_FILE_RE = re.compile(
    r"^(?P<dataset>[^-]+)-(?P<version>.+?)_ground-truth-files_dataset\.csv$"
)


def _version_key(version: str) -> List[object]:
    parts = re.split(r"(\d+)", version)
    key: List[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def _normalize_bug(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "defective", "buggy"}:
        return 1
    if normalized in {"0", "false", "no", "n", "benign", "clean"}:
        return 0
    return int(bool(normalized))


def _catalog() -> Dict[str, List[str]]:
    datasets: Dict[str, List[str]] = {}
    for path in FILE_LEVEL_DIR.glob("*_ground-truth-files_dataset.csv"):
        match = DATASET_FILE_RE.match(path.name)
        if not match:
            continue
        dataset = match.group("dataset")
        version = match.group("version")
        datasets.setdefault(dataset, []).append(version)

    for dataset, versions in datasets.items():
        datasets[dataset] = sorted(versions, key=_version_key)
    return datasets


def list_available_dataset_versions(dataset_name: Optional[str] = None):
    """
    List available dataset versions from the bundled CSV files.
    """
    catalog = _catalog()
    if dataset_name is None:
        return catalog
    if dataset_name not in catalog:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {', '.join(sorted(catalog))}"
        )
    return catalog[dataset_name]


def _resolve_pair(
    dataset_name: str,
    old_version: Optional[str] = None,
    new_version: Optional[str] = None,
) -> Tuple[str, str]:
    versions = list_available_dataset_versions(dataset_name)
    if len(versions) < 2:
        raise ValueError(f"Dataset '{dataset_name}' does not have at least two versions.")

    if old_version and old_version not in versions:
        raise KeyError(
            f"Version '{old_version}' not found for dataset '{dataset_name}'. Available: {versions}"
        )
    if new_version and new_version not in versions:
        raise KeyError(
            f"Version '{new_version}' not found for dataset '{dataset_name}'. Available: {versions}"
        )

    if old_version and new_version:
        if versions.index(old_version) >= versions.index(new_version):
            raise ValueError(
                f"Expected old_version < new_version, got '{old_version}' and '{new_version}'."
            )
        return old_version, new_version

    if new_version:
        new_idx = versions.index(new_version)
        if new_idx == 0:
            raise ValueError(
                f"Version '{new_version}' has no earlier paired version in dataset '{dataset_name}'."
            )
        return versions[new_idx - 1], new_version

    if old_version:
        old_idx = versions.index(old_version)
        if old_idx >= len(versions) - 1:
            raise ValueError(
                f"Version '{old_version}' has no later paired version in dataset '{dataset_name}'."
            )
        return old_version, versions[old_idx + 1]

    return versions[-2], versions[-1]


def _dataset_path(dataset_name: str, version: str) -> Path:
    return FILE_LEVEL_DIR / f"{dataset_name}-{version}_ground-truth-files_dataset.csv"


def _load_dataframe(dataset_name: str, version: str) -> pd.DataFrame:
    path = _dataset_path(dataset_name, version)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    last_error = None
    df = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if df is None:
        raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode dataset file")

    required_columns = {"File", "Bug", "SRC"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Dataset file {path.name} is missing required columns: {missing}")

    df = df.copy()
    df["Bug"] = df["Bug"].apply(_normalize_bug).astype(int)
    df.attrs["dataset_name"] = dataset_name
    df.attrs["dataset_version"] = version
    df.attrs["dataset_path"] = str(path)
    return df


def load_dataset_pair(
    dataset_name: str,
    old_version: Optional[str] = None,
    new_version: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a pair of dataset versions.

    If no versions are specified, the latest adjacent pair is used.
    """
    resolved_old, resolved_new = _resolve_pair(dataset_name, old_version, new_version)
    return (
        _load_dataframe(dataset_name, resolved_old),
        _load_dataframe(dataset_name, resolved_new),
    )


def load_datasets(
    dataset_names: Optional[Iterable[str]] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load the latest adjacent pair for one or more datasets.
    """
    catalog = _catalog()
    names = list(dataset_names) if dataset_names is not None else sorted(catalog)
    return {name: load_dataset_pair(name) for name in names}


def load_all_datasets() -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Backward-compatible alias for loading all datasets.
    """
    return load_datasets()


def match_files_and_create_hunks(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    *,
    use_ast_matching: bool = False,
    similarity_threshold: float = 0.8,
    max_context_lines: int = 400,
) -> List[Hunk]:
    """
    Match files across versions and construct changed-file hunks.

    By default, only exact path matches are used so the function works without
    optional Java/rapidfuzz dependencies.
    """
    if use_ast_matching:
        matches = match_files_full(
            df_old,
            df_new,
            src_col="SRC",
            file_col="File",
            similarity_threshold=similarity_threshold,
        )
    else:
        matches = exact_path_matches(df_old, df_new, file_col="File")

    old_lookup = df_old.set_index("File", drop=False)
    new_lookup = df_new.set_index("File", drop=False)

    hunks: List[Hunk] = []
    for _, match_row in matches.iterrows():
        old_file = match_row["df_old_file"]
        new_file = match_row["df_new_file"]
        old_row = old_lookup.loc[old_file]
        new_row = new_lookup.loc[new_file]

        src1 = old_row["SRC"]
        src2 = new_row["SRC"]
        diffs = compute_diff(src1, src2)
        if not any(diffs.values()):
            continue

        try:
            relevant_context = extract_relevant_context(
                src2,
                diffs,
                max_lines=max_context_lines,
            )
        except Exception:
            relevant_context = ""

        hunks.append(
            Hunk(
                file_path=str(new_file),
                src1=src1,
                src2=src2,
                unified_diff=unified_diff(src1, src2, fromfile=str(old_file), tofile=str(new_file)),
                changes_dict=diffs,
                relevant_context=relevant_context,
                label=int(new_row["Bug"]),
                old_label=int(old_row["Bug"]),
                metadata={
                    "old_file": str(old_file),
                    "new_file": str(new_file),
                    "similarity": float(match_row.get("similarity", 1.0)),
                },
            )
        )

    return hunks
