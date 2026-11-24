"""
Regenerate exact Top-K repair candidate files from the LATEST RQ1 fairness
localization outputs.  Robust-cache-search revision.

This version is intentionally strict about provenance so that an old RQ1
localization/cache cannot be mixed with a newly rerun localization result.

Supported localization methods:
  - CFRepair
  - Arachne

Typical usage (latest RQ1 results, 4096 samples, seed 256):
    python read.py cfrepair adult 256 4096
    python read.py cfrepair bank 256 4096
    python read.py cfrepair default 256 4096

    python read.py arachne adult 256 4096
    python read.py arachne bank 256 4096
    python read.py arachne default 256 4096

Backward-compatible CFRepair shorthand:
    python read.py adult 256 4096

Batch regeneration:
    python read.py --batch \
      --methods cfrepair arachne \
      --models adult bank default \
      --seed 256 --samples 4096

Default behavior:
  1. Find the newest matching RQ1 localization summary by modification time.
  2. Read the sample fingerprint/protocol metadata from that summary.
  3. Resolve a full-ranking cache ONLY when it belongs to the same method,
     dataset, sample count, seed, and (when available) exact sample fingerprint.
  4. Generate the exact Top-K repair candidates.
  5. Overwrite the fixed candidate files used by the existing repair scripts.
     Existing candidate files are backed up first unless --no-backup is used.
  6. Write a manifest containing SHA-256 provenance for the localization and
     ranking source, so a later repair run can be traced back to the exact
     RQ1 localization result.

Important:
  - CFRepair candidate names are converted from actual model layer indices
    (e.g. Dense_5_42) to the sequential Dense/Conv indices expected by the
    CFRepair repair scripts (e.g. dense_1_42).
  - Arachne keeps the original actual model layer indices because its repair
    script resolves model.layers indices directly.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_METHODS = ("cfrepair", "arachne")
LOCALIZATION_NAME_RE = re.compile(
    r"top(?P<topk>\d+)_samples(?P<samples>\d+)_(?P<method>cfrepair|arachne)_seed(?P<seed>\d+)\.txt$",
    re.IGNORECASE,
)
PLAYER_RE = re.compile(
    r"^(?P<layer_type>[A-Za-z][A-Za-z0-9]*)_(?P<layer_idx>\d+)_(?P<neuron_idx>\d+)$"
)


def _normalize_method(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if raw in ("", "cfrepair", "cf-repair", "cf_repair"):
        return "cfrepair"
    if raw in ("arachne", "fi_gl", "fi+gl", "fi-gl", "arachne_fi_gl"):
        return "arachne"
    raise ValueError(f"Unsupported localization method: {value}. Use one of {SUPPORTED_METHODS}.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _backup_existing(paths: Sequence[Path], output_dir: Path) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = output_dir / "_previous_candidates" / stamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = output_dir / "_previous_candidates" / f"{stamp}-{suffix}"
        suffix += 1
    backup_dir.mkdir(parents=True, exist_ok=False)

    for path in existing:
        shutil.copy2(str(path), str(backup_dir / path.name))
    return backup_dir


def _load_model(model_path: str):
    """Import TensorFlow lazily; Arachne conversion does not require it."""
    from tensorflow.keras.models import load_model

    return load_model(model_path, compile=False)


def get_layer_index_mapping(model) -> Dict[Tuple[str, int], int]:
    """Map actual model layer indices to sequential Conv/Dense indices."""
    layer_to_seq: Dict[Tuple[str, int], int] = {}
    conv_count = 0
    dense_count = 0

    for actual_index, layer in enumerate(model.layers):
        try:
            weights = layer.get_weights()
        except Exception:
            continue
        if not weights:
            continue

        layer_class = type(layer).__name__.lower()
        if "conv" in layer_class:
            layer_to_seq[("conv", actual_index)] = conv_count
            conv_count += 1
        elif "dense" in layer_class:
            layer_to_seq[("dense", actual_index)] = dense_count
            dense_count += 1

    return layer_to_seq


def convert_player_name(player: str, layer_to_seq: Dict[Tuple[str, int], int]) -> str:
    """Convert Dense_5_42 to dense_1_42 for CFRepair repair scripts."""
    raw = str(player).strip()
    match = PLAYER_RE.match(raw)
    if not match:
        return raw

    layer_type = match.group("layer_type").lower()
    actual_index = int(match.group("layer_idx"))
    neuron_index = int(match.group("neuron_idx"))
    if layer_type.startswith("conv"):
        family = "conv"
    elif layer_type.startswith("dense"):
        family = "dense"
    else:
        return raw

    key = (family, actual_index)
    if key not in layer_to_seq:
        raise ValueError(
            f"Could not map localized neuron {raw}: actual model layer {family}_{actual_index} "
            "was not found among weighted Conv/Dense layers. Refusing to silently keep an "
            "unconverted CFRepair candidate because that can repair the wrong layer."
        )
    return f"{family}_{layer_to_seq[key]}_{neuron_index}"


def _read_json_or_text(path: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Localization file is empty: {path}")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        candidates = [item.strip() for item in re.split(r"[,\n]+", text) if item.strip()]
        return None, candidates

    if isinstance(parsed, dict):
        return parsed, []
    if isinstance(parsed, list):
        return None, [str(item).strip() for item in parsed if str(item).strip()]
    raise ValueError(f"Unsupported localization content in {path}: {type(parsed).__name__}")


def _parse_name_metadata(path: Path) -> Dict[str, Any]:
    match = LOCALIZATION_NAME_RE.search(path.name)
    if not match:
        return {"top_k": None, "sample_count": None, "seed": None, "method": None}
    return {
        "top_k": int(match.group("topk")),
        "sample_count": int(match.group("samples")),
        "seed": int(match.group("seed")),
        "method": _normalize_method(match.group("method")),
    }


def _infer_result_root(localization_file: Path) -> Path:
    """Infer .../result from .../result/fairness/<model>/<summary>."""
    for parent in localization_file.resolve().parents:
        if parent.name == "result":
            return parent
    raise ValueError(
        "Could not infer the result root from the localization path. Pass --cache-dir explicitly."
    )


def _method_cache_patterns(method: str, sample_count: Optional[int], seed: Optional[int]) -> List[str]:
    patterns: List[str] = []
    if method == "arachne":
        if sample_count is not None and seed is not None:
            patterns.extend(
                [
                    f"arachne_fairness_{sample_count}_seed{seed}_top*.json",
                    f"arachne_fairness*{sample_count}*seed{seed}*.json",
                    f"*arachne*fairness*{sample_count}*seed{seed}*.json",
                ]
            )
        if seed is not None:
            patterns.append(f"*arachne*fairness*seed{seed}*.json")
        patterns.extend(["arachne_fairness*.json", "*arachne*fairness*.json"])
    else:
        if sample_count is not None and seed is not None:
            patterns.extend(
                [
                    f"cfrepair_fairness*{sample_count}*seed{seed}*.json",
                    f"*cfrepair*fairness*{sample_count}*seed{seed}*.json",
                ]
            )
        if seed is not None:
            patterns.append(f"*cfrepair*fairness*seed{seed}*.json")
        patterns.extend(["cfrepair_fairness*.json", "*cfrepair*fairness*.json"])
    return patterns


def _candidate_cache_files(
    localization_file: Path,
    model_name: str,
    method: str,
    sample_count: Optional[int],
    seed: Optional[int],
    cache_dir: Optional[Path],
) -> List[Path]:
    """Return every plausible full-ranking JSON for this model.

    The old reader only globbed filenames such as ``cfrepair_fairness*.json``.
    That is too brittle after RQ1 code changes because the newest fairness locator
    may change the cache filename while keeping the same metadata/fingerprint.

    We now scan the model's whole ``result/loc_result/<model>`` directory and let
    the metadata checks (method/sample/seed/fingerprint) decide which JSON is exact.
    This preserves strict provenance without relying on one historical filename.
    """
    result_root = _infer_result_root(localization_file)
    primary = cache_dir if cache_dir is not None else (result_root / "loc_result" / model_name)

    if not primary.exists():
        raise FileNotFoundError(f"Localization cache directory not found: {primary}")

    found: List[Path] = []
    seen = set()

    def add(path: Path) -> None:
        if not path.is_file() or path.suffix.lower() != ".json":
            return
        resolved = str(path.resolve())
        if resolved not in seen:
            found.append(path)
            seen.add(resolved)

    # First keep the historical fast-path patterns.
    for pattern in _method_cache_patterns(method, sample_count, seed):
        for filename in glob.glob(str(primary / pattern)):
            add(Path(filename))

    # Then scan every JSON under the model cache directory.  This is the key fix
    # for renamed RQ1 cache files.
    for path in primary.rglob("*.json"):
        add(path)

    return found


def _cache_method_matches(data: Dict[str, Any], method: str, cache_file: Path) -> bool:
    cached_method = str(data.get("method", "")).strip().lower()
    filename = cache_file.name.lower()
    if method == "arachne":
        return (
            "arachne" in filename
            or cached_method in {"arachne", "fi_gl", "fi+gl", "arachne_fi_gl"}
            or "arachne" in cached_method
        )

    # The newest CFRepair fairness locator stores the *formula name* in the
    # cache ``method`` field rather than the literal string ``cfrepair``:
    # directional_soft_correctness_ig_times_pearson_times_directed_output_zeroout_causal_score
    # Accept that exact CFRepair-family signature as well as historical names.
    cfrepair_formula_cache = (
        "directional_soft_correctness" in cached_method
        and "pearson" in cached_method
        and "zeroout" in cached_method
    )
    return "cfrepair" in filename or "cfrepair" in cached_method or cfrepair_formula_cache


def _seed_matches(data: Dict[str, Any], cache_file: Path, expected_seed: Optional[int]) -> bool:
    if expected_seed is None:
        return True
    cached_seed = data.get("seed")
    if cached_seed is not None:
        try:
            return int(cached_seed) == int(expected_seed)
        except Exception:
            return False
    return re.search(rf"seed{int(expected_seed)}(?:\D|$)", cache_file.name, re.IGNORECASE) is not None


def _cache_matches(
    data: Dict[str, Any],
    cache_file: Path,
    method: str,
    model_name: str,
    sample_count: Optional[int],
    seed: Optional[int],
    fingerprint: Optional[str],
    strict_latest: bool,
) -> bool:
    if str(data.get("task", "fairness")).lower() != "fairness":
        return False
    if not _cache_method_matches(data, method, cache_file):
        return False

    cached_model = data.get("model")
    if cached_model is not None and str(cached_model).lower() != model_name.lower():
        return False

    if sample_count is not None:
        cached_count = data.get("actual_sample_count", data.get("num_examples"))
        if cached_count is None:
            if strict_latest:
                return False
        elif int(cached_count) != int(sample_count):
            return False

    if not _seed_matches(data, cache_file, seed):
        return False

    if fingerprint:
        cached_fingerprint = data.get("fairness_sample_fingerprint")
        if strict_latest:
            # This is the key stale-cache protection.  A cache without the exact
            # fingerprint of the newest summary is NOT allowed in strict mode.
            if not cached_fingerprint or cached_fingerprint != fingerprint:
                return False
        elif cached_fingerprint and cached_fingerprint != fingerprint:
            return False

    return isinstance(data.get("players"), list) and isinstance(data.get("scores"), list)


def _validate_summary_protocol(
    metadata: Optional[Dict[str, Any]],
    method: str,
    model_name: str,
    expected_sample_count: Optional[int],
    expected_seed: Optional[int],
    strict_latest: bool,
) -> None:
    if not metadata:
        if strict_latest:
            print(
                "Warning: localization summary is plain text, so protocol metadata/fingerprint "
                "cannot be verified from the summary itself. The newest matching file will still be used."
            )
        return

    task = str(metadata.get("task", "fairness")).lower()
    if task != "fairness":
        raise ValueError(f"Latest localization summary task is {task!r}, expected 'fairness'.")

    meta_model = metadata.get("model")
    if meta_model is not None and str(meta_model).lower() != model_name.lower():
        raise ValueError(f"Localization model mismatch: summary={meta_model}, requested={model_name}")

    meta_method = metadata.get("method")
    if meta_method is not None:
        try:
            normalized = _normalize_method(str(meta_method))
        except Exception:
            normalized = str(meta_method).strip().lower()
        if normalized != method:
            raise ValueError(f"Localization method mismatch: summary={meta_method}, requested={method}")

    if expected_sample_count is not None:
        actual = metadata.get("actual_sample_count", metadata.get("num_examples"))
        if actual is not None and int(actual) != int(expected_sample_count):
            raise ValueError(
                f"Latest localization sample-count mismatch: actual={actual}, expected={expected_sample_count}"
            )

    if expected_seed is not None and metadata.get("seed") is not None:
        if int(metadata["seed"]) != int(expected_seed):
            raise ValueError(
                f"Latest localization seed mismatch: summary={metadata['seed']}, expected={expected_seed}"
            )

    if strict_latest and metadata.get("localization_validation_same_samples") is False:
        raise ValueError(
            "Latest localization summary says localization_validation_same_samples=False. "
            "Refusing to generate new repair candidates from a non-unified fairness protocol."
        )


def _recursive_exact_cache_search(
    localization_file: Path,
    method: str,
    model_name: str,
    sample_count: Optional[int],
    seed: Optional[int],
    fingerprint: Optional[str],
    strict_latest: bool,
    already_seen: Sequence[Path],
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Last-resort exact search across the RQ1 result tree.

    This handles cases where a valid ranking JSON was written outside the usual
    ``result/loc_result/<model>`` directory.  It never weakens the fingerprint
    check: in strict mode the exact newest-summary fingerprint is still required.
    """
    if not fingerprint:
        return []
    result_root = _infer_result_root(localization_file)
    seen = {str(p.resolve()) for p in already_seen}
    matches: List[Tuple[Path, Dict[str, Any]]] = []
    scanned = 0
    for path in result_root.rglob("*.json"):
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        scanned += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("fairness_sample_fingerprint") != fingerprint:
            continue
        if _cache_matches(
            data=data,
            cache_file=path,
            method=method,
            model_name=model_name,
            sample_count=sample_count,
            seed=seed,
            fingerprint=fingerprint,
            strict_latest=strict_latest,
        ):
            matches.append((path, data))
    if scanned:
        print(f"[RQ1 cache fallback] recursively scanned {scanned} additional JSON file(s).")
    return matches


def _print_cache_fingerprint_diagnostics(cache_files: Sequence[Path]) -> None:
    rows = []
    for path in cache_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if not isinstance(data.get("players"), list) or not isinstance(data.get("scores"), list):
            continue
        rows.append((
            path,
            data.get("fairness_sample_fingerprint"),
            data.get("actual_sample_count", data.get("num_examples")),
            data.get("seed"),
            data.get("method"),
        ))
    if not rows:
        return
    rows.sort(key=lambda row: row[0].stat().st_mtime, reverse=True)
    print("\n[Cache diagnostics] full-ranking caches found but rejected:")
    for path, fp, count, seed, meth in rows[:12]:
        print(f"  {path.name}")
        print(f"    fingerprint={fp}")
        print(f"    samples={count}, seed={seed}, method={meth}")



def _resolve_summary_ranking_sidecar(
    localization_file: Path,
    metadata: Optional[Dict[str, Any]],
    method: str,
    model_name: str,
    sample_count: Optional[int],
    seed: Optional[int],
    fingerprint: Optional[str],
    strict_latest: bool,
) -> Optional[Tuple[List[str], List[float], Path]]:
    """Resolve the ranking sidecar written by the patched RQ1 locator.

    New RQ1 runs persist the full ranking next to the summary so candidate
    regeneration does not depend on the historical component-cache filename.
    """
    candidates: List[Path] = []
    seen = set()

    def add(path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    if metadata:
        raw = metadata.get("ranking_file")
        if raw:
            raw_path = Path(str(raw))
            add(raw_path)
            # If the project was moved/copied, try the sidecar basename beside
            # the summary as well as the absolute path recorded originally.
            add(localization_file.parent / raw_path.name)

    stem = localization_file.stem
    add(localization_file.with_name(stem + "_ranking.json"))

    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: failed to read ranking sidecar {path}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        if _cache_matches(
            data=data,
            cache_file=path,
            method=method,
            model_name=model_name,
            sample_count=sample_count,
            seed=seed,
            fingerprint=fingerprint,
            strict_latest=strict_latest,
        ):
            print(f"Resolved full {method} ranking from summary sidecar: {path}")
            return (
                [str(item) for item in data["players"]],
                [float(item) for item in data["scores"]],
                path,
            )
    return None


def resolve_ranked_players_and_scores(
    localization_file: Path,
    metadata: Optional[Dict[str, Any]],
    plain_candidates: Sequence[str],
    method: str,
    model_name: str,
    sample_count: Optional[int],
    seed: Optional[int],
    cache_dir: Optional[Path],
    strict_latest: bool = True,
) -> Tuple[List[str], List[float], Optional[Path]]:
    """Read players/scores from the summary or resolve its exact full-ranking cache."""
    if metadata and "players" in metadata and "scores" in metadata:
        players = [str(item) for item in metadata["players"]]
        scores = [float(item) for item in metadata["scores"]]
        if len(players) != len(scores):
            raise ValueError(
                f"Latest localization summary players/scores mismatch: {len(players)} vs {len(scores)}"
            )
        print(f"Resolved full {method} ranking directly from latest localization summary: {localization_file}")
        return players, scores, localization_file

    fingerprint = metadata.get("fairness_sample_fingerprint") if metadata else None

    sidecar = _resolve_summary_ranking_sidecar(
        localization_file=localization_file,
        metadata=metadata,
        method=method,
        model_name=model_name,
        sample_count=sample_count,
        seed=seed,
        fingerprint=fingerprint,
        strict_latest=strict_latest,
    )
    if sidecar is not None:
        return sidecar
    cache_files = _candidate_cache_files(
        localization_file=localization_file,
        model_name=model_name,
        method=method,
        sample_count=sample_count,
        seed=seed,
        cache_dir=cache_dir,
    )

    matching: List[Tuple[Path, Dict[str, Any]]] = []
    for cache_file in cache_files:
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: skipping unreadable cache {cache_file}: {exc}")
            continue
        if isinstance(data, dict) and _cache_matches(
            data=data,
            cache_file=cache_file,
            method=method,
            model_name=model_name,
            sample_count=sample_count,
            seed=seed,
            fingerprint=fingerprint,
            strict_latest=strict_latest,
        ):
            matching.append((cache_file, data))

    if not matching and fingerprint:
        matching.extend(
            _recursive_exact_cache_search(
                localization_file=localization_file,
                method=method,
                model_name=model_name,
                sample_count=sample_count,
                seed=seed,
                fingerprint=fingerprint,
                strict_latest=strict_latest,
                already_seen=cache_files,
            )
        )

    if matching:
        # Exact fingerprint matches are already mandatory in strict mode.  Newest
        # matching cache wins if several identical-protocol caches exist.
        matching.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
        cache_file, data = matching[0]
        print(f"Resolved full {method} localization ranking from: {cache_file}")
        if fingerprint:
            print(f"  exact fairness fingerprint: {fingerprint}")
        print(f"  ranking cache mtime: {_mtime_iso(cache_file)}")
        return (
            [str(item) for item in data["players"]],
            [float(item) for item in data["scores"]],
            cache_file,
        )

    # A Top-K plain-text summary is itself sufficient to regenerate the exact
    # repair candidates, because its order is the localization ranking order.
    if plain_candidates:
        print(
            "No exact full-ranking cache was found. Using the exact candidate order stored "
            "in the newest localization summary. Scores are unavailable and will be recorded as 0."
        )
        return list(plain_candidates), [0.0] * len(plain_candidates), None

    _print_cache_fingerprint_diagnostics(cache_files)
    attempted = _method_cache_patterns(method, sample_count, seed)
    extra = ""
    if fingerprint and strict_latest:
        extra = (
            f" Required exact fairness_sample_fingerprint={fingerprint}. "
            "Old/no-fingerprint caches are intentionally rejected."
        )
    raise FileNotFoundError(
        f"The newest {method} localization summary contains no players/scores and no exact matching "
        f"cache JSON was found, even after recursively scanning the RQ1 result tree. "
        f"Historical cache patterns tried: {attempted}.{extra} "
        "This summary was produced by the old save format, which overwrote the ranked-player text "
        "with metadata JSON. If its matching cache is gone, neuron identities are not present anywhere "
        "in the summary and cannot be reconstructed by read.py. Rerun this dataset once with the patched "
        "RQ1 locator that persists players/scores and a *_ranking.json sidecar."
    )


def _deduplicate_ranked(items: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    first_order: Dict[str, int] = {}
    for order, (player, score) in enumerate(items):
        player = str(player).strip()
        if not player:
            continue
        if player not in first_order:
            first_order[player] = order
            best[player] = float(score)
        else:
            best[player] = max(best[player], float(score))
    ranked = [(player, best[player]) for player in best]
    ranked.sort(key=lambda item: (-item[1], first_order[item[0]]))
    return ranked


def _deduplicate_preserve_order(items: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Used for plain Top-K summaries when score values are unavailable."""
    out: List[Tuple[str, float]] = []
    seen = set()
    for player, score in items:
        player = str(player).strip()
        if player and player not in seen:
            seen.add(player)
            out.append((player, float(score)))
    return out


def _validate_raw_players(players: Sequence[str], method: str) -> None:
    invalid = [player for player in players if not PLAYER_RE.match(str(player).strip())]
    if invalid:
        raise ValueError(
            f"{method} produced unsupported player names. Expected LayerType_layerIndex_neuronIndex; "
            f"examples: {invalid[:5]}"
        )


def generate_repair_candidates(
    localization_file: str,
    output_dir: str,
    method: str = "cfrepair",
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    top_k: Optional[int] = None,
    convert_layer_indices: Optional[bool] = None,
    strict_latest: bool = True,
    backup_existing: bool = True,
    expected_sample_count: Optional[int] = None,
    expected_seed: Optional[int] = None,
) -> Dict[str, Any]:
    method = _normalize_method(method)
    localization_path = Path(localization_file)
    if not localization_path.exists():
        raise FileNotFoundError(f"Localization file not found: {localization_path}")

    filename_meta = _parse_name_metadata(localization_path)
    metadata, plain_candidates = _read_json_or_text(localization_path)

    file_method = filename_meta.get("method")
    if file_method and file_method != method:
        raise ValueError(
            f"Requested method={method}, but localization filename identifies method={file_method}: "
            f"{localization_path.name}"
        )

    resolved_model = model_name or (
        str(metadata.get("model")) if metadata and metadata.get("model") else None
    )
    if not resolved_model:
        resolved_model = localization_path.parent.name

    _validate_summary_protocol(
        metadata=metadata,
        method=method,
        model_name=resolved_model,
        expected_sample_count=expected_sample_count,
        expected_seed=expected_seed,
        strict_latest=strict_latest,
    )

    resolved_top_k = top_k
    if resolved_top_k is None and metadata and metadata.get("top_k") is not None:
        resolved_top_k = int(metadata["top_k"])
    if resolved_top_k is None:
        resolved_top_k = filename_meta["top_k"]
    if resolved_top_k is None or resolved_top_k <= 0:
        raise ValueError("Could not determine Top-K. Pass --top-k explicitly.")

    sample_count = None
    if metadata:
        sample_count = metadata.get("actual_sample_count", metadata.get("num_examples"))
    if sample_count is None:
        sample_count = filename_meta["sample_count"]
    sample_count = int(sample_count) if sample_count is not None else None

    seed = filename_meta["seed"]
    if seed is None and metadata and metadata.get("seed") is not None:
        seed = int(metadata["seed"])

    players, scores, score_source = resolve_ranked_players_and_scores(
        localization_file=localization_path,
        metadata=metadata,
        plain_candidates=plain_candidates,
        method=method,
        model_name=resolved_model,
        sample_count=sample_count,
        seed=seed,
        cache_dir=Path(cache_dir) if cache_dir else None,
        strict_latest=strict_latest,
    )

    if len(players) != len(scores):
        raise ValueError(f"Localization players/scores length mismatch: {len(players)} vs {len(scores)}")

    if score_source is None and plain_candidates:
        ranked = _deduplicate_preserve_order(zip(players, scores))
    else:
        ranked = _deduplicate_ranked(zip(players, scores))

    if len(ranked) < resolved_top_k:
        raise ValueError(f"Only {len(ranked)} neurons are available, but Top-K={resolved_top_k}.")

    selected_raw = ranked[:resolved_top_k]
    raw_players = [player for player, _ in selected_raw]
    selected_scores = [float(score) for _, score in selected_raw]
    _validate_raw_players(raw_players, method)

    # CFRepair uses sequential Conv/Dense layer indices; Arachne uses actual model layer indices.
    if convert_layer_indices is None:
        convert_layer_indices = method == "cfrepair"
    if method == "arachne" and convert_layer_indices:
        raise ValueError(
            "Arachne candidates must keep actual model layer indices. Do not enable layer-index conversion."
        )

    output_players = list(raw_players)
    mapping_applied = False
    if convert_layer_indices:
        if not model_path:
            raise ValueError("CFRepair layer-index conversion requires --model-path.")
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model path not found: {model_path_obj}")
        print(f"Loading model for CFRepair layer-index conversion: {model_path_obj}")
        model = _load_model(str(model_path_obj))
        layer_to_seq = get_layer_index_mapping(model)
        output_players = [convert_player_name(player, layer_to_seq) for player in raw_players]
        mapping_applied = True

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = method
    players_file = out_dir / f"{prefix}_players.txt"
    scores_file = out_dir / f"{prefix}_fairness_vals.txt"
    manifest_file = out_dir / f"{prefix}_candidate_repair_neurons.json"

    backup_dir = None
    if backup_existing:
        backup_dir = _backup_existing([players_file, scores_file, manifest_file], out_dir)
        if backup_dir:
            print(f"Backed up previous repair candidates to: {backup_dir}")

    _atomic_write_text(players_file, ",".join(output_players))
    _atomic_write_text(scores_file, ",".join(f"{score:.17g}" for score in selected_scores))

    fingerprint = metadata.get("fairness_sample_fingerprint") if metadata else None
    manifest = {
        "task": "fairness",
        "localization_method": method,
        "model": resolved_model,
        "top_k": int(resolved_top_k),
        "sample_count": sample_count,
        "requested_sample_count": (
            metadata.get("requested_num_examples") if metadata else expected_sample_count
        ),
        "seed": seed,
        "fairness_sample_fingerprint": fingerprint,
        "localization_validation_same_samples": (
            metadata.get("localization_validation_same_samples") if metadata else None
        ),
        "strict_latest_rq1": bool(strict_latest),
        "generated_at_utc": _utc_now_iso(),
        "localization_file": str(localization_path.resolve()),
        "localization_file_mtime_utc": _mtime_iso(localization_path),
        "localization_file_sha256": _sha256(localization_path),
        "score_source": str(score_source.resolve()) if score_source else None,
        "score_source_mtime_utc": _mtime_iso(score_source) if score_source else None,
        "score_source_sha256": _sha256(score_source),
        "model_path": str(Path(model_path).resolve()) if model_path else None,
        "layer_index_semantics": (
            "sequential_conv_dense_index" if mapping_applied else "actual_model_layer_index"
        ),
        "layer_index_conversion_applied": bool(mapping_applied),
        "candidate_policy": "exact_latest_rq1_localized_top_k",
        "candidate_count": len(output_players),
        "raw_players": raw_players,
        "players": output_players,
        "scores": selected_scores,
        "players_file": str(players_file.resolve()),
        "scores_file": str(scores_file.resolve()),
        "previous_candidates_backup": str(backup_dir.resolve()) if backup_dir else None,
    }
    _atomic_write_text(manifest_file, json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"\n{method.upper()} exact Top-K repair candidates from LATEST RQ1 localization:")
    for rank, (raw_player, output_player, score) in enumerate(
        zip(raw_players, output_players, selected_scores), start=1
    ):
        if raw_player == output_player:
            print(f"  {rank}. {output_player}: {score:.10g}")
        else:
            print(f"  {rank}. {raw_player} -> {output_player}: {score:.10g}")

    print(f"\n{method.upper()} latest-RQ1 candidate regeneration complete")
    print(f"  Localization summary : {localization_path}")
    print(f"  Summary mtime        : {_mtime_iso(localization_path)}")
    print(f"  Summary SHA-256      : {_sha256(localization_path)}")
    print(f"  Full ranking source  : {score_source}")
    if score_source:
        print(f"  Ranking SHA-256      : {_sha256(score_source)}")
    print(f"  Sample count         : {sample_count}")
    print(f"  Seed                 : {seed}")
    print(f"  Fingerprint          : {fingerprint}")
    print(f"  Exact Top-K          : {resolved_top_k}")
    print(f"  Players              : {players_file}")
    print(f"  Scores               : {scores_file}")
    print(f"  Manifest             : {manifest_file}")
    return manifest


def _find_localization_file(
    rq1_root: Path,
    method: str,
    model_name: str,
    seed: int,
    sample_count: int,
) -> Path:
    """Find the newest exact RQ1 localization summary for this protocol."""
    directory = rq1_root / "result" / "fairness" / model_name
    pattern = f"top*_samples{sample_count}_{method}_seed{seed}.txt"
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No localization summary matched: {directory / pattern}")

    selected = matches[0]
    print("\n[Latest RQ1 localization selection]")
    print(f"  method       : {method}")
    print(f"  model        : {model_name}")
    print(f"  samples      : {sample_count}")
    print(f"  seed         : {seed}")
    print(f"  selected     : {selected}")
    print(f"  selected time: {_mtime_iso(selected)}")
    if len(matches) > 1:
        print(f"  matched files: {len(matches)} (newest selected)")
        for idx, path in enumerate(matches[:5], start=1):
            print(f"    {idx}. {_mtime_iso(path)}  {path.name}")
    return selected


def _run_one(
    *,
    method: str,
    model_name: str,
    seed: int,
    samples: int,
    rq1_root: Path,
    output_root: Path,
    model_path: Optional[str],
    cache_dir: Optional[str],
    top_k: Optional[int],
    no_layer_index_conversion: bool,
    strict_latest: bool,
    backup_existing: bool,
) -> Dict[str, Any]:
    localization_file = _find_localization_file(
        rq1_root=rq1_root,
        method=method,
        model_name=model_name,
        seed=seed,
        sample_count=samples,
    )

    output_dir = output_root / model_name

    if method == "arachne":
        convert_layer_indices = False
        resolved_model_path = model_path
    else:
        convert_layer_indices = not no_layer_index_conversion
        resolved_model_path = model_path
        if resolved_model_path is None and convert_layer_indices:
            resolved_model_path = str(Path("..") / "model" / f"{model_name}.h5")

    return generate_repair_candidates(
        localization_file=str(localization_file),
        output_dir=str(output_dir),
        method=method,
        model_path=resolved_model_path,
        model_name=model_name,
        cache_dir=cache_dir,
        top_k=top_k,
        convert_layer_indices=convert_layer_indices,
        strict_latest=strict_latest,
        backup_existing=backup_existing,
        expected_sample_count=samples,
        expected_seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate repair candidates from the newest matching RQ1 fairness localization, "
            "with strict stale-cache/fingerprint protection."
        )
    )
    parser.add_argument(
        "shorthand",
        nargs="*",
        help=(
            "Either '<model> <seed> <samples>' for CFRepair compatibility, or "
            "'<method> <model> <seed> <samples>'."
        ),
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--localization-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--rq1-root", default="../cfrepair_RQ1")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--no-layer-index-conversion",
        action="store_true",
        help="For CFRepair, keep raw actual model layer indices.",
    )
    parser.add_argument(
        "--allow-stale-cache",
        action="store_true",
        help=(
            "Disable strict latest-RQ1 fingerprint checking. Not recommended for the rerun described here."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Overwrite current candidate files without backing them up first.",
    )

    parser.add_argument("--batch", action="store_true", help="Regenerate several datasets/methods at once.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=None,
        help="Methods for --batch. Default: cfrepair.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Datasets for --batch, e.g. adult bank default.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for --batch.")
    parser.add_argument("--samples", type=int, default=None, help="Sample count for --batch.")
    parser.add_argument(
        "--output-root",
        default="loc_result",
        help="Batch output root. Each dataset is written to <output-root>/<dataset>.",
    )
    return parser


def _parse_shorthand(
    tokens: Sequence[str], explicit_method: Optional[str]
) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
    if not tokens:
        return _normalize_method(explicit_method), None, None, None
    if len(tokens) == 3:
        method = _normalize_method(explicit_method or "cfrepair")
        model_name, seed_raw, samples_raw = tokens
    elif len(tokens) == 4:
        token_method, model_name, seed_raw, samples_raw = tokens
        method = _normalize_method(explicit_method or token_method)
        if explicit_method and _normalize_method(token_method) != method:
            raise ValueError("Conflicting positional method and --method values.")
    else:
        raise ValueError(
            "Shorthand must be either '<model> <seed> <samples>' or "
            "'<method> <model> <seed> <samples>'."
        )
    return method, model_name, int(seed_raw), int(samples_raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    strict_latest = not args.allow_stale_cache
    backup_existing = not args.no_backup

    if args.batch:
        if args.shorthand or args.localization_file:
            raise SystemExit("--batch cannot be combined with positional shorthand or --localization-file.")
        methods = [_normalize_method(m) for m in (args.methods or ["cfrepair"])]
        models = args.models or ["adult", "bank", "default"]
        if args.seed is None or args.samples is None:
            raise SystemExit("--batch requires --seed and --samples.")

        failures: List[Tuple[str, str, str]] = []
        completed = 0
        for method in methods:
            for model_name in models:
                print("\n" + "=" * 80)
                print(
                    f"Regenerating latest RQ1 candidates: method={method}, model={model_name}, "
                    f"seed={args.seed}, samples={args.samples}"
                )
                print("=" * 80)
                try:
                    _run_one(
                        method=method,
                        model_name=model_name,
                        seed=args.seed,
                        samples=args.samples,
                        rq1_root=Path(args.rq1_root),
                        output_root=Path(args.output_root),
                        model_path=args.model_path,
                        cache_dir=args.cache_dir,
                        top_k=args.top_k,
                        no_layer_index_conversion=args.no_layer_index_conversion,
                        strict_latest=strict_latest,
                        backup_existing=backup_existing,
                    )
                    completed += 1
                except Exception as exc:
                    failures.append((method, model_name, str(exc)))
                    print(f"ERROR [{method}/{model_name}]: {exc}", file=sys.stderr)

        print("\n" + "=" * 80)
        print(f"Batch complete: {completed} succeeded, {len(failures)} failed")
        if failures:
            for method, model_name, message in failures:
                print(f"  FAILED {method}/{model_name}: {message}")
            return 1
        return 0

    method, shorthand_model, shorthand_seed, shorthand_samples = _parse_shorthand(
        args.shorthand, args.method
    )

    if args.localization_file:
        localization_file = Path(args.localization_file)
        filename_meta = _parse_name_metadata(localization_file)
        if args.method is None and not args.shorthand and filename_meta.get("method"):
            method = filename_meta["method"]
        model_name = args.model_name or shorthand_model or localization_file.parent.name
        expected_seed = shorthand_seed or filename_meta.get("seed")
        expected_samples = shorthand_samples or filename_meta.get("sample_count")

        if method == "arachne":
            convert_layer_indices = False
            model_path = args.model_path
        else:
            convert_layer_indices = not args.no_layer_index_conversion
            model_path = args.model_path
            if model_path is None and convert_layer_indices:
                model_path = str(Path("..") / "model" / f"{model_name}.h5")

        output_dir = args.output_dir or str(Path("loc_result") / model_name)
        generate_repair_candidates(
            localization_file=str(localization_file),
            output_dir=output_dir,
            method=method,
            model_path=model_path,
            model_name=model_name,
            cache_dir=args.cache_dir,
            top_k=args.top_k,
            convert_layer_indices=convert_layer_indices,
            strict_latest=strict_latest,
            backup_existing=backup_existing,
            expected_sample_count=expected_samples,
            expected_seed=expected_seed,
        )
        return 0

    if shorthand_model is None or shorthand_seed is None or shorthand_samples is None:
        raise SystemExit(
            "Use explicit --localization-file, shorthand, or --batch:\n"
            "  python read.py adult 256 4096\n"
            "  python read.py arachne adult 256 4096\n"
            "  python read.py --batch --methods cfrepair arachne --models adult bank default --seed 256 --samples 4096"
        )

    model_name = args.model_name or shorthand_model
    output_root = Path(args.output_dir).parent if args.output_dir else Path("loc_result")

    # If a fully explicit --output-dir was supplied, bypass _run_one's dataset suffix.
    if args.output_dir:
        localization_file = _find_localization_file(
            rq1_root=Path(args.rq1_root),
            method=method,
            model_name=model_name,
            seed=shorthand_seed,
            sample_count=shorthand_samples,
        )
        if method == "arachne":
            convert_layer_indices = False
            model_path = args.model_path
        else:
            convert_layer_indices = not args.no_layer_index_conversion
            model_path = args.model_path
            if model_path is None and convert_layer_indices:
                model_path = str(Path("..") / "model" / f"{model_name}.h5")
        generate_repair_candidates(
            localization_file=str(localization_file),
            output_dir=args.output_dir,
            method=method,
            model_path=model_path,
            model_name=model_name,
            cache_dir=args.cache_dir,
            top_k=args.top_k,
            convert_layer_indices=convert_layer_indices,
            strict_latest=strict_latest,
            backup_existing=backup_existing,
            expected_sample_count=shorthand_samples,
            expected_seed=shorthand_seed,
        )
    else:
        _run_one(
            method=method,
            model_name=model_name,
            seed=shorthand_seed,
            samples=shorthand_samples,
            rq1_root=Path(args.rq1_root),
            output_root=Path("loc_result"),
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            top_k=args.top_k,
            no_layer_index_conversion=args.no_layer_index_conversion,
            strict_latest=strict_latest,
            backup_existing=backup_existing,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
