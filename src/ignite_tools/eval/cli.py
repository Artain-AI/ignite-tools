"""
CLI entry point for ``ignite-eval``.

Phases:
  1.   Read corpus
  1.5  Auto-select models (based on data signals + requirements + hardware)
  2.   Embed (subprocess per model, throughput + memory)
  3.   Score (pair generation, AUC with bootstrap CI)
  4.   Recommend (pick best model, explain why)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ignite_tools.core.cli import add_read_flags, load_config_from_args
from ignite_tools.core.config import FormatConfig
from ignite_tools.core.format import Item, ReadSummary, read_corpus
from ignite_tools.eval.models import load_registry, get_model_by_id
from ignite_tools.eval.requirements import EvalRequirements
from ignite_tools.eval.runner import (
    ask_download_consent,
    detect_device,
    download_models,
    estimate_download_size,
    run_embeddings,
)
from ignite_tools.eval.selector import DataSignals, select_models
from ignite_tools.read.core import detect_scripts


# Large-corpus threshold: if more than this many records and no sampling
# configured, we warn and offer to sample.
_LARGE_CORPUS_THRESHOLD = 10_000
_DEFAULT_SAMPLE_SIZE = 5_000


@dataclass
class _ReadPhase:
    items: list[Item]
    texts: list[str]
    lengths: list[int]
    summary: ReadSummary


@dataclass
class _DevicePhase:
    device: str
    has_gpu: bool


@dataclass
class _SelectionPhase:
    model_ids: list[str]
    model_names: dict[str, str]
    requirements: EvalRequirements | None


@dataclass
class _ScorePhase:
    has_labels: bool
    pairs: list
    score_result: object | None
    use_scores_for_recommendation: bool = False
    score_recommendation_reason: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ignite-eval",
        description="Evaluate embedding models on your data. "
        "Helps you pick the right model without needing to know "
        "what embedding models are.",
    )
    add_read_flags(parser)

    g = parser.add_argument_group("eval options")
    g.add_argument(
        "--models", type=str, default=None, metavar="MODEL1,MODEL2,...",
        help="Explicit model list (comma-separated HuggingFace IDs). "
        "Skips auto-selection.",
    )
    g.add_argument(
        "--max-models", type=int, default=4, metavar="N",
        help="Maximum models to auto-select (default 4).",
    )
    g.add_argument(
        "--show", type=int, default=5, metavar="N",
        help="Number of sample texts to display (default 5).",
    )
    g.add_argument(
        "--gpu", action="store_true", default=False,
        help="(Deprecated, use --device) Tell the selector you have a GPU.",
    )
    g.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for embedding. 'auto' (default) picks the best available. "
        "Use 'cpu' to benchmark CPU specifically even when GPU is available.",
    )
    g.add_argument(
        "--output-dir", type=str, default=None, metavar="PATH",
        help="Directory for embedding vectors (default: /tmp/ignite-eval-<id>/).",
    )
    g.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Skip actual embedding (model selection only, no downloads).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    loaded = load_config_from_args(args, tool_name="ignite-eval")

    if loaded.saved_to is not None:
        return 0

    if loaded.config is None:
        print("ERROR: no config resolved.", file=sys.stderr)
        return 1

    w = 60
    sep = "═" * w

    read_phase = _read_phase(args, loaded, sep, w)
    if isinstance(read_phase, int):
        return read_phase

    device_phase = _device_phase(args, w)
    registry = load_registry()

    selection = _selection_phase(
        args, loaded, read_phase.items, read_phase.lengths, device_phase, registry, sep, w
    )
    if isinstance(selection, int):
        return selection

    model_ids = selection.model_ids
    model_names = selection.model_names

    print(f"\n── Phase 2: Embed {'─' * (w - 19)}")

    if args.dry_run:
        print(f"  (dry run - skipping actual embedding)")
        print(f"  Would embed {len(read_phase.texts)} texts with {len(model_ids)} models.")
        print(f"\n{sep}")
        return 0

    model_ids = _download_phase(model_ids, registry, args, sep)
    if isinstance(model_ids, int):
        return model_ids
    run_result = _embed_phase(
        read_phase.texts, model_ids, model_names, device_phase.device, args, w
    )
    score_phase = _score_phase(
        read_phase.items,
        read_phase.texts,
        run_result,
        model_names,
        loaded.config.labels.field if loaded.config.labels else None,
        w,
    )
    requirements = selection.requirements
    _recommend_phase(
        score_phase,
        run_result,
        requirements,
        read_phase.texts,
        read_phase.lengths,
        w,
    )

    print(f"\n{sep}")
    return 0


def _read_phase(args: argparse.Namespace, loaded, sep: str, w: int) -> _ReadPhase | int:
    summary = ReadSummary()
    has_configured_sampling = (
        loaded.config.sampling is not None and loaded.config.sampling.mode != "full"
    )
    items = _read_eval_items(
        loaded.config,
        strict=loaded.strict,
        summary=summary,
        auto_yes=args.yes,
        has_configured_sampling=has_configured_sampling,
    )

    print(sep)
    print("  ignite-eval")
    print(sep)

    print(f"\n── Phase 1: Read {'─' * (w - 18)}")
    print(f"  Records loaded:  {len(items)}")
    print(f"  Files scanned:   {summary.files_scanned}")
    if summary.total_skipped:
        print(f"  Total skipped:   {summary.total_skipped}")

    if not items:
        print("\n  No records to evaluate. Check your config and data path.")
        print(f"\n{sep}")
        return 1

    labels = {it.label for it in items if it.label}
    sources = {it.source_file for it in items}
    lengths = [len(it.text) for it in items]

    print(f"  Labels:          {len(labels)} distinct")
    print(f"  Sources:         {len(sources)} files")
    print(f"  Text length:     {min(lengths)}–{max(lengths)} chars (avg {sum(lengths)//len(lengths)})")

    show = min(args.show, len(items))
    if show > 0:
        # Pick diverse samples (from different sources/labels, not just first N).
        sample_items = _pick_diverse_samples(items, show)
        print(f"\n  Sample ({len(sample_items)} of {len(items)}):")
        for item in sample_items:
            label_part = f" [{item.label}]" if item.label else ""
            text = item.text if len(item.text) <= 65 else item.text[:62] + "..."
            print(f"    {item.id}{label_part}: {text!r}")

    texts = [it.text for it in items]
    lengths = [len(t) for t in texts]  # recompute after potential sampling

    return _ReadPhase(items=items, texts=texts, lengths=lengths, summary=summary)


def _device_phase(args: argparse.Namespace, w: int) -> _DevicePhase:
    if args.device and args.device != "auto":
        device = args.device
    else:
        device = detect_device()

    has_gpu = device in ("cuda", "mps")

    print(f"\n── Hardware {'─' * (w - 13)}")
    if device == "mps":
        print(f"  Device:    Apple Silicon GPU (MPS), single device")
        print(f"  Impact:    medium-to-large models are practical, faster embedding")
    elif device == "cuda":
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_name = "NVIDIA GPU"
            gpu_count = 1
        print(f"  Device:    {gpu_name} x1 ({gpu_count} available)")
        print(f"  Impact:    large models are practical, much faster embedding")
    else:
        print(f"  Device:    CPU only")
        print(f"  Impact:    preferring smaller models to keep evaluation fast")

    return _DevicePhase(device=device, has_gpu=has_gpu)


def _selection_phase(
    args: argparse.Namespace,
    loaded,
    items: list[Item],
    lengths: list[int],
    device_phase: _DevicePhase,
    registry,
    sep: str,
    w: int,
) -> _SelectionPhase | int:
    print(f"\n── Phase 1.5: Model Selection {'─' * (w - 31)}")

    requirements = None

    if args.models:
        # User specified explicit models.
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_ids:
            print("ERROR: --models requires at least one model ID.", file=sys.stderr)
            print(f"\n{sep}")
            return 1
        model_names = {}
        for mid in model_ids:
            entry = get_model_by_id(registry, mid)
            model_names[mid] = entry.name if entry else mid
        print(f"  Mode: explicit ({len(model_ids)} user-specified models)")
        for mid in model_ids:
            print(f"    • {model_names[mid]} ({mid})")
    else:
        signals = _build_signals(items, lengths, device_phase.has_gpu)

        requirements = None
        if loaded.tool_block:
            try:
                requirements = EvalRequirements.model_validate(loaded.tool_block)
            except Exception as exc:
                print(f"  ERROR: invalid ignite-eval config: {exc}", file=sys.stderr)
                print(f"\n{sep}")
                return 1

        selection = select_models(
            signals,
            max_candidates=args.max_models,
            registry=registry,
            requirements=requirements,
        )

        print(f"  Three factors determine model selection:\n")

        # Factor 1: Data set
        print(f"  1. DATA SET (observed from your corpus):")
        lang_line = next((l for l in selection.signals_summary.split("\n") if "Languages" in l), "")
        print(f"     Language:    {lang_line.strip().replace('Languages: ', '') if lang_line else 'English (100%)'}")
        print(f"     Text length: {'short' if signals.is_short_text else 'long' if signals.is_long_text else 'medium'} (avg {signals.avg_text_length:.0f} chars)")
        print(f"     Corpus size: {signals.record_count:,} records")
        if signals.is_code:
            print(f"     Domain:      code/engineering")
        else:
            top = ", ".join(signals.top_words[:5])
            print(f"     Domain:      general ({top})")

        # Factor 2: Requirements
        print(f"\n  2. REQUIREMENTS (from config):")
        if requirements and (
            requirements.task
            or requirements.priority != "balanced"
            or requirements.mode != "batch"
            or requirements.prefer
            or requirements.constraints.max_size_mb
            or requirements.constraints.max_dim
            or requirements.constraints.min_quality
            or requirements.constraints.languages
        ):
            if requirements.task:
                print(f"     Task:     {requirements.task}")
            if requirements.priority != "balanced":
                print(f"     Priority: {requirements.priority}")
            if requirements.mode != "batch":
                print(f"     Mode:     {requirements.mode}")
            if requirements.prefer:
                print(f"     Prefer:   {requirements.prefer}")
            c = requirements.constraints
            if c.max_size_mb:
                print(f"     Max size: {c.max_size_mb} MB")
            if c.max_dim:
                print(f"     Max dim:  {c.max_dim}")
            if c.min_quality:
                print(f"     Min quality: {c.min_quality}")
            if c.languages:
                print(f"     Languages: {', '.join(c.languages)}")
        else:
            print(f"     (none specified - using data-driven defaults)")

        # Factor 3: Hardware
        print(f"\n  3. HARDWARE (detected):")
        print(f"     Device: {device_phase.device}")
        if device_phase.has_gpu:
            print(f"     Impact: can run medium-to-large models efficiently")
        else:
            print(f"     Impact: preferring smaller, faster models")

        print(f"\n  → Selected {len(selection.candidates)} models:\n")
        for i, choice in enumerate(selection.candidates, 1):
            m = choice.model
            print(f"  {i}. {m.name} ({m.id})")
            print(f"     {m.params} params, {m.dim}d, {m.speed_tier}, ~{m.size_mb} MB")
            print(f"     Why: {choice.reason}")
            print()

        if selection.skipped_reasons:
            print(f"  Skipped:")
            for reason in selection.skipped_reasons:
                print(f"    • {reason}")

        model_ids = [c.model.id for c in selection.candidates]
        model_names = {c.model.id: c.model.name for c in selection.candidates}

        if not model_ids:
            print("\n  No models matched your data + constraints. Try relaxing requirements.")
            print(f"\n{sep}")
            return 1

    return _SelectionPhase(
        model_ids=model_ids,
        model_names=model_names,
        requirements=requirements,
    )


def _download_phase(
    model_ids: list[str], registry, args: argparse.Namespace, sep: str
) -> list[str] | int:
    needs_download, total_mb = estimate_download_size(model_ids, registry)
    if needs_download:
        consent = ask_download_consent(
            needs_download, total_mb, auto_yes=args.yes
        )
        if not consent:
            print("  Aborted - no models downloaded.", file=sys.stderr)
            print(f"\n{sep}")
            return 1

        # Download all models at once before benchmarking.
        print(f"\n  ── Downloading ──", file=sys.stderr)
        dl_time, dl_results = download_models(needs_download)
        succeeded = sum(1 for v in dl_results.values() if v)
        failed = sum(1 for v in dl_results.values() if not v)
        print(f"  Downloaded {succeeded} models in {dl_time:.1f}s", file=sys.stderr)
        if failed:
            print(f"  WARNING: {failed} model(s) failed to download", file=sys.stderr)
            # Remove failed models from the benchmark list.
            model_ids = [mid for mid in model_ids if mid not in dl_results or dl_results.get(mid, True)]
        if not model_ids:
            print("  ERROR: all model downloads failed. Cannot benchmark.", file=sys.stderr)
            print(f"\n{sep}")
            return 1
        print()
    else:
        print(f"  All models already cached - no download needed.")

    return model_ids


def _embed_phase(
    texts: list[str],
    model_ids: list[str],
    model_names: dict[str, str],
    device: str,
    args: argparse.Namespace,
    w: int,
):
    print(f"\n  ── Benchmarking ──")
    print(f"  Corpus: {len(texts)} texts")
    print(f"  Models: {len(model_ids)}")
    print(f"  Device: {device}")
    print()

    run_result = run_embeddings(
        texts=texts,
        model_ids=model_ids,
        model_names=model_names,
        device=device,
        output_dir=args.output_dir,
    )

    # ── Results table ────────────────────────────────────────────────────
    print(f"\n  {'Model':<20} {'Texts/s':>8} {'Tokens/s':>9} {'Memory':>8} {'Dim':>5} {'Load':>6}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 9} {'─' * 8} {'─' * 5} {'─' * 6}")
    for r in run_result.results:
        if r.error:
            print(f"  {r.model_name:<20} ERROR: {r.error}")
        else:
            tok_str = f"{r.tokens_per_sec/1000:.1f}K" if r.tokens_per_sec >= 1000 else f"{r.tokens_per_sec:.0f}"
            print(
                f"  {r.model_name:<20} {r.throughput_warm:>7.0f}/s "
                f"{tok_str:>8}/s "
                f"{r.peak_memory_mb:>6.0f} MB "
                f"{r.embed_dim:>5}d "
                f"{r.load_time_s:>5.1f}s"
            )

    print(f"\n  Vectors saved to: {run_result.output_dir}")

    return run_result



def _score_phase(
    items: list[Item],
    texts: list[str],
    run_result,
    model_names: dict[str, str],
    label_field: str | None,
    w: int,
) -> _ScorePhase:
    print(f"\n── Phase 3: Score {'─' * (w - 19)}")

    item_labels = [it.label for it in items]
    has_labels = any(l is not None for l in item_labels)
    score_result = None
    pairs = []
    use_scores_for_recommendation = False
    score_recommendation_reason = ""

    if not has_labels:
        print("  No labels in data - cannot generate evaluation pairs.")
        print("  Add 'labels: { field: <your_label_field> }' to your config")
        print("  to enable quality scoring.")
    else:
        from ignite_tools.eval.recommend import assess_label_quality_signal
        from ignite_tools.eval.scorer import generate_pairs, score_all_models

        # Generate pairs from labeled items.
        pairs = generate_pairs(
            texts, item_labels,
            max_pairs=min(2000, len(texts) * 5),
            seed=42,
        )
        n_pos = sum(1 for p in pairs if p.label == "positive")
        n_neg = sum(1 for p in pairs if p.label == "negative")

        print(f"  What this does:")
        print(f"    We created {len(pairs)} test pairs from your labeled data:")
        print(f"    - {n_pos} 'same label' pairs (these SHOULD be similar)")
        print(f"    - {n_neg} 'different label' pairs (these SHOULD be different)")
        print(f"    Then we check: can each model tell the difference?")

        if pairs:
            # Load vectors for each model.
            import numpy as np
            vectors_by_model: dict[str, "np.ndarray"] = {}
            for r in run_result.results:
                if r.error or not r.vectors_path:
                    continue
                try:
                    vectors_by_model[r.model_id] = np.load(r.vectors_path)
                except Exception as exc:
                    print(
                        f"  WARNING: failed to load vectors for {r.model_name}: "
                        f"{r.vectors_path} ({exc})",
                        file=sys.stderr,
                    )

            if vectors_by_model:
                # Score all models.
                score_result = score_all_models(
                    vectors_by_model, pairs, model_names, seed=42
                )
                decision = assess_label_quality_signal(
                    label_field,
                    score_result.scores,
                )
                use_scores_for_recommendation = decision.use_for_recommendation
                score_recommendation_reason = decision.reason

                if use_scores_for_recommendation:
                    print(f"\n  Quality scoring:")
                    print(f"    Using label pairs to rank model quality.")
                else:
                    print(f"\n  Diagnostic label check:")
                    print(f"    Label scores will not drive the recommendation.")
                    print(f"    Reason: {score_recommendation_reason}")

                # Legend.
                print(f"\n  How to read the results:")
                print(f"    AUC:        How well the model separates your labels.")
                print(f"                1.0 = perfect, 0.5 = random guess, <0.5 = labels don't match semantics")
                print(f"    Separation: Avg similarity(same label) minus avg similarity(different label).")
                print(f"                Positive = model sees the difference. Higher = better.")
                print(f"    CI:         95% confidence interval - if intervals overlap, models are tied.")

                # Quality table.
                print(f"\n  {'Model':<25} {'AUC':>8} {'CI (95%)':>16} {'Separation':>11}")
                print(f"  {'─' * 25} {'─' * 8} {'─' * 16} {'─' * 11}")
                for ms in score_result.scores:
                    if ms.error:
                        print(f"  {ms.model_name:<25} ERROR: {ms.error}")
                    else:
                        ci_str = f"[{ms.auc_ci_low:.3f}, {ms.auc_ci_high:.3f}]"
                        print(
                            f"  {ms.model_name:<25} {ms.auc:>7.4f} {ci_str:>16} "
                            f"{ms.separation:>10.4f}"
                        )

                # Interpretation.
                print(f"\n  Interpretation:")
                best = score_result.scores[0] if score_result.scores else None
                if best and best.auc >= 0.7:
                    print(f"    {best.model_name} clearly separates your labels (AUC {best.auc:.3f}).")
                    if len(score_result.scores) > 1:
                        second = score_result.scores[1]
                        if best.auc_ci_low > second.auc_ci_high:
                            print(f"    It's statistically better than {second.model_name}.")
                        else:
                            print(f"    But CIs overlap with {second.model_name} - they may be tied.")
                elif best and best.auc >= 0.5:
                    print(f"    Models show weak separation. {best.model_name} is slightly ahead")
                    print(f"    but the difference is small. More data would give clearer results.")
                elif best:
                    print(f"    All models scored below 0.5 (worse than random).")
                    print(f"    This usually means: your labels don't strongly correspond to")
                    print(f"    semantic similarity in the text. Possible reasons:")
                    print(f"    - Too few texts per label (you have ~{len(texts) // max(len(set(l for l in item_labels if l)), 1)} per label)")
                    print(f"    - Labels are too fine-grained for the text length")
                    print(f"    - The text is too short for models to find semantic patterns")
                    print(f"    This doesn't mean the models are bad - it means the task")
                    print(f"    needs more data to evaluate properly.")

                # Sample size warning.
                if score_result.total_pairs < 100:
                    print(f"\n  NOTE: Only {score_result.total_pairs} pairs - results are noisy.")
                    print(f"  Add more labeled data for reliable estimates.")
            else:
                print("  No vectors loaded - cannot compute quality scores.")

    return _ScorePhase(
        has_labels=has_labels,
        pairs=pairs,
        score_result=score_result,
        use_scores_for_recommendation=use_scores_for_recommendation,
        score_recommendation_reason=score_recommendation_reason,
    )


def _recommend_phase(
    score_phase: _ScorePhase,
    run_result,
    requirements: EvalRequirements | None,
    texts: list[str],
    lengths: list[int],
    w: int,
) -> None:
    print(f"\n── Result {'─' * (w - 11)}")

    from ignite_tools.eval.recommend import recommend, recommend_operationally

    if score_phase.use_scores_for_recommendation:
        rec = recommend(
            embed_results=run_result.results,
            score_results=score_phase.score_result.scores if score_phase.score_result else [],
            requirements=requirements,
            corpus_size=len(texts),
        )
    else:
        if not score_phase.has_labels:
            note = "no labels were configured"
            print("  No labels configured; recommending from throughput and model size.")
        elif not score_phase.pairs:
            note = "not enough labeled pairs were generated"
            print("  Not enough labeled pairs; recommending from throughput and model size.")
        else:
            note = score_phase.score_recommendation_reason or "quality signal is unavailable"
            print("  Label scores are diagnostic only; recommending from operational results.")

        rec = recommend_operationally(
            embed_results=run_result.results,
            requirements=requirements,
            corpus_size=len(texts),
            note=note,
        )

    if rec is None:
        print("  Cannot generate recommendation (no valid results).")
        return

    # Confidence indicator.
    conf_icon = {"high": "+++", "medium": "++", "low": "+"}
    conf_note = {
        "high": "clear winner",
        "medium": "slight edge, CIs overlap",
        "low": "inconclusive - diagnostic recommendation",
    }

    print(f"\n  Recommendation: {rec.model_name}")
    print(f"  {rec.reason}")
    print(f"  Confidence: {conf_icon.get(rec.confidence, '?')} ({conf_note.get(rec.confidence, rec.confidence)})")

    if rec.runners_up:
        print(f"\n  Why not the others:")
        for explanation in rec.runners_up:
            print(f"    • {explanation}")

    # Hardware context.
    print(f"\n  Benchmarked on:")
    print(f"    {run_result.hardware_summary}")
    print(f"    Corpus: {len(texts)} texts, {min(lengths)}-{max(lengths)} chars")
    if run_result.device != "cpu":
        print(f"    Note: throughput numbers reflect this hardware.")
        print(f"    On CPU-only deployment, expect 2-5x slower.")

    if rec.scale_note:
        print(f"\n  Scale projection: {rec.scale_note}")


def _pick_diverse_samples(items: list, n: int) -> list:
    """Pick N samples spread across different labels/sources."""
    if not items:
        return []
    # Group by label.
    by_label: dict[str, list] = {}
    for item in items:
        key = item.label or item.source_file
        by_label.setdefault(key, []).append(item)
    # Round-robin from each group.
    result = []
    groups = list(by_label.values())
    idx = 0
    while len(result) < n:
        group = groups[idx % len(groups)]
        pos = len(result) // len(groups)
        if pos < len(group):
            result.append(group[pos])
        idx += 1
        if idx >= n * len(groups):  # safety
            break
    return result[:n]


def _read_eval_items(
    config: FormatConfig,
    *,
    strict: bool,
    summary: ReadSummary,
    auto_yes: bool,
    has_configured_sampling: bool,
) -> list[Item]:
    """Materialize eval items, applying large-corpus sampling while streaming."""
    stream = read_corpus(config, strict=strict, summary=summary)
    if has_configured_sampling:
        return list(stream)

    items: list[Item] = []
    for item in stream:
        items.append(item)
        if len(items) <= _LARGE_CORPUS_THRESHOLD:
            continue

        if auto_yes:
            print(
                f"  Auto-sampling {_DEFAULT_SAMPLE_SIZE:,} records "
                f"(large corpus + --yes).",
                file=sys.stderr,
            )
            return _reservoir_sample_eval(items, stream, _DEFAULT_SAMPLE_SIZE)

        print(
            f"\n  Your corpus has more than {_LARGE_CORPUS_THRESHOLD:,} records. "
            f"Embedding all of them with multiple models may take a while.",
            file=sys.stderr,
        )
        response = input(
            f"  Sample {_DEFAULT_SAMPLE_SIZE:,} records instead? [y/n/all] "
        ).strip().lower()
        if response in {"y", "yes", ""}:
            sampled = _reservoir_sample_eval(items, stream, _DEFAULT_SAMPLE_SIZE)
            print(f"  Sampled {len(sampled):,} records for evaluation.", file=sys.stderr)
            return sampled

        items.extend(stream)
        return items

    return items


def _reservoir_sample_eval(
    prefix_items: list[Item], stream, target: int, seed: int = 42
) -> list[Item]:
    """Reservoir sample over already-read prefix plus remaining stream."""
    import random

    rng = random.Random(seed)
    reservoir = prefix_items[:target]
    seen = target
    for item in prefix_items[target:]:
        seen += 1
        j = rng.randint(0, seen - 1)
        if j < target:
            reservoir[j] = item
    for item in stream:
        seen += 1
        j = rng.randint(0, seen - 1)
        if j < target:
            reservoir[j] = item
    return reservoir


def _build_signals(items: list, lengths: list[int], has_gpu: bool) -> DataSignals:
    """Build DataSignals from the read corpus."""
    import re
    from ignite_tools.read.core import DEFAULT_STOP_WORDS

    # Language detection.
    lang_counter: Counter[str] = Counter()
    for item in items:
        for script in detect_scripts(item.text):
            lang_counter[script] += 1
    total_lang = sum(lang_counter.values()) or 1
    languages = {s: c / total_lang for s, c in lang_counter.items()}

    # Top words.
    word_re = re.compile(r"[a-z][a-z0-9_-]*[a-z0-9]|[a-z]", re.IGNORECASE)
    word_counter: Counter[str] = Counter()
    for item in items:
        for match in word_re.finditer(item.text.lower()):
            word = match.group()
            if len(word) >= 3 and word not in DEFAULT_STOP_WORDS:
                word_counter[word] += 1
    top_words = [w for w, _ in word_counter.most_common(30)]

    return DataSignals(
        languages=languages,
        avg_text_length=sum(lengths) / len(lengths) if lengths else 0,
        max_text_length=max(lengths) if lengths else 0,
        record_count=len(items),
        top_words=top_words,
        has_gpu=has_gpu,
    )


if __name__ == "__main__":
    sys.exit(main())
