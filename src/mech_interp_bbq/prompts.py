"""Build base / nudge-stereo / nudge-other prompts for the 3-choice BBQ backfire setup.

This logic was previously inlined in ``scripts/run_backfire_3choice.py``.  It is
factored out here so the GPU caching phase (``collect_cache.py``) and the live
experiments (``run_live_experiments.py``) construct *identical* prompts, which is
essential for cache reuse and causal patching to line up.

The key subtlety is recovering the "other" (non-stereo, non-unknown) group's tag.
``BBQFullExample`` only stores the stereo group tag, so we match each full example
to its ``BBQPairExample`` (same ``example_id``) and read the tag of whichever pair
slot corresponds to the other answer text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import ContextCondition, load_bbq_full, load_bbq_pairwise
from .nudges import apply_nudge, group_label, singular_group_label


@dataclass
class PromptBundle:
    """Everything needed to run and classify the bidirectional backfire setup."""

    examples: list                  # list[BBQFullExample]
    baseline_prompts: list[str]
    nudge_stereo_prompts: list[str]
    nudge_other_prompts: list[str]
    stereo_ids: np.ndarray          # (n,) index of stereo answer choice
    unknown_ids: np.ndarray         # (n,) index of "cannot determine" choice
    other_ids: np.ndarray           # (n,) index of the other real group choice
    has_other_tag: np.ndarray       # (n,) bool — pairwise tag match succeeded

    @property
    def n(self) -> int:
        return len(self.baseline_prompts)


def build_prompts(
    category: str,
    nudge: str,
    context_condition: str = "ambig",
    max_examples: int = 10_000,
) -> PromptBundle:
    """Construct the prompt bundle for one (category, nudge, condition).

    Args:
        category: A HiTZ BBQ category, e.g. ``"Gender_identity"``.
        nudge: A key into ``NUDGE_TEMPLATES`` (e.g. ``"user_preference"``).
        context_condition: ``"ambig"``, ``"disambig"``, or ``"both"``.
        max_examples: Truncation for smoke tests.
    """
    cond: ContextCondition | None = (
        None if context_condition == "both" else context_condition  # type: ignore[assignment]
    )

    examples = load_bbq_full(category, context_condition=cond, max_examples=max_examples)
    pair_by_id = {ex.example_id: ex for ex in load_bbq_pairwise(category, context_condition=cond)}

    baseline_prompts: list[str] = []
    nudge_stereo_prompts: list[str] = []
    nudge_other_prompts: list[str] = []
    stereo_ids: list[int] = []
    unknown_ids: list[int] = []
    has_other_tag: list[bool] = []

    for ex in examples:
        stereo_tag = ex.stereotyped_groups[0] if ex.stereotyped_groups else "unknown"
        stereo_gl = group_label(category, stereo_tag)
        stereo_sg = singular_group_label(category, stereo_tag)

        base_prompt = ex.prompt()
        baseline_prompts.append(base_prompt)
        stereo_ids.append(ex.stereotyped_ans_id)
        unknown_ids.append(ex.unknown_ans_id)

        nudge_stereo_prompts.append(
            apply_nudge(
                base_prompt,
                nudge_type=nudge,
                group_label_str=stereo_gl,
                other_group_label_str="the other group",
                singular_group_label_str=stereo_sg,
            )
        )

        other_idx = ({0, 1, 2} - {ex.stereotyped_ans_id, ex.unknown_ans_id}).pop()
        other_ans_text = ex.answers[other_idx]
        pair = pair_by_id.get(ex.example_id)
        other_tag = None
        if pair is not None:
            if pair.group_a_answer.strip() == other_ans_text.strip():
                other_tag = pair.group_a_tag
            elif pair.group_b_answer.strip() == other_ans_text.strip():
                other_tag = pair.group_b_tag

        if other_tag is not None:
            other_gl = group_label(category, other_tag)
            other_sg = singular_group_label(category, other_tag)
            nudge_other_prompts.append(
                apply_nudge(
                    base_prompt,
                    nudge_type=nudge,
                    group_label_str=other_gl,
                    other_group_label_str=stereo_gl,
                    singular_group_label_str=other_sg,
                )
            )
            has_other_tag.append(True)
        else:
            nudge_other_prompts.append(base_prompt)  # no-op fallback
            has_other_tag.append(False)

    n = len(baseline_prompts)
    stereo_arr = np.array(stereo_ids, dtype=np.int64)
    unknown_arr = np.array(unknown_ids, dtype=np.int64)
    other_arr = np.array(
        [({0, 1, 2} - {int(stereo_arr[i]), int(unknown_arr[i])}).pop() for i in range(n)],
        dtype=np.int64,
    )

    return PromptBundle(
        examples=examples,
        baseline_prompts=baseline_prompts,
        nudge_stereo_prompts=nudge_stereo_prompts,
        nudge_other_prompts=nudge_other_prompts,
        stereo_ids=stereo_arr,
        unknown_ids=unknown_arr,
        other_ids=other_arr,
        has_other_tag=np.array(has_other_tag, dtype=bool),
    )


def cache_slug(model: str, category: str, nudge: str, condition: str, n: int) -> str:
    """Cache filename stem — matches the original ``run_backfire_3choice.py`` scheme."""
    return f"3choice_{model.replace('/', '_')}_{category}_{nudge}_{condition}_n{n}"


def model_slug(model: str) -> str:
    """Filesystem-safe model name, e.g. ``meta-llama/Llama-3.2-1B`` → ``meta-llama_Llama-3.2-1B``."""
    return model.replace("/", "_")


def model_cache_dir(cache_dir: str | Path, model: str) -> Path:
    """Per-model cache subdirectory, e.g. ``cache/Qwen_Qwen3-32B/``."""
    return Path(cache_dir) / model_slug(model)
