"""Map BBQ prompt strings to semantic regions and per-token region labels.

Regions (fixed order, matches ``REGION_NAMES``):
  context, question, choices, nudge

Used by ``collect_prompt_attention.py`` to aggregate last-token attention mass
by prompt section.
"""

from __future__ import annotations

import numpy as np

REGION_NAMES = ("context", "question", "choices", "nudge")
REGION_INDEX: dict[str, int] = {name: i for i, name in enumerate(REGION_NAMES)}
N_REGIONS = len(REGION_NAMES)


def append_ladder_nudge(base_prompt: str, sentence: str) -> str:
    """Insert *sentence* before the trailing ``Answer:`` (matches nudge_sensitivity)."""
    if base_prompt.rstrip().endswith("Answer:"):
        body = base_prompt.rstrip()[:-7].rstrip()
        return body + "\n" + sentence + "\nAnswer:"
    return base_prompt + "\n" + sentence + "\nAnswer:"


def char_region_spans(prompt: str) -> dict[str, tuple[int, int]]:
    """Character spans ``[start, end)`` for each region in a formatted BBQ prompt."""
    ans_start = prompt.rfind("\nAnswer:")
    if ans_start == -1:
        ans_start = prompt.rfind("Answer:")
    if ans_start == -1:
        raise ValueError(f"Prompt missing Answer: marker: {prompt[:120]!r}...")

    ctx_marker = "Context: "
    q_marker = "\nQuestion: "
    a_marker = "\nA. "

    ctx_start = prompt.find(ctx_marker)
    q_start = prompt.find(q_marker)
    a_start = prompt.find(a_marker)
    if min(ctx_start, q_start, a_start) == -1:
        raise ValueError(f"Malformed BBQ prompt: {prompt[:120]!r}...")

    spans: dict[str, tuple[int, int]] = {
        "context": (ctx_start + len(ctx_marker), q_start),
        "question": (q_start + len(q_marker), a_start),
    }

    choices_start = a_start + 1  # skip newline before ``A.``
    c_line_start = prompt.rfind("\nC. ", 0, ans_start)
    if c_line_start == -1:
        c_line_start = prompt.rfind("C. ", 0, ans_start)
    c_line_end = prompt.find("\n", c_line_start + 1)
    if c_line_end == -1 or c_line_end > ans_start:
        c_line_end = ans_start

    nudge_text = prompt[c_line_end:ans_start].strip()
    if nudge_text:
        nudge_start = prompt.index(nudge_text, c_line_end)
        spans["choices"] = (choices_start, nudge_start)
        spans["nudge"] = (nudge_start, nudge_start + len(nudge_text))
    else:
        spans["choices"] = (choices_start, ans_start)
        spans["nudge"] = (0, 0)

    return spans


def _assign_region(mid: int, spans: dict[str, tuple[int, int]]) -> int:
    for name in REGION_NAMES:
        start, end = spans[name]
        if name == "nudge" and start == end == 0:
            continue
        if start <= mid < end:
            return REGION_INDEX[name]
    return -1


def token_region_labels(
    tokenizer,
    prompt: str,
    *,
    add_special_tokens: bool = True,
) -> np.ndarray:
    """Region index per token (``0..3``), or ``-1`` for padding / unassigned."""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    spans = char_region_spans(prompt)
    labels = np.full(len(offsets), -1, dtype=np.int8)
    for i, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        mid = (start + end) // 2
        labels[i] = _assign_region(mid, spans)
    return labels


def batch_token_region_labels(
    tokenizer,
    prompts: list[str],
    *,
    add_special_tokens: bool = True,
) -> list[np.ndarray]:
    """Per-prompt region labels aligned with left-padded batch tokenization."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    out: list[np.ndarray] = []
    for b, prompt in enumerate(prompts):
        offsets = enc["offset_mapping"][b].tolist()
        spans = char_region_spans(prompt)
        labels = np.full(len(offsets), -1, dtype=np.int8)
        for i, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            mid = (start + end) // 2
            labels[i] = _assign_region(mid, spans)
        out.append(labels)
    return out
