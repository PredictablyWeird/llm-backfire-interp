"""Prompt helpers for reason-before-answer experiments."""

from __future__ import annotations

REASONING_INSTRUCTION = "Think carefully step by step before choosing an answer."
REASONING_SUFFIX = f"{REASONING_INSTRUCTION}\nReasoning:"
ANSWER_SUFFIX = "\nAnswer:"

# Generation stops when the model begins a new Answer: line (we append it ourselves).
REASONING_STOP_STRINGS = ("\nAnswer:", "\nAnswer:\n")


def append_reasoning_scaffold(prompt: str) -> str:
    """Replace trailing ``Answer:`` with a reasoning scaffold (no ``Answer:`` yet)."""
    if prompt.rstrip().endswith("Answer:"):
        body = prompt.rstrip()[:-7].rstrip()
        return body + "\n" + REASONING_SUFFIX
    return prompt + "\n" + REASONING_SUFFIX


def finalize_after_reasoning(scaffold_prompt: str, reasoning: str) -> str:
    """Append generated reasoning and the final ``Answer:`` slot for logit scoring."""
    text = reasoning.strip()
    for stop in REASONING_STOP_STRINGS:
        if stop in text:
            text = text[: text.index(stop)].strip()
    return scaffold_prompt + text + ANSWER_SUFFIX


def truncate_reasoning(text: str) -> str:
    """Strip stop strings and whitespace from a raw generation."""
    out = text.strip()
    for stop in REASONING_STOP_STRINGS:
        if stop in out:
            out = out[: out.index(stop)].strip()
    return out
