"""Loaders for the BBQ (Bias Benchmark for QA) dataset.

Two mirrors are supported:

* ``walledai/BBQ`` – simple parquet mirror, used by the basic probe script.
* ``HiTZ/bbq``    – full-metadata mirror with ``answer_info``,
  ``context_condition``, and ``question_polarity``, used by the bias-probe
  experiment.  Configs are named ``<Category>_ambig`` / ``<Category>_disambig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from datasets import Dataset, concatenate_datasets, load_dataset

# ── simple mirror ────────────────────────────────────────────────────────────
WALLEDAI_ID = "walledai/BBQ"

CATEGORY_TO_SPLIT: dict[str, str] = {
    "Age": "age",
    "Disability_status": "disabilityStatus",
    "Gender_identity": "genderIdentity",
    "Nationality": "nationality",
    "Physical_appearance": "physicalAppearance",
    "Race_ethnicity": "raceEthnicity",
    "Race_x_SES": "raceXSes",
    "Race_x_gender": "raceXGender",
    "Religion": "religion",
    "SES": "ses",
    "Sexual_orientation": "sexualOrientation",
}

# ── full-metadata mirror ─────────────────────────────────────────────────────
HITZ_ID = "HiTZ/bbq"

# HiTZ configs expose only these categories (no Race_x_* or Race_x_gender)
HITZ_CATEGORIES = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
)

BBQ_CATEGORIES = tuple(CATEGORY_TO_SPLIT.keys())

ContextCondition = Literal["ambig", "disambig"]
Polarity = Literal["neg", "nonneg"]


# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class BBQExample:
    """Simple BBQ example (walledai mirror, no stereotype metadata)."""

    category: str
    context: str
    question: str
    answers: list[str]
    label: int

    def prompt(self) -> str:
        choices = "\n".join(
            f"{chr(ord('A') + i)}. {ans}" for i, ans in enumerate(self.answers)
        )
        return (
            f"Context: {self.context}\n"
            f"Question: {self.question}\n"
            f"{choices}\n"
            f"Answer:"
        )


@dataclass
class BBQFullExample:
    """BBQ example with full bias metadata (HiTZ mirror)."""

    example_id: int
    category: str
    context: str
    question: str
    answers: list[str]          # [ans0, ans1, ans2]
    label: int                  # gold answer index
    stereotyped_ans_id: int     # index of the stereotyped answer choice
    unknown_ans_id: int         # index of the "not enough info" answer
    context_condition: ContextCondition
    polarity: Polarity
    stereotyped_groups: list[str] = field(default_factory=list)

    def prompt(self) -> str:
        choices = "\n".join(
            f"{chr(ord('A') + i)}. {ans}" for i, ans in enumerate(self.answers)
        )
        return (
            f"Context: {self.context}\n"
            f"Question: {self.question}\n"
            f"{choices}\n"
            f"Answer:"
        )


# ── walledai loader ──────────────────────────────────────────────────────────

def load_bbq(
    category: str | None = None,
    max_examples: int | None = None,
) -> Dataset:
    """Load BBQ from ``walledai/BBQ``.

    Args:
        category: BBQ category (see ``BBQ_CATEGORIES``). ``None`` loads all.
        max_examples: Optional truncation for quick smoke tests.
    """
    if category is not None:
        if category not in CATEGORY_TO_SPLIT:
            raise ValueError(
                f"Unknown BBQ category: {category!r}. "
                f"Choose from: {list(CATEGORY_TO_SPLIT)}"
            )
        ds = load_dataset(WALLEDAI_ID, split=CATEGORY_TO_SPLIT[category])
    else:
        splits = [
            load_dataset(WALLEDAI_ID, split=s) for s in CATEGORY_TO_SPLIT.values()
        ]
        ds = concatenate_datasets(splits)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


def to_examples(ds: Dataset) -> list[BBQExample]:
    """Convert walledai rows into ``BBQExample`` instances."""
    examples: list[BBQExample] = []
    for row in ds:
        examples.append(
            BBQExample(
                category=row["category"],
                context=row["context"],
                question=row["question"],
                answers=list(row["choices"]),
                label=int(row["answer"]),
            )
        )
    return examples


# ── HiTZ loader ──────────────────────────────────────────────────────────────

def _derive_stereotype_idx(
    answer_info: dict,
    polarity: str,
    stereotyped_groups: list[str],
) -> int:
    """Return the index of the stereotyped answer choice.

    Logic:
    - ``neg`` polarity  → stereotyped answer = choice whose type_code matches a
      stereotyped group (e.g. "Who was the criminal?" → blames the stigmatised group).
    - ``nonneg`` polarity → stereotyped answer = non-unknown choice whose type_code
      does NOT match (e.g. "Who was the CEO?" → credits the privileged group).
    """
    sg_lower = {g.lower() for g in stereotyped_groups}

    def _is_stereotyped_group(type_code: str) -> bool:
        tc = type_code.lower()
        return tc in sg_lower or any(g in tc or tc in g for g in sg_lower)

    candidates: list[tuple[int, bool]] = []
    for key, (_, type_code) in answer_info.items():
        if type_code == "unknown":
            continue
        idx = int(key[-1])
        candidates.append((idx, _is_stereotyped_group(type_code)))

    if polarity == "neg":
        for idx, is_sg in candidates:
            if is_sg:
                return idx
    else:
        for idx, is_sg in candidates:
            if not is_sg:
                return idx

    # Fallback: return first non-unknown index
    return candidates[0][0]


def load_bbq_full(
    category: str,
    context_condition: ContextCondition | None = None,
    max_examples: int | None = None,
    split: str = "test",
) -> list[BBQFullExample]:
    """Load BBQ with full bias metadata from ``HiTZ/bbq``.

    Args:
        category: BBQ category (must be one of ``HITZ_CATEGORIES``).
        context_condition: ``"ambig"``, ``"disambig"``, or ``None`` for both.
        max_examples: Optional truncation.
        split: HF split to load (``"test"`` or ``"train"``).
    """
    if category not in HITZ_CATEGORIES:
        raise ValueError(
            f"Unknown category for HiTZ/bbq: {category!r}. "
            f"Available: {list(HITZ_CATEGORIES)}"
        )

    conditions: list[ContextCondition] = (
        ["ambig", "disambig"] if context_condition is None else [context_condition]
    )

    rows: list[BBQFullExample] = []
    for cond in conditions:
        config = f"{category}_{cond}"
        ds = load_dataset(HITZ_ID, config, split=split)
        for row in ds:
            answer_info: dict = row["answer_info"]
            polarity: str = row["question_polarity"]
            stereotyped_groups: list[str] = row["additional_metadata"]["stereotyped_groups"]

            unknown_idx = next(
                int(k[-1])
                for k, (_, tc) in answer_info.items()
                if tc == "unknown"
            )
            stereotyped_idx = _derive_stereotype_idx(
                answer_info, polarity, stereotyped_groups
            )

            rows.append(
                BBQFullExample(
                    example_id=int(row["example_id"]),
                    category=row["category"],
                    context=row["context"],
                    question=row["question"],
                    answers=[row["ans0"], row["ans1"], row["ans2"]],
                    label=int(row["label"]),
                    stereotyped_ans_id=stereotyped_idx,
                    unknown_ans_id=unknown_idx,
                    context_condition=cond,
                    polarity=polarity,
                    stereotyped_groups=stereotyped_groups,
                )
            )
            if max_examples is not None and len(rows) >= max_examples:
                return rows

    return rows


# ── Pairwise (binary) format ─────────────────────────────────────────────────

@dataclass
class BBQPairExample:
    """BBQ example converted to a binary forced-choice by dropping the 'unknown' answer.

    This mirrors the format used by the PredictablyWeird/Choices repo's BBQ
    adapter (``experiments/bbq/bbq_data.py``), enabling direction-flipped
    contextual-influence (nudge) experiments.

    ``group_a`` and ``group_b`` are the two non-unknown answer choices.
    The stereotyped group is identified by ``stereotyped_groups``.
    """

    example_id: int
    category: str
    context: str
    question: str
    group_a_answer: str
    group_a_tag: str            # type_code from answer_info, e.g. "old", "woman"
    group_b_answer: str
    group_b_tag: str
    correct_group: str | None   # "a", "b", or None (ambiguous — no correct answer)
    context_condition: ContextCondition
    polarity: Polarity
    stereotyped_groups: list[str] = field(default_factory=list)

    def prompt_with_sentence(self, sentence: str | None = None, position: str = "end") -> str:
        """Format as a binary A/B prompt, optionally injecting a nudge sentence."""
        body = (
            f"Context: {self.context}\n"
            f"Question: {self.question}\n"
            f"A. {self.group_a_answer}\n"
            f"B. {self.group_b_answer}\n"
            f"Answer:"
        )
        if sentence is None:
            return body
        if position == "start":
            return sentence + "\n\n" + body
        elif position == "end":
            body_no_answer = body[:-7].rstrip()
            return body_no_answer + "\n" + sentence + "\nAnswer:"
        elif position == "after_context":
            lines = body.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Question:"):
                    lines.insert(i, sentence)
                    lines.insert(i, "")
                    return "\n".join(lines)
        return body + "\n" + sentence


def _extract_pair(row: dict, cond: ContextCondition) -> BBQPairExample | None:
    """Convert a HiTZ BBQ row into a ``BBQPairExample``, dropping the unknown answer.

    Returns ``None`` if fewer or more than two non-unknown answers are found.
    """
    answer_info: dict = row["answer_info"]
    stereotyped_groups: list[str] = row["additional_metadata"]["stereotyped_groups"]

    non_unknown = []
    for key, (_, type_code) in answer_info.items():
        if type_code == "unknown":
            continue
        idx = int(key[-1])
        non_unknown.append({"idx": idx, "answer": row[key], "tag": type_code})

    if len(non_unknown) != 2:
        return None

    a, b = non_unknown[0], non_unknown[1]
    gold = int(row["label"])
    if gold == a["idx"]:
        correct_group = "a"
    elif gold == b["idx"]:
        correct_group = "b"
    else:
        correct_group = None  # gold was the unknown option (ambiguous example)

    return BBQPairExample(
        example_id=int(row["example_id"]),
        category=row["category"],
        context=row["context"],
        question=row["question"],
        group_a_answer=a["answer"],
        group_a_tag=a["tag"],
        group_b_answer=b["answer"],
        group_b_tag=b["tag"],
        correct_group=correct_group,
        context_condition=cond,
        polarity=row["question_polarity"],
        stereotyped_groups=stereotyped_groups,
    )


def load_bbq_pairwise(
    category: str,
    context_condition: ContextCondition | None = None,
    max_examples: int | None = None,
    split: str = "test",
) -> list[BBQPairExample]:
    """Load BBQ as binary forced-choice examples (unknown answer dropped).

    Uses the ``HiTZ/bbq`` mirror for full metadata. Each 3-choice example
    is converted to a 2-choice pair by discarding the "cannot be determined"
    option, matching the setup of the PredictablyWeird/Choices BBQ adapter.

    Args:
        category: BBQ category (must be in ``HITZ_CATEGORIES``).
        context_condition: ``"ambig"``, ``"disambig"``, or ``None`` for both.
        max_examples: Optional truncation.
        split: HF split (``"test"`` or ``"train"``).
    """
    if category not in HITZ_CATEGORIES:
        raise ValueError(
            f"Unknown category for HiTZ/bbq: {category!r}. "
            f"Available: {list(HITZ_CATEGORIES)}"
        )

    conditions: list[ContextCondition] = (
        ["ambig", "disambig"] if context_condition is None else [context_condition]
    )

    rows: list[BBQPairExample] = []
    for cond in conditions:
        config = f"{category}_{cond}"
        ds = load_dataset(HITZ_ID, config, split=split)
        for row in ds:
            pair = _extract_pair(row, cond)
            if pair is not None:
                rows.append(pair)
            if max_examples is not None and len(rows) >= max_examples:
                return rows

    return rows
