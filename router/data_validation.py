"""Validation helpers for prompt--model action-space coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ActionSpaceIssue:
    qid: str
    action_count: int
    unique_pair_count: int
    missing_pairs: tuple[tuple[str, str], ...]
    unexpected_pairs: tuple[tuple[str, str], ...]

    def summary(self) -> str:
        return (
            f"{self.qid}: actions={self.action_count}, "
            f"unique_pairs={self.unique_pair_count}, "
            f"missing={list(self.missing_pairs)}, "
            f"unexpected={list(self.unexpected_pairs)}"
        )


def find_action_space_issues(
    rows: Iterable[dict],
    expected_prompts: Sequence[str],
    expected_models: Sequence[str],
) -> list[ActionSpaceIssue]:
    """Return every query whose observed action space is not the full Cartesian set."""
    expected_pairs = {
        (prompt, model)
        for prompt in expected_prompts
        for model in expected_models
    }
    issues: list[ActionSpaceIssue] = []

    for row in rows:
        edges = row.get("action_edges", [])
        pairs = [
            (edge.get("prompt"), edge.get("model"))
            for edge in edges
        ]
        unique_pairs = set(pairs)
        if len(pairs) == len(expected_pairs) and unique_pairs == expected_pairs:
            continue

        issues.append(
            ActionSpaceIssue(
                qid=str(row.get("qid")),
                action_count=len(pairs),
                unique_pair_count=len(unique_pairs),
                missing_pairs=tuple(sorted(expected_pairs - unique_pairs)),
                unexpected_pairs=tuple(sorted(unique_pairs - expected_pairs)),
            )
        )

    return issues


def require_complete_action_space(
    rows: Iterable[dict],
    expected_prompts: Sequence[str],
    expected_models: Sequence[str],
) -> None:
    """Fail before training when any query has a partial or duplicated action space."""
    issues = find_action_space_issues(rows, expected_prompts, expected_models)
    if not issues:
        return

    examples = "\n".join(f"  - {issue.summary()}" for issue in issues[:10])
    remainder = len(issues) - min(len(issues), 10)
    suffix = f"\n  ... {remainder} additional queries" if remainder else ""
    raise ValueError(
        f"Found {len(issues)} queries with incomplete or invalid action spaces.\n"
        f"{examples}{suffix}\n"
        "Build a complete-only dataset before training."
    )
