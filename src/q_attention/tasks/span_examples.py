"""Dataclasses for span-centric NLP examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class LabeledSpan:
    """A half-open token span with an optional label."""

    start: int
    end: int
    label: str | None = None

    def as_tuple(self) -> tuple[int, int]:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span: ({self.start}, {self.end})")
        return self.start, self.end


@dataclass(frozen=True)
class SpanCentricExample:
    """Generic span-centric NLP example."""

    text: str
    tokens: tuple[str, ...]
    anchors: tuple[LabeledSpan, ...]
    label: str
    evidence: tuple[LabeledSpan, ...] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class RelationExtractionExample(SpanCentricExample):
    """Relation extraction example with subject and object anchors."""

    subject: LabeledSpan | None = None
    object: LabeledSpan | None = None

    @classmethod
    def from_entities(
        cls,
        *,
        text: str,
        tokens: tuple[str, ...],
        subject: LabeledSpan,
        object: LabeledSpan,
        relation: str,
        evidence: tuple[LabeledSpan, ...] = (),
        metadata: Mapping[str, str] | None = None,
    ) -> "RelationExtractionExample":
        return cls(
            text=text,
            tokens=tokens,
            anchors=(subject, object),
            label=relation,
            evidence=evidence,
            metadata={} if metadata is None else metadata,
            subject=subject,
            object=object,
        )
