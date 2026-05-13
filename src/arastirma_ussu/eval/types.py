"""Layer 5.5 — RAGAS judge result types."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JudgeResult:
    """Output of a single RAGAS-style evaluation pass.

    All metric fields are in ``[0.0, 1.0]``; ``nan`` if not computed
    (e.g., ``context_recall`` without a ground-truth reference).
    """

    faithfulness: float
    answer_relevancy: float
    context_recall: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        """Mean of the three metrics, excluding ``nan`` entries.

        Returns ``nan`` when every metric is ``nan``.
        """
        vals = [
            v
            for v in (self.faithfulness, self.answer_relevancy, self.context_recall)
            if not math.isnan(v)
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def is_passing(self, threshold: float = 0.7) -> bool:
        """Return True when ``overall`` is finite and ``>= threshold``."""
        ov = self.overall
        return (not math.isnan(ov)) and ov >= threshold
