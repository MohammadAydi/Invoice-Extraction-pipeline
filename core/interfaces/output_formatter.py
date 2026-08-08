from __future__ import annotations

from typing import Any, Protocol

from core.domain.result import PipelineResult


class OutputFormatter(Protocol):
    def format(self, result: PipelineResult) -> Any: ...
