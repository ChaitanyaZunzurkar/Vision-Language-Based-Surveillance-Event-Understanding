"""Lightweight stage timing for pipeline diagnostics and benchmarks."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterator


@dataclass
class StageTimings:
    values: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.values[name] = round(perf_counter() - started, 4)

    @property
    def total_sec(self) -> float:
        return round(sum(self.values.values()), 4)
