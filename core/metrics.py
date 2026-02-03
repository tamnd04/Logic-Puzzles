from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import time
import tracemalloc

T = TypeVar("T")


@dataclass(frozen=True)
class Metrics(Generic[T]):
    result: T
    runtime_sec: float
    peak_mem_kb: float


def run_with_metrics(fn: Callable[[], T]) -> Metrics[T]:
    """Run `fn` and measure wall-clock runtime + peak Python memory (KB).

    Memory is measured via tracemalloc (Python allocations), not total OS RSS.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    exc: BaseException | None = None
    out: T | None = None
    try:
        out = fn()
    except BaseException as e:
        exc = e
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if exc is not None:
        raise exc
    assert out is not None
    return Metrics(result=out, runtime_sec=(t1 - t0), peak_mem_kb=(peak / 1024.0))
