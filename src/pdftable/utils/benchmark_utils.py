# -*- coding: utf-8 -*-
"""
The util functions used in inference speed comparison.
"""

import time
from contextlib import contextmanager
from typing import List, Dict

import numpy as np


def print_timings(name: str, timings: List[float]) -> Dict[str, float]:
    """
    Format and print inference latencies.
    :param name: inference engine name
    :param timings: latencies measured during the inference
    """
    mean_time = 1e3 * np.mean(timings)
    std_time = 1e3 * np.std(timings)
    min_time = 1e3 * np.min(timings)
    max_time = 1e3 * np.max(timings)
    total_time = np.sum(timings)
    median, percent_95_time, percent_99_time = 1e3 * np.percentile(timings, [50, 95, 99])
    print(
        f"[{name}] "
        f"mean={mean_time:.2f}ms, "
        f"sd={std_time:.2f}ms, "
        f"min={min_time:.2f}ms, "
        f"max={max_time:.2f}ms, "
        f"total={total_time:.2f}ms, "
        f"median={median:.2f}ms, "
        f"95p={percent_95_time:.2f}ms, "
        f"99p={percent_99_time:.2f}ms"
    )

    results = {
        "name": name,
        "mean": mean_time,
        "sd": std_time,
        "min": min_time,
        "max": max_time,
        "median": median,
        "95p": percent_95_time,
        "99p": percent_99_time,
        "total": total_time
        # "timings": timings
    }
    return results


@contextmanager
def track_infer_time(buffer: List[int]) -> None:
    """
    A context manager to perform latency measures
    :param buffer: a List where to save latencies for each input
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    buffer.append(end - start)
