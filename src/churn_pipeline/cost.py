"""W4: hosting cost arithmetic for the three serving options.

Pure functions, no I/O. Every price is an INPUT supplied by the user; no AWS price is
hard-coded here, because real prices depend on region, instance type, date and any
discount (Savings Plans and so on). The results are what the formulas give for the
inputs: estimates for comparing options, not quotes.

Formulas (from the W4 plan):

* real-time, always on:  instances × price_per_instance_hour × hours
* serverless:            requests × seconds_per_request × memory_gb × price_per_gb_second
                         (+ requests × price_per_request, if given)
* batch:                 job_hours_per_day × price_per_instance_hour × days × instances

The auto scaling target turns a measured single-request latency into a per-instance
``InvocationsPerInstance`` target (invocations per minute), with headroom:

    target = 60 / latency_seconds × concurrency × utilization

Every numeric input must be a finite number > 0 (bool is rejected). ``utilization``
must also be ≤ 1. Zero is rejected on purpose: a zero price or volume is almost
always a data-entry mistake, and the plan's acceptance requires refusing it.
"""
from __future__ import annotations

import math


class CostInputError(ValueError):
    """An input is missing, non-numeric, non-finite, or not strictly positive."""


def _positive(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CostInputError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise CostInputError(f"{name} must be a finite number > 0")
    return float(value)


def realtime_cost(*, instances, price_per_instance_hour, hours) -> float:
    return (_positive("instances", instances) * _positive("price_per_instance_hour",
                                                          price_per_instance_hour)
            * _positive("hours", hours))


def serverless_cost(*, requests, seconds_per_request, memory_gb, price_per_gb_second,
                    price_per_request=None) -> float:
    compute = (_positive("requests", requests)
               * _positive("seconds_per_request", seconds_per_request)
               * _positive("memory_gb", memory_gb)
               * _positive("price_per_gb_second", price_per_gb_second))
    if price_per_request is None:
        return compute
    return compute + requests * _positive("price_per_request", price_per_request)


def batch_cost(*, job_hours_per_day, price_per_instance_hour, days, instances=1) -> float:
    return (_positive("job_hours_per_day", job_hours_per_day)
            * _positive("price_per_instance_hour", price_per_instance_hour)
            * _positive("days", days) * _positive("instances", instances))


def invocations_per_instance_target(*, latency_seconds, concurrency=1,
                                    utilization=0.5) -> float:
    """Invocations per minute per instance to use as an auto scaling target."""
    u = _positive("utilization", utilization)
    if u > 1:
        raise CostInputError("utilization must be <= 1")
    return 60.0 / _positive("latency_seconds", latency_seconds) \
        * _positive("concurrency", concurrency) * u


def compare(*, realtime: dict, serverless: dict, batch: dict) -> dict:
    """Cost of each option for the same period, cheapest first. Inputs are the keyword
    arguments of the three cost functions."""
    costs = {"realtime": realtime_cost(**realtime),
             "serverless": serverless_cost(**serverless),
             "batch": batch_cost(**batch)}
    return {"costs": costs, "cheapest_first": sorted(costs, key=costs.get),
            "note": "estimates from user-supplied prices; not AWS quotes"}
