"""W4: cost arithmetic. Hand-computed examples use binary-exact numbers, so results are
compared with ==, as the plan's acceptance requires. Prices here are made up."""
import math

import pytest

from churn_pipeline import cost


def test_realtime_hand_example():
    # 2 instances x 0.25/h x 720 h = 360
    assert cost.realtime_cost(instances=2, price_per_instance_hour=0.25, hours=720) == 360.0


def test_serverless_hand_example_with_and_without_request_price():
    # 1000 requests x 0.5 s x 2 GB x 0.25 per GB-s = 250
    kw = dict(requests=1000, seconds_per_request=0.5, memory_gb=2, price_per_gb_second=0.25)
    assert cost.serverless_cost(**kw) == 250.0
    # + 1000 requests x 0.5 per request = 500 -> 750
    assert cost.serverless_cost(**kw, price_per_request=0.5) == 750.0


def test_batch_hand_example():
    # 3 h/day x 0.5/h x 30 days x 2 instances = 90
    assert cost.batch_cost(job_hours_per_day=3, price_per_instance_hour=0.5, days=30,
                           instances=2) == 90.0
    assert cost.batch_cost(job_hours_per_day=3, price_per_instance_hour=0.5, days=30) == 45.0


def test_invocations_per_instance_target_hand_example():
    # 60 / 0.25 s = 240 per minute, x 2 concurrent, x 0.5 headroom = 240
    assert cost.invocations_per_instance_target(latency_seconds=0.25, concurrency=2,
                                                utilization=0.5) == 240.0


def test_compare_orders_options_cheapest_first():
    out = cost.compare(
        realtime=dict(instances=2, price_per_instance_hour=0.25, hours=720),          # 360
        serverless=dict(requests=1000, seconds_per_request=0.5, memory_gb=2,
                        price_per_gb_second=0.25),                                    # 250
        batch=dict(job_hours_per_day=3, price_per_instance_hour=0.5, days=30))        # 45
    assert out["costs"] == {"realtime": 360.0, "serverless": 250.0, "batch": 45.0}
    assert out["cheapest_first"] == ["batch", "serverless", "realtime"]
    assert "not AWS quotes" in out["note"]


BAD = [0, -1, -0.5, True, False, math.nan, math.inf, "1", None]

CASES = [
    (cost.realtime_cost, dict(instances=1, price_per_instance_hour=1, hours=1), "hours"),
    (cost.realtime_cost, dict(instances=1, price_per_instance_hour=1, hours=1), "instances"),
    (cost.serverless_cost, dict(requests=1, seconds_per_request=1, memory_gb=1,
                                price_per_gb_second=1), "memory_gb"),
    (cost.serverless_cost, dict(requests=1, seconds_per_request=1, memory_gb=1,
                                price_per_gb_second=1, price_per_request=1), "price_per_request"),
    (cost.batch_cost, dict(job_hours_per_day=1, price_per_instance_hour=1, days=1), "days"),
    (cost.invocations_per_instance_target, dict(latency_seconds=1), "latency_seconds"),
]


# price_per_request=None is the documented "no per-request price" default, so it is valid.
@pytest.mark.parametrize("fn,kwargs,field,bad", [
    (fn, kw, field, bad) for fn, kw, field in CASES for bad in BAD
    if not (field == "price_per_request" and bad is None)])
def test_zero_negative_and_non_numeric_inputs_are_refused(fn, kwargs, field, bad):
    with pytest.raises(cost.CostInputError, match=field):
        fn(**{**kwargs, field: bad})


def test_utilization_above_one_is_refused():
    with pytest.raises(cost.CostInputError, match="<= 1"):
        cost.invocations_per_instance_target(latency_seconds=1, utilization=1.5)


def test_module_hard_codes_no_prices():
    import inspect
    source = inspect.getsource(cost)
    assert "us-east" not in source and "$" not in source
