
"""Serverless cold-start predictor with ML-triggered warm-up."""

import json
import math
import os
import time
import datetime
from decimal import Decimal
import boto3
from collections import defaultdict, deque
import uuid

with open(
    os.path.join(os.path.dirname(__file__), "models.json"),
    encoding="utf-8"
) as f:
    MODELS = json.load(f)

FEATURES = MODELS["features"]
LEVELS = ("on_demand", "warm_pool", "provisioned_concurrency")

_COLD = True
_EVENTS = defaultdict(lambda: deque(maxlen=100))
_POLICY = {
    k: {
        "a": [[float(i == j) for j in range(5)] for i in range(5)],
        "b": [0.0] * 5
    }
    for k in LEVELS
}
_POLICY_UPDATES = 0
_PENDING = {}

WARMER_FUNCTION = os.environ.get(
    "WARMER_FUNCTION_NAME",
    "coldstart-ml-warmer"
)

lambda_client = boto3.client("lambda")
dynamodb = boto3.resource("dynamodb")
LIVE_STATE_TABLE = os.environ.get("LIVE_STATE_TABLE", "coldstart-live-state")
HISTORY_TABLE = os.environ.get(
    "HISTORY_TABLE",
    "coldstart-invocation-history"
)

def _live_table():
    """Resolve the configured DynamoDB table only when live state is used."""
    return dynamodb.Table(LIVE_STATE_TABLE)


def _record_telemetry(event):
    """Store live state and a separate history record for each observation."""
    fid = str(event.get("function_id") or event.get("function_name") or "")
    if not fid:
        raise ValueError("telemetry mode requires function_id or function_name")

    now = time.time()
    now_iso = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    old = _live_table().get_item(
        Key={"function_id": fid},
        ConsistentRead=True
    ).get("Item", {})

    recent = [
        float(t)
        for t in old.get("invocation_timestamps", [])
        if now - float(t) <= 300
    ]
    recent.append(now)

    duration = float(
        event.get("duration_ms", old.get("last_duration_ms", 0))
    )
    init_duration = float(event.get("init_duration_ms", 0))
    concurrency = int(
        event.get("concurrency", old.get("last_concurrency_observed", 1))
    )
    cold_start = bool(event.get("cold_start", False))

    # Update the latest state.
    live_item = {
        "function_id": fid,
        "last_invocation_epoch": Decimal(str(now)),
        "invocation_timestamps": [
            Decimal(str(t)) for t in recent[-300:]
        ],
        "invocations_last_5m": len(recent),
        "last_duration_ms": Decimal(str(duration)),
        "last_init_duration_ms": Decimal(str(init_duration)),
        "last_concurrency_observed": concurrency,
        "last_cold_start_observed": cold_start,
        "last_observation_epoch": Decimal(str(now)),
    }

    _live_table().put_item(Item=live_item)

    # Store a separate observation in the history table.
    observation_id = f"{int(now * 1000)}#{uuid.uuid4().hex}"

    history_item = {
        "function_id": fid,
        "observation_id": observation_id,
        "timestamp": now_iso,
        "timestamp_epoch": Decimal(str(now)),
        "duration_ms": Decimal(str(duration)),
        "init_duration_ms": Decimal(str(init_duration)),
        "concurrency": concurrency,
        "cold_start": cold_start,
        "invocations_last_5m": len(recent),
    }

    dynamodb.Table(HISTORY_TABLE).put_item(Item=history_item)

    return {
        "status": "telemetry_recorded",
        "function_id": fid,
        "observation_id": observation_id,
        "invocations_last_5m": len(recent),
        "last_invocation_epoch": now,
    }

def _load_live_state(fid):
    """Load persisted state. None means no target-function telemetry is available."""
    item = _live_table().get_item(Key={"function_id": fid}, ConsistentRead=True).get("Item")
    if not item:
        return None
    now = time.time()
    stamps = [float(t) for t in item.get("invocation_timestamps", []) if now - float(t) <= 300]
    last = float(item.get("last_invocation_epoch", now))
    return {
        "seconds_since_last_invocation": max(0.0, now - last),
        "invocations_last_5m": len(stamps),
        "concurrency": float(item.get("last_concurrency_observed", 1)),
        "last_duration_ms": float(item.get("last_duration_ms", 0)),
        "last_init_duration_ms": float(item.get("last_init_duration_ms", 0)),
        "last_cold_start_observed": bool(item.get("last_cold_start_observed", False)),
        "last_observation_epoch": float(item.get("last_observation_epoch", last)),
        # This is evidence from telemetry, NOT an exact count of reusable Lambda environments.
        "warm_availability_estimate": "unknown"
    }


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-min(x, 700))
        return 1 / (1 + z)
    z = math.exp(max(x, -700))
    return z / (1 + z)


def _tree_prob(tree, x):
    node = 0
    while tree["children_left"][node] != tree["children_right"][node]:
        f = tree["feature"][node]
        node = (
            tree["children_left"][node]
            if x[f] <= tree["threshold"][node]
            else tree["children_right"][node]
        )
    v = tree["value"][node][0]
    total = sum(v)
    return float(v[1] / total) if total else 0.0


def _predict(x, edge=False):
    if edge:
        m = MODELS["logistic_regression"]
        z = m["intercept"] + sum(
            ((x[i] - m["mean"][i]) / (m["scale"][i] or 1))
            * m["coef"][i]
            for i in range(len(x))
        )
        return _sigmoid(z), "logistic_regression"

    forest = MODELS["random_forest"]
    return (
        sum(_tree_prob(tree, x) for tree in forest) / len(forest),
        "random_forest"
    )


def _inv(matrix):
    n = len(matrix)
    a = [
        list(map(float, row)) +
        [float(i == j) for j in range(n)]
        for i, row in enumerate(matrix)
    ]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[pivot][i]) < 1e-12:
            raise ValueError("Singular policy matrix")

        a[i], a[pivot] = a[pivot], a[i]
        d = a[i][i]
        a[i] = [v / d for v in a[i]]

        for r in range(n):
            if r != i:
                q = a[r][i]
                a[r] = [
                    a[r][c] - q * a[i][c]
                    for c in range(2 * n)
                ]

    return [row[n:] for row in a]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _schedule(fid, p, gap, burst, package, memory,
              warm, provisioned, budget):
    x = [
        1.0,
        p,
        min(math.log1p(max(0, gap)) / 8, 1),
        min(max(burst, 0), 3) / 3,
        min(max(budget, 0), 1)
    ]

    eligible = ["on_demand"]

    if warm > 0 and budget >= 0.12:
        eligible.append("warm_pool")

    if provisioned > 0 and budget >= 0.42:
        eligible.append("provisioned_concurrency")

    scores = {}

    for level in eligible:
        inv = _inv(_POLICY[level]["a"])
        theta = [
            _dot(row, _POLICY[level]["b"])
            for row in inv
        ]
        uncertainty = math.sqrt(
            max(0, _dot(x, [_dot(row, x) for row in inv]))
        )

        delay = p * (0.25 + package / 500 + memory / 4096)

        utility = {
            "on_demand": -delay,
            "warm_pool": -delay * 0.48 - 0.12,
            "provisioned_concurrency": -delay * 0.10 - 0.42
        }[level]

        scores[level] = (
            _dot(theta, x) + 0.25 * uncertainty + utility
        )

    action = max(scores, key=scores.get)
    _PENDING[fid] = (action, x)

    return {
        "action": action,
        "scores": {k: round(v, 5) for k, v in scores.items()},
        "resource_constraints": {
            "warm_capacity": warm,
            "provisioned_capacity": provisioned,
            "budget_ratio": budget
        },
        "policy": (
            "LinUCB contextual bandit with latency/cost utility; "
            "state is per execution environment"
        )
    }


def _observe(fid, reward):
    global _POLICY_UPDATES

    if fid not in _PENDING:
        return {
            "error": (
                "No pending scheduler decision for function_id "
                "in this execution environment"
            )
        }

    action, x = _PENDING.pop(fid)
    state = _POLICY[action]
    reward = max(-2, min(2, float(reward)))

    for i in range(5):
        state["b"][i] += reward * x[i]
        for j in range(5):
            state["a"][i][j] += x[i] * x[j]

    _POLICY_UPDATES += 1

    return {
        "action": action,
        "reward": reward,
        "updated_decisions": _POLICY_UPDATES
    }


def _features(event):
    fid = str(event.get("function_id", "coldstart_ml_demo1"))

    runtime = event.get("runtime_code")
    if runtime is None:
        runtime = {
            "python": 0,
            "node": 1,
            "nodejs": 1,
            "java": 2,
            "dotnet": 3
        }.get(str(event.get("runtime_type", "python")).lower(), 0)

    gap = float(event.get(
        "seconds_since_last_invocation",
        event.get("time_since_last_invocation", 600)
    ))

    count = float(event.get(
        "invocations_last_5m",
        event.get("recent_invocation_count", 0)
    ))

    if (
        "traffic_rate" in event
        and "invocations_last_5m" not in event
        and "recent_invocation_count" not in event
    ):
        count = max(0, float(event["traffic_rate"]) * 300)

    burst = float(event.get(
        "burstiness",
        event.get("traffic_burst_indicator", 0)
    ))

    if (
        "history_size" in event
        and "active_warm_containers" not in event
    ):
        warm = (
            1 if gap < 600 and int(event["history_size"]) > 0
            else 0
        )
    else:
        warm = int(event.get("active_warm_containers", 0))

    values = {
        "seconds_since_last_invocation": gap,
        "invocations_last_5m": count,
        "concurrency": float(event.get("concurrency", 1)),
        "memory_mb": float(event.get("memory_mb", 128)),
        "package_mb": float(event.get("package_mb", 30)),
        "runtime_code": float(runtime),
        "hour_of_day": float(event.get(
            "hour_of_day",
            datetime.datetime.now(datetime.timezone.utc).hour
        )),
        "burstiness": burst,
        "active_warm_containers": float(warm)
    }

    return fid, [values[k] for k in FEATURES], values



def _cost_aware_decision(probability, event, warm_capacity):
    """Choose warm_pool vs on_demand using ML risk and configured costs.

    Costs must use the same unit (for example, USD per request). Values may
    be supplied in the event for experiments or configured as Lambda env vars.
    No probability threshold is hard-coded.
    """
    def configured(name):
        raw = event.get(name.lower())
        if raw is None or raw == "":
            raw = os.environ.get(name)
        if raw is None or raw == "":
            raise ValueError(
                f"Missing {name}: supply event field '{name.lower()}' "
                f"or configure Lambda environment variable {name}"
            )
        return float(raw)

    cold_start_cost = configured("COLD_START_COST")
    warm_action_cost = configured("WARM_ACTION_COST")
    warm_effectiveness = configured("WARM_EFFECTIVENESS")
    budget_remaining = configured("BUDGET_REMAINING")

    if cold_start_cost < 0 or warm_action_cost < 0:
        raise ValueError("Costs must be non-negative")
    if not 0 <= warm_effectiveness <= 1:
        raise ValueError("WARM_EFFECTIVENESS must be between 0 and 1")
    if budget_remaining < 0:
        raise ValueError("BUDGET_REMAINING must be non-negative")

    preference = str(event.get("user_preference", "auto")).strip().lower()
    if preference not in {"auto", "warm", "cost_saving"}:
        raise ValueError("user_preference must be auto, warm, or cost_saving")

    # Expected costs are comparable because all configured costs use one unit.
    on_demand_expected_cost = probability * cold_start_cost
    warm_expected_cost = (
        warm_action_cost
        + probability * (1.0 - warm_effectiveness) * cold_start_cost
    )

    warm_is_feasible = (
        warm_capacity > 0 and budget_remaining >= warm_action_cost
    )

    if preference == "warm":
        if not warm_is_feasible:
            action = "on_demand"
            reason = "user_preferred_warm_but_capacity_or_budget_unavailable"
        else:
            action = "warm_pool"
            reason = "user_preference_warm_within_constraints"
    elif preference == "cost_saving":
        action = "on_demand"
        reason = "user_preference_cost_saving"
    elif warm_is_feasible and warm_expected_cost < on_demand_expected_cost:
        action = "warm_pool"
        reason = "ml_probability_and_expected_cost_favor_warming"
    else:
        action = "on_demand"
        reason = "expected_cost_does_not_justify_warming"

    return {
        "action": action,
        "decision_reason": reason,
        "user_preference": preference,
        "expected_costs": {
            "on_demand": round(on_demand_expected_cost, 8),
            "warm_pool": round(warm_expected_cost, 8),
        },
        "cost_unit": event.get("cost_unit", os.environ.get("COST_UNIT", "configured_unit")),
        "constraints": {
            "warm_capacity": warm_capacity,
            "budget_remaining": budget_remaining,
            "warm_action_cost": warm_action_cost,
            "warm_feasible": warm_is_feasible,
        },
        "decision_method": "Random Forest probability + expected-cost comparison; no fixed probability threshold",
    }

def lambda_handler(event, context):
    global _COLD

    event = event or {}
    started = time.perf_counter()
    cold = _COLD
    _COLD = False

    mode = event.get("mode", "predict")

    # Warm-up requests exit here; they cannot trigger another warm-up.
    if mode == "warmup":
        result = {
            "status": "warmup",
            "function_id": str(event.get("function_id", "warmup")),
            "observed_lambda_environment_cold_start": cold,
            "runtime_duration_ms": round(
                (time.perf_counter() - started) * 1000, 3
            )
        }

        print(json.dumps({
            "event": "coldstart_warmup",
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "result": result
        }))

        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(result)
        }

    if mode == "telemetry":
        try:
            telemetry = _record_telemetry(event)
            return {"statusCode": 200, "headers": {"content-type": "application/json"},
                    "body": json.dumps(telemetry)}
        except Exception as exc:
            return {"statusCode": 500, "headers": {"content-type": "application/json"},
                    "body": json.dumps({"status": "telemetry_error", "error": str(exc)})}

    if mode == "feedback":
        feedback = _observe(
            str(event.get("function_id", "coldstart_ml_demo1")),
            float(event.get("reward", 0))
        )

        return {
            "statusCode": 200 if "error" not in feedback else 400,
            "body": json.dumps({"mode": "feedback", **feedback})
        }

    fid, x, features = _features(event)

    if mode == "event":
        history = _EVENTS[fid]
        now = time.time()
        gap = now - history[-1] if history else features[
            "seconds_since_last_invocation"
        ]
        history.append(now)

        features["seconds_since_last_invocation"] = max(0, gap)
        features["invocations_last_5m"] = sum(
            1 for t in history if now - t <= 300
        )
        x = [features[k] for k in FEATURES]

    live_state = None
    if bool(event.get("use_live_state", True)):
        try:
            live_state = _load_live_state(fid)
        except Exception as exc:
            # Don't silently pretend live data was used; report the failure below.
            live_state = {"state_error": str(exc)}
        if live_state and "state_error" not in live_state:
            features["seconds_since_last_invocation"] = live_state["seconds_since_last_invocation"]
            features["invocations_last_5m"] = live_state["invocations_last_5m"]
            features["concurrency"] = live_state["concurrency"]
            x = [features[k] for k in FEATURES]

    edge = bool(event.get("edge", False))
    probability, model = _predict(x, edge)

    # Preserve the old scheduler output for diagnostics, but make the actual
    # warm/on-demand choice with the ML probability and explicit cost inputs.
    rec = _schedule(
        fid,
        probability,
        features["seconds_since_last_invocation"],
        features["burstiness"],
        features["package_mb"],
        features["memory_mb"],
        int(event.get("warm_capacity", 2)),
        int(event.get("provisioned_capacity", 1)),
        float(event.get("budget_ratio", 0.8))
    )
    try:
        decision = _cost_aware_decision(
            probability, event, int(event.get("warm_capacity", 2))
        )
    except (TypeError, ValueError) as exc:
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"status": "configuration_error", "error": str(exc)})
        }
    rec.update(decision)
    # Ensure feedback, if used, is associated with the action actually taken.
    _PENDING[fid] = (rec["action"], [
        1.0,
        probability,
        min(math.log1p(max(0, features["seconds_since_last_invocation"])) / 8, 1),
        min(max(features["burstiness"], 0), 3) / 3,
        min(max(float(event.get("budget_ratio", 0.8)), 0), 1)
    ])

    warmup_triggered = False
    warmup_error = None

    # Invoke the warmer only when the scheduler chooses warm_pool.
    if rec["action"] == "warm_pool":
        try:
            response = lambda_client.invoke(
                FunctionName=WARMER_FUNCTION,
                InvocationType="Event",
                Payload=json.dumps({
                    "mode": "warmup",
                    "function_id": fid,
                    "reason": rec.get("decision_reason", "model_recommendation"),
                    "cold_start_probability": round(probability, 6)
                }).encode("utf-8")
            )

            warmup_triggered = response.get("StatusCode") == 202

            if not warmup_triggered:
                warmup_error = (
                    "Warmer invocation not accepted: "
                    + str(response.get("StatusCode"))
                )

        except Exception as exc:
            warmup_error = str(exc)

    duration_ms = (time.perf_counter() - started) * 1000

    result = {
        "status": ("warmup_requested" if warmup_triggered else ("warmup_failed" if rec["action"] == "warm_pool" else "ok")),
        "function_id": fid,
        "cold_start_probability": round(probability, 6),
        "model": model,
        "algorithm": (
            "Random Forest (primary)"
            if model == "random_forest"
            else "Logistic Regression (lightweight/edge)"
        ),
        "recommendation": rec,
        "warmup_triggered": warmup_triggered,
        "warmup_error": warmup_error,
        "observed_lambda_environment_cold_start": cold,
        "runtime_duration_ms": round(duration_ms, 3),
        "initialization_cost_ms": event.get("initialization_cost_ms"),
        "features": features,
        "live_state": live_state,
        "data_source": ("dynamodb_live_telemetry" if live_state and "state_error" not in live_state else "event_fallback_or_live_state_unavailable"),
        "note": (
            "warmup_triggered indicates that AWS accepted the "
            "asynchronous warmer invocation. It does not guarantee "
            "reuse of that environment for a later real request."
        )
    }

    print(json.dumps({
        "event": "coldstart_prediction",
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "result": result
    }, separators=(",", ":")))

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(result)
    }
