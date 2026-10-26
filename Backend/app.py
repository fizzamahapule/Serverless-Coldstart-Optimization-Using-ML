import json
import os
from datetime import datetime, timezone
import time

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

st.set_page_config(page_title="Serverless Cold-Start Controller", page_icon="⚡", layout="wide")
st.title("⚡ Serverless Cold-Start Controller")
st.caption("Live AWS Lambda inventory · click Test to send that function's context to your ML controller")

st.divider()
st.header("Live AWS Telemetry — all deployed Lambda functions")
st.caption(
    "The inventory comes from AWS Lambda. CloudWatch metrics are best-effort and may be delayed; "
    "last invocation and cold-start details are shown only when recorded in DynamoDB."
)

LIVE_STATE_TABLE = os.getenv("LIVE_STATE_TABLE", "coldstart-live-state")

if st.button("Refresh live telemetry", key="refresh_live_telemetry"):
    st.cache_data.clear()

@st.cache_data(ttl=5)
def load_live_state(region, table_name):
    """Load recorded observations; failure is handled separately from AWS inventory."""
    session = boto3.Session(region_name=region)
    table = session.resource("dynamodb").Table(table_name)
    items = []
    response = table.scan(Limit=200)
    items.extend(response.get("Items", []))
    while response.get("LastEvaluatedKey") and len(items) < 1000:
        response = table.scan(
            ExclusiveStartKey=response["LastEvaluatedKey"],
            Limit=min(200, 1000 - len(items)),
        )
        items.extend(response.get("Items", []))
    return items

@st.cache_data(ttl=60)
def load_lambda_inventory(region):
    """Discover every Lambda function visible to this AWS identity in this region."""
    client = boto3.Session(region_name=region).client("lambda")
    functions = []
    paginator = client.get_paginator("list_functions")
    for page in paginator.paginate():
        for f in page.get("Functions", []):
            functions.append({
                "Function": f["FunctionName"],
                "Runtime": f.get("Runtime") or "unknown",
                "Memory (MB)": int(f.get("MemorySize", 128)),
                "Package (MB)": round(float(f.get("CodeSize", 0)) / (1024 * 1024), 3),
                "Timeout (s)": int(f.get("Timeout", 3)),
            })
    return sorted(functions, key=lambda x: x["Function"].lower())

@st.cache_data(ttl=60)
def load_function_metrics(region, function_name):
    """Return CloudWatch values only when datapoints exist; do not present missing data as zero."""
    from datetime import timedelta
    cw = boto3.Session(region_name=region).client("cloudwatch")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=1)
    out = {
        "Invocations (last 5m)": None,
        "Invocations (last 1h)": None,
        "Avg duration (ms, 1h)": None,
        "Errors (last 1h)": None,
        "CloudWatch status": "No datapoints yet",
    }
    try:
        def query(metric, minutes, statistic="Sum"):
            response = cw.get_metric_statistics(
                Namespace="AWS/Lambda",
                MetricName=metric,
                Dimensions=[{"Name": "FunctionName", "Value": function_name}],
                StartTime=end - timedelta(minutes=minutes),
                EndTime=end,
                Period=300,
                Statistics=[statistic],
            )
            return response.get("Datapoints", [])

        inv5 = query("Invocations", 5, "Sum")
        inv60 = query("Invocations", 60, "Sum")
        duration = query("Duration", 60, "Average")
        errors = query("Errors", 60, "Sum")
        out["Invocations (last 5m)"] = int(round(sum(float(p["Sum"]) for p in inv5))) if inv5 else None
        out["Invocations (last 1h)"] = int(round(sum(float(p["Sum"]) for p in inv60))) if inv60 else None
        out["Avg duration (ms, 1h)"] = round(sum(float(p["Average"]) for p in duration) / len(duration), 2) if duration else None
        out["Errors (last 1h)"] = int(round(sum(float(p["Sum"]) for p in errors))) if errors else None
        if any((inv5, inv60, duration, errors)):
            out["CloudWatch status"] = "Available (may be delayed)"
    except (BotoCoreError, ClientError) as exc:
        out["CloudWatch status"] = f"Unavailable: {type(exc).__name__}"
    return out

live_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1"
try:
    inventory_rows = load_lambda_inventory(live_region)
    try:
        live_items = load_live_state(live_region, LIVE_STATE_TABLE)
        state_status = f"DynamoDB observations loaded: {len(live_items)} record(s)"
    except Exception as state_exc:
        live_items = []
        state_status = f"DynamoDB observations unavailable ({type(state_exc).__name__}); AWS inventory and CloudWatch metrics still shown"

    state_by_name = {str(item.get("function_id")): item for item in live_items if item.get("function_id")}
    telemetry_rows = []
    metrics_by_name = {}
    now_epoch = datetime.now(timezone.utc).timestamp()
    for fn in inventory_rows:
        name = fn["Function"]
        metrics = load_function_metrics(live_region, name)
        metrics_by_name[name] = metrics
        state = state_by_name.get(name, {})
        last_invocation = float(state.get("last_invocation_epoch", 0) or 0)
        last_observation = float(state.get("last_observation_epoch", 0) or 0)
        telemetry_rows.append({
            **fn,
            **metrics,
            "Last invocation (UTC)": datetime.fromtimestamp(last_invocation, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if last_invocation > 0 else "Unknown (no recorded timestamp)",
            "Seconds since recorded invocation": round(max(0, now_epoch-last_invocation), 1) if last_invocation > 0 else None,
            "Last cold start observed": (
              str(state.get("last_cold_start_observed"))
              if isinstance(state.get("last_cold_start_observed"), bool)
               else "Unknown"
               ),
            "Last init duration (ms)": state.get("last_init_duration_ms"),
            "Telemetry record": "Yes" if state else "No — only AWS inventory/metrics",
            "Observation age (s)": round(max(0, now_epoch-last_observation), 1) if last_observation > 0 else None,
        })

    st.caption(f"Region: {live_region} · {len(inventory_rows)} real Lambda function(s) discovered · {state_status}")
    if telemetry_rows:
        df_live = pd.DataFrame(telemetry_rows).sort_values("Function")
        st.metric("Real Lambda functions discovered", len(df_live))
        st.dataframe(df_live, use_container_width=True, hide_index=True)
        st.subheader("Selected function telemetry")
        selected_function = st.selectbox("Choose a real AWS Lambda function", df_live["Function"].tolist(), key="live_telemetry_function")
        selected_row = next(row for row in telemetry_rows if row["Function"] == selected_function)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invocations (last 5m)", selected_row["Invocations (last 5m)"] if selected_row["Invocations (last 5m)"] is not None else "No datapoint")
        c2.metric("Invocations (last 1h)", selected_row["Invocations (last 1h)"] if selected_row["Invocations (last 1h)"] is not None else "No datapoint")
        c3.metric("Average duration (1h)", f"{selected_row['Avg duration (ms, 1h)']} ms" if selected_row["Avg duration (ms, 1h)"] is not None else "No datapoint")
        c4.metric("Last init duration", f"{float(selected_row['Last init duration (ms)']):.2f} ms" if selected_row["Last init duration (ms)"] is not None else "Not recorded")
        st.write("**Last cold-start observation:**", selected_row["Last cold start observed"])
        st.write("**Last invocation timestamp:**", selected_row["Last invocation (UTC)"])
        st.caption("CloudWatch metrics are not instantaneous and may be delayed. Last invocation and observed cold-start fields require your telemetry writer to record them in DynamoDB. A function without a DynamoDB record is still included in the AWS inventory.")
    else:
        st.info(f"No Lambda functions were returned in {live_region}. Check AWS credentials, region, and lambda:ListFunctions permission.")
except (BotoCoreError, ClientError, Exception) as exc:
    st.error(f"Could not load AWS Lambda inventory/telemetry: {type(exc).__name__}: {exc}")
    st.info("Check AWS credentials, AWS_REGION, lambda:ListFunctions, cloudwatch:GetMetricStatistics, and DynamoDB permissions. The inventory is regional.")


REGION = (
    os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "ap-south-1"
)
CONTROLLER_NAME = os.getenv("CONTROLLER_FUNCTION_NAME", "").strip()
COLD_START_COST = float(os.getenv("COLD_START_COST", "0.10"))
WARM_ACTION_COST = float(os.getenv("WARM_ACTION_COST", "0.01"))
WARM_EFFECTIVENESS = float(os.getenv("WARM_EFFECTIVENESS", "0.80"))
BUDGET_REMAINING = float(os.getenv("BUDGET_REMAINING", "1.00"))
COST_UNIT = os.getenv("COST_UNIT", "configured_unit")

@st.cache_resource
def clients(region):
    session = boto3.Session(region_name=region)
    return session.client("lambda"), session.client("cloudwatch")

def list_functions(client):
    result = []
    paginator = client.get_paginator("list_functions")
    for page in paginator.paginate():
        for f in page.get("Functions", []):
            result.append({
                "name": f["FunctionName"],
                "arn": f["FunctionArn"],
                "runtime": f.get("Runtime") or "unknown",
                "memory_mb": int(f.get("MemorySize", 128)),
                "package_mb": round(float(f.get("CodeSize", 0)) / (1024 * 1024), 3),
                "timeout": int(f.get("Timeout", 3)),
                "last_modified": f.get("LastModified", ""),
            })
    return sorted(result, key=lambda x: x["name"].lower())

def get_metrics(cw, function_name):
    """Collect available CloudWatch observations without silently treating missing datapoints as zero."""
    from datetime import timedelta
    end = datetime.now(timezone.utc)
    out = {
        "invocations_last_5m": None,
        "invocations_last_1h": None,
        "concurrency": None,
        "average_duration_ms": None,
        "errors_last_1h": None,
        "latest_invocation_metric_epoch": None,
        "metrics_status": "no datapoints yet",
    }
    try:
        def datapoints(metric, minutes, statistic="Sum"):
            response = cw.get_metric_statistics(
                Namespace="AWS/Lambda", MetricName=metric,
                Dimensions=[{"Name": "FunctionName", "Value": function_name}],
                StartTime=end-timedelta(minutes=minutes), EndTime=end,
                Period=300, Statistics=[statistic])
            return response.get("Datapoints", [])
        inv5 = datapoints("Invocations", 5)
        inv60 = datapoints("Invocations", 60)
        conc = datapoints("ConcurrentExecutions", 60, "Maximum")
        duration = datapoints("Duration", 60, "Average")
        errors = datapoints("Errors", 60)
        out["invocations_last_5m"] = int(round(sum(float(p["Sum"]) for p in inv5))) if inv5 else None
        out["invocations_last_1h"] = int(round(sum(float(p["Sum"]) for p in inv60))) if inv60 else None
        out["concurrency"] = max((float(p["Maximum"]) for p in conc), default=None)
        out["average_duration_ms"] = round(sum(float(p["Average"]) for p in duration)/len(duration), 2) if duration else None
        out["errors_last_1h"] = int(round(sum(float(p["Sum"]) for p in errors))) if errors else None
        if inv60:
            out["latest_invocation_metric_epoch"] = max(p["Timestamp"].timestamp() for p in inv60)
        if any((inv5, inv60, conc, duration, errors)):
            out["metrics_status"] = "available (CloudWatch may be delayed)"
    except (BotoCoreError, ClientError) as e:
        out["metrics_status"] = f"unavailable ({type(e).__name__}: {e})"
    return out

def runtime_code(runtime):
    r = (runtime or "").lower()
    if "node" in r: return 1
    if "java" in r: return 2
    if "dotnet" in r: return 3
    return 0

def unpack_lambda_payload(raw_text):
    """Unwrap Lambda JSON and API-style body wrappers without assuming one shape."""
    try:
        value = json.loads(raw_text) if raw_text else {}
    except (TypeError, json.JSONDecodeError):
        return {"raw_response": raw_text}
    for _ in range(3):
        if isinstance(value, dict) and isinstance(value.get("body"), str):
            try:
                value = json.loads(value["body"])
                continue
            except json.JSONDecodeError:
                value = {**value, "body_text": value["body"]}
        break
    return value if isinstance(value, dict) else {"value": value}


def core_scenario_payload(scenario, target, selected_preference):
    """Explicit workload/event mapping used only by the live dashboard test runner."""
    common = {
        "mode": "predict",
        "function_id": target["name"],
        "runtime_type": target["runtime"],
        "runtime_code": runtime_code(target["runtime"]),
        "hour_of_day": datetime.now(timezone.utc).hour,
        "active_warm_containers": 0,
        "provisioned_capacity": 0,
        "budget_ratio": 0.8,
        "cold_start_cost": COLD_START_COST,
        "warm_action_cost": WARM_ACTION_COST,
        "warm_effectiveness": WARM_EFFECTIVENESS,
        "budget_remaining": BUDGET_REMAINING,
        "cost_unit": COST_UNIT,
        # Test inputs must not be overwritten by old DynamoDB state.
        "use_live_state": False,
    }
    profiles = {
        "warm": {
            "label": "Warm",
            "preference": "warm",
            "expected_action": "warm_pool",
            "warm_capacity": 2,
            "seconds_since_last_invocation": 900,
            "invocations_last_5m": 0,
            "concurrency": 1,
            "memory_mb": target["memory_mb"],
            "package_mb": target["package_mb"],
            "burstiness": 0.0,
        },
        "cold_on_demand": {
            "label": "Cold on-demand",
            "preference": "cost_saving",
            "expected_action": "on_demand",
            "warm_capacity": 0,
            "seconds_since_last_invocation": 3600,
            "invocations_last_5m": 0,
            "concurrency": 1,
            "memory_mb": target["memory_mb"],
            "package_mb": target["package_mb"],
            "burstiness": 0.0,
        },
        "user_preferred_warm": {
            "label": "User-preferred warm",
            "preference": "warm",
            "expected_action": "warm_pool",
            "warm_capacity": 2,
            "seconds_since_last_invocation": 300,
            "invocations_last_5m": 3,
            "concurrency": 2,
            "memory_mb": target["memory_mb"],
            "package_mb": target["package_mb"],
            "burstiness": 0.5,
        },
        "light": {
            "label": "Light script",
            # This profile explicitly requests cost-saving on-demand behavior;
            # the low-load feature values are still sent to the ML predictor.
            "preference": "cost_saving",
            "expected_action": "on_demand",
            "warm_capacity": 2,
            "seconds_since_last_invocation": 1200,
            "invocations_last_5m": 1,
            "concurrency": 1,
            "memory_mb": min(128, target["memory_mb"]),
            "package_mb": min(1.0, target["package_mb"]),
            "burstiness": 0.0,
        },
        "heavy": {
            "label": "Heavy script",
            # Preserve the user's chosen preference (default: auto). The
            # test fails honestly if the deployed policy does not choose warm.
            "preference": selected_preference,
            "expected_action": "warm_pool",
            "warm_capacity": 2,
            "seconds_since_last_invocation": 2,
            "invocations_last_5m": 120,
            "concurrency": 20,
            "memory_mb": max(1024, target["memory_mb"]),
            "package_mb": max(50.0, target["package_mb"]),
            "burstiness": 3.0,
        },
    }
    profile = profiles[scenario]
    event = {
        **common,
        **{k: v for k, v in profile.items() if k not in {"label", "expected_action"}},
        "user_preference": profile["preference"],
    }
    return event, profile


def run_core_scenario(lambda_client, target, scenario, selected_preference):
    """Invoke the deployed controller and then invoke the real target once."""
    if not CONTROLLER_NAME:
        raise RuntimeError("Set CONTROLLER_FUNCTION_NAME to the deployed controller Lambda name.")
    event, profile = core_scenario_payload(scenario, target, selected_preference)
    controller_response = lambda_client.invoke(
        FunctionName=CONTROLLER_NAME,
        InvocationType="RequestResponse",
        Payload=json.dumps(event).encode("utf-8"),
    )
    controller_raw = controller_response["Payload"].read().decode("utf-8", errors="replace")
    controller_result = unpack_lambda_payload(controller_raw)
    recommendation = controller_result.get("recommendation", {}) if isinstance(controller_result, dict) else {}
    actual_action = recommendation.get("action", controller_result.get("action", "unknown"))
    warmup_triggered = bool(controller_result.get("warmup_triggered", False))

    # Give the asynchronous warmer a moment to start before the normal request.
    if warmup_triggered:
        time.sleep(2)

    target_event = {
        "mode": "normal_invocation",
        "function_id": target["name"],
        "test_scenario": scenario,
        "workload_class": "heavy" if scenario == "heavy" else "light" if scenario == "light" else "standard",
    }
    target_response = lambda_client.invoke(
        FunctionName=target["name"],
        InvocationType="RequestResponse",
        Payload=json.dumps(target_event).encode("utf-8"),
    )
    target_raw = target_response["Payload"].read().decode("utf-8", errors="replace")
    target_result = unpack_lambda_payload(target_raw)

    expected_action = profile["expected_action"]
    action_pass = actual_action == expected_action
    expected_warmup = expected_action == "warm_pool"
    warmup_pass = warmup_triggered == expected_warmup
    target_pass = target_response.get("StatusCode") == 200 and not target_response.get("FunctionError")
    passed = action_pass and warmup_pass and target_pass
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scenario": profile["label"],
        "scenario_key": scenario,
        "target_function": target["name"],
        "preference_sent": profile["preference"],
        "expected_action": expected_action,
        "actual_action": actual_action,
        "action_pass": action_pass,
        "warmup_expected": expected_warmup,
        "warmup_triggered": warmup_triggered,
        "warmup_pass": warmup_pass,
        "controller_status_code": controller_response.get("StatusCode"),
        "controller_request_id": controller_response.get("ResponseMetadata", {}).get("RequestId"),
        "cold_start_probability": controller_result.get("cold_start_probability"),
        "target_status_code": target_response.get("StatusCode"),
        "target_function_error": target_response.get("FunctionError"),
        "target_cold_start_observed": target_result.get("cold_start_observed", target_result.get("observed_lambda_environment_cold_start")),
        "target_environment_id": target_result.get("environment_id"),
        "target_duration_ms": target_result.get("duration_ms", target_result.get("runtime_duration_ms")),
        "target_pass": target_pass,
        "pass": passed,
        "failure_reason": "; ".join(
            reason for condition, reason in [
                (not action_pass, f"expected {expected_action}, got {actual_action}"),
                (not warmup_pass, f"warmup expected={expected_warmup}, got={warmup_triggered}"),
                (not target_pass, "target invocation failed"),
            ] if condition
        ),
        "controller_result": controller_result,
        "target_result": target_result,
        "payload": event,
    }



def invoke_controller(lambda_client, function, cw, allow_warm):
    if not CONTROLLER_NAME:
        raise RuntimeError("Set CONTROLLER_FUNCTION_NAME to the deployed controller Lambda name.")
    metrics = get_metrics(cw, function["name"])
    # Prefer the telemetry writer's exact recorded invocation time. If unavailable,
    # CloudWatch's latest Invocations datapoint is only a coarse 5-minute-bucket proxy.
    state_items = []
    state_status = "unavailable"
    try:
        state_items = load_live_state(REGION, LIVE_STATE_TABLE)
        state_status = "available"
    except Exception:
        pass
    state = next((item for item in state_items if item.get("function_id") == function["name"]), {})
    last_epoch = float(state.get("last_invocation_epoch", 0) or 0)
    seconds_source = "DynamoDB recorded invocation timestamp"
    if last_epoch <= 0 and metrics.get("latest_invocation_metric_epoch"):
        last_epoch = float(metrics["latest_invocation_metric_epoch"])
        seconds_source = "approximate CloudWatch 5-minute metric bucket"
    if last_epoch > 0:
        seconds_since_last = max(0.0, datetime.now(timezone.utc).timestamp() - last_epoch)
    else:
        # Numeric fallback required by the current model interface; explicitly disclose it.
        seconds_since_last = 3600.0
        seconds_source = "fallback 3600s because no invocation timestamp was available"
    inv5 = metrics.get("invocations_last_5m")
    inv60 = metrics.get("invocations_last_1h")
    if inv5 is not None and inv60 is not None:
        baseline_5m = inv60 / 12.0
        burstiness = round(inv5 / max(baseline_5m, 1.0), 3)
    else:
        burstiness = None
    event = {
        "mode": "predict",
        "function_id": function["name"],
        "runtime_type": function["runtime"],
        "runtime_code": runtime_code(function["runtime"]),
        "seconds_since_last_invocation": seconds_since_last,
        "seconds_since_last_invocation_source": seconds_source,
        "invocations_last_5m": inv5 if inv5 is not None else 0,
        "invocations_last_1h": inv60,
        "concurrency": metrics["concurrency"] if metrics["concurrency"] is not None else 1.0,
        "average_duration_ms": metrics["average_duration_ms"],
        "errors_last_1h": metrics["errors_last_1h"],
        "burstiness": burstiness if burstiness is not None else 0.0,
        "burstiness_source": "derived from CloudWatch 5m and 1h invocation counts" if burstiness is not None else "fallback 0 because invocation metrics were unavailable",
        "active_warm_containers": 0,
        "active_warm_containers_source": "not directly observable from standard Lambda metrics; placeholder value",
        "warm_capacity": 2 if allow_warm else 0,
        "provisioned_capacity": 0,
        "budget_ratio": 0.8,
        "cold_start_cost": COLD_START_COST,
        "warm_action_cost": WARM_ACTION_COST,
        "warm_effectiveness": WARM_EFFECTIVENESS,
        "budget_remaining": BUDGET_REMAINING,
        "cost_unit": COST_UNIT,
        "user_preference": "auto",
        "use_live_state": True
    }
    response = lambda_client.invoke(
        FunctionName=CONTROLLER_NAME,
        InvocationType="RequestResponse",
        Payload=json.dumps(event).encode("utf-8")
    )
    raw = response["Payload"].read().decode("utf-8", errors="replace")
    try:
        outer = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        outer = {"raw_response": raw}
    body = outer.get("body", outer)
    if isinstance(body, str):
        try: body = json.loads(body)
        except json.JSONDecodeError: body = {"raw_body": body}
    return {
        "controller": CONTROLLER_NAME,
        "http_status": response.get("StatusCode"),
        "function": function["name"],
        "metrics_status": metrics["metrics_status"],
        "result": body,
        "executed_version": response.get("ExecutedVersion", "$LATEST"),
        "request_id": response.get("ResponseMetadata", {}).get("RequestId")
    }

with st.sidebar:
    st.subheader("AWS connection")
    st.write("Region:", REGION or "AWS default region")
    st.write("Controller:", CONTROLLER_NAME or "Not configured")
    st.caption("Credentials are read from the normal boto3/AWS credential chain; they are not stored in this app.")
    allow_warm = st.checkbox(
        "Allow real warm-up requests",
        value=False,
        help="OFF is safe mode: warm_capacity=0 prevents the controller from requesting the warmer. ON may trigger the configured warmer and incur charges."
    )
    st.caption("Cost assumptions passed to the controller")
    st.write(f"Cold-start cost: {COLD_START_COST:g} {COST_UNIT}")
    st.write(f"Warm-action cost: {WARM_ACTION_COST:g} {COST_UNIT}")
    st.write(f"Warm effectiveness assumption: {WARM_EFFECTIVENESS:g}")
    st.write(f"Remaining budget: {BUDGET_REMAINING:g} {COST_UNIT}")
    refresh = st.button("Refresh function list", use_container_width=True)

if "test_results" not in st.session_state:
    st.session_state.test_results = {}

if not CONTROLLER_NAME:
    st.warning("Configure CONTROLLER_FUNCTION_NAME before testing. The function list can still load if AWS credentials and region are configured.")

try:
    lambda_client, cw = clients(REGION)
    functions = list_functions(lambda_client)
    st.caption(f"Live inventory loaded at {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} · {len(functions)} function(s)")
    st.divider()
    st.header("🧪 Core Implementation Testing")
    st.caption("Runs the deployed controller and then invokes the selected real Lambda target. These are live AWS invocations and may incur charges.")
    if functions:
        preferred_target_index = next((i for i, f in enumerate(functions) if f["name"] == "coldstart-live-target"), 0)
        target_names = [f["name"] for f in functions]
        target_name = st.selectbox("Target Lambda", target_names, index=preferred_target_index, key="core_test_target")
        target_function = next(f for f in functions if f["name"] == target_name)
        c1, c2 = st.columns(2)
        with c1:
            scenario_label = st.selectbox(
                "Scenario to test",
                ["Heavy script", "Warm", "Cold on-demand", "User-preferred warm", "Light script"],
                index=0,
                key="core_test_scenario",
            )
        with c2:
            selected_preference_label = st.selectbox(
                "User preference (used by Heavy script when selected)",
                ["auto", "warm", "on-demand / cost-saving"],
                index=0,
                key="core_test_preference",
            )
        selected_preference = {"auto": "auto", "warm": "warm", "on-demand / cost-saving": "cost_saving"}[selected_preference_label]
        scenario_keys = {
            "Heavy script": "heavy",
            "Warm": "warm",
            "Cold on-demand": "cold_on_demand",
            "User-preferred warm": "user_preferred_warm",
            "Light script": "light",
        }
        st.info("Mapping: Warm → prefer warm; Cold on-demand → capacity 0 + cost-saving; User-preferred warm → prefer warm; Light script → low-load features + cost-saving; Heavy script → high-load features + selected preference (default auto). Live DynamoDB state is disabled for these tests so the test features are actually evaluated.")
        run_one, run_all = st.columns(2)
        if run_one.button("▶ Run selected scenario", type="primary", use_container_width=True, disabled=not bool(CONTROLLER_NAME)):
            key = scenario_keys[scenario_label]
            with st.spinner(f"Running {scenario_label}: controller + target Lambda..."):
                try:
                    outcome = run_core_scenario(lambda_client, target_function, key, selected_preference)
                    st.session_state.setdefault("core_test_results", []).insert(0, outcome)
                except Exception as exc:
                    st.session_state.setdefault("core_test_results", []).insert(0, {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "scenario": scenario_label, "scenario_key": key,
                        "target_function": target_name, "pass": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    })
        if run_all.button("▶ Run all five scenarios", use_container_width=True, disabled=not bool(CONTROLLER_NAME)):
            with st.spinner("Running five live AWS scenarios. This can take a little time..."):
                for key in ["warm", "cold_on_demand", "user_preferred_warm", "light", "heavy"]:
                    try:
                        outcome = run_core_scenario(lambda_client, target_function, key, selected_preference)
                    except Exception as exc:
                        outcome = {
                            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "scenario": key, "scenario_key": key,
                            "target_function": target_name, "pass": False,
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }
                    st.session_state.setdefault("core_test_results", []).insert(0, outcome)
        core_results = st.session_state.get("core_test_results", [])
        if core_results:
            st.subheader("Test results")
            summary_rows = [{
                "Time": r.get("timestamp"),
                "Scenario": r.get("scenario"),
                "Target": r.get("target_function"),
                "Preference": r.get("preference_sent", "—"),
                "Expected action": r.get("expected_action", "—"),
                "Actual action": r.get("actual_action", "—"),
                "Warm-up accepted": r.get("warmup_triggered", "—"),
                "Target status": r.get("target_status_code", "—"),
                "Cold start observed": r.get("target_cold_start_observed", "—"),
                "Duration ms": r.get("target_duration_ms", "—"),
                "Result": "PASS" if r.get("pass") else "FAIL",
                "Details": r.get("failure_reason", ""),
            } for r in core_results]
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            csv_bytes = pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv_bytes, file_name="core_implementation_test_results.csv", mime="text/csv")
            latest = core_results[0]
            with st.expander("Latest test evidence (full controller + target responses)", expanded=False):
                st.json({k: v for k, v in latest.items() if k not in {"payload"}})
                if latest.get("payload"):
                    st.caption("Exact controller payload sent")
                    st.json(latest["payload"])
        if not functions:
           st.info(
            "AWS returned no Lambda functions in this region. "
            "Check AWS_REGION and IAM permissions."
           )

except (BotoCoreError, ClientError, Exception) as exc:
    st.error(
        f"Could not load AWS Lambda inventory: "
        f"{type(exc).__name__}: {exc}"
    )
    st.info(
        "Check AWS credentials, AWS_REGION, "
        "and lambda:ListFunctions permission."
    )

st.divider()
st.caption(
    "Research prototype. Inventory is live AWS data; model predictions "
    "are estimates. A warm-up request is not proof of environment reuse "
    "or measured latency savings."
)