import json
import os
import time
import boto3

lambda_client = boto3.client("lambda")

CONTROLLER_NAME = os.environ["CONTROLLER_FUNCTION_NAME"]
FUNCTION_ID = os.environ.get(
    "TARGET_FUNCTION_ID", "coldstart-live-target"
)

_module_loaded_at = time.perf_counter()
_first_invocation = True


def lambda_handler(event, context):
    global _first_invocation
    event = event or {}

    if event.get("mode") == "warmup":
        _first_invocation = False
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "warmup_request_handled",
                "function_id": FUNCTION_ID,
                "note": (
                    "Warm-up invocation completed. AWS may not reuse "
                    "this environment for later requests."
                )
            })
        }

    handler_started_at = time.perf_counter()
    cold_start_observed = _first_invocation
    init_duration_ms = (
        (handler_started_at - _module_loaded_at) * 1000
        if cold_start_observed else None
    )
    _first_invocation = False

    work_ms = int(event.get("work_ms", 100))
    time.sleep(max(0, min(work_ms, 2000)) / 1000)

    duration_ms = (
        time.perf_counter() - handler_started_at
    ) * 1000

    telemetry = {
        "mode": "telemetry",
        "function_id": FUNCTION_ID,
        "duration_ms": round(duration_ms, 2),
        "cold_start": cold_start_observed
    }

    if init_duration_ms is not None:
        telemetry["init_duration_ms"] = round(
            init_duration_ms, 2
        )

    try:
        lambda_client.invoke(
            FunctionName=CONTROLLER_NAME,
            InvocationType="Event",
            Payload=json.dumps(telemetry).encode("utf-8")
        )
        telemetry_status = "submitted"
    except Exception as exc:
        print("Telemetry submission failed:", str(exc))
        telemetry_status = "failed"

    return {
        "statusCode": 200,
        "body": json.dumps({
            "function_id": FUNCTION_ID,
            "duration_ms": round(duration_ms, 2),
            "cold_start_observed": cold_start_observed,
            "telemetry_status": telemetry_status
        })
    }
