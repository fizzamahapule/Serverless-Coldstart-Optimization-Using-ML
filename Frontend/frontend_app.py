import os, json, time, threading
from datetime import datetime, timezone
import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

st.set_page_config(page_title="Serverless Cold Start Optimization", page_icon="⚡", layout="wide")
st.markdown('''<style>
.stApp{background:#0b1220;color:#e5edf9}
[data-testid="stSidebar"]{background:#111c30}
.hero{padding:1.4rem 1.6rem;border:1px solid #263a56;border-radius:18px;background:linear-gradient(120deg,#172b47,#101827);margin-bottom:1rem}
.hero h1{margin:0;color:#f5f8ff;font-size:2rem}.hero p{color:#a8bbd6}
div[data-testid="stMetric"]{background:#111c30;border:1px solid #263a56;padding:1rem;border-radius:14px}
.cond{padding:1.1rem 1.4rem;border-radius:16px;font-size:1.6rem;font-weight:700;margin:.5rem 0 1rem 0}
.cond small{display:block;font-size:.95rem;font-weight:400;opacity:.9;margin-top:.25rem}
.cond-warm{background:#3a1d12;border:1px solid #ff8a3d;color:#ffb27a}
.cond-cold{background:#10283f;border:1px solid #4fb3ff;color:#9fd4ff}
.cond-err{background:#3b1520;border:1px solid #ff5c7a;color:#ff9fb1}
</style><div class="hero"><h1>⚡ Serverless Cold Start Optimization</h1>
<p>ML-driven Lambda warm-up simulation · Runs automatically every 10 seconds and recalculates the moment a parameter changes.</p></div>''',
            unsafe_allow_html=True)

if not hasattr(st, "fragment"):
    st.error("This app needs Streamlit 1.37 or newer for the 10-second auto-run (st.fragment). Please upgrade Streamlit.")
    st.stop()

REGION = os.getenv("AWS_REGION") or None
CONTROLLER = os.getenv("CONTROLLER_FUNCTION_NAME", "coldstart-ml-controller")
TARGET = os.getenv("TARGET_FUNCTION_NAME", "coldstart-live-target")
TARGET_ID = os.getenv("TARGET_FUNCTION_ID", "coldstart-live-target")
SNS_TOPIC = os.getenv("WARMUP_NOTIFICATION_TOPIC_ARN", "")
TOPIC_NAME = os.getenv("WARMUP_NOTIFICATION_TOPIC_NAME", "coldstart-warmup-notifications")
REFRESH_SECONDS = 10

SCENARIOS = {
    "Warm-up condition": {"user_preference": "auto", "seconds_since_last_invocation": 900, "invocations_last_5m": 18, "concurrency": 6, "burstiness": 2.8, "memory_mb": 512, "package_mb": 35, "warm_capacity": 2, "budget_remaining": 1.0, "budget_ratio": 0.9, "provisioned_capacity": 0},
    "Cold condition": {"user_preference": "auto", "seconds_since_last_invocation": 3600, "invocations_last_5m": 0, "concurrency": 1, "burstiness": 0.0, "memory_mb": 128, "package_mb": 20, "warm_capacity": 0, "budget_remaining": 0.0, "budget_ratio": 0.1, "provisioned_capacity": 0},
    "Heavy simulated traffic": {"user_preference": "auto", "seconds_since_last_invocation": 30, "invocations_last_5m": 120, "concurrency": 20, "burstiness": 3.0, "memory_mb": 1024, "package_mb": 45, "warm_capacity": 2, "budget_remaining": 1.0, "budget_ratio": 0.9, "provisioned_capacity": 0},
    "Light simulated traffic": {"user_preference": "cost_saving", "seconds_since_last_invocation": 1200, "invocations_last_5m": 2, "concurrency": 1, "burstiness": 0.2, "memory_mb": 256, "package_mb": 20, "warm_capacity": 1, "budget_remaining": 0.25, "budget_ratio": 0.3, "provisioned_capacity": 0},
    "User prefers warming": {"user_preference": "warm", "seconds_since_last_invocation": 180, "invocations_last_5m": 35, "concurrency": 8, "burstiness": 2.4, "memory_mb": 512, "package_mb": 30, "warm_capacity": 2, "budget_remaining": 1.0, "budget_ratio": 0.9, "provisioned_capacity": 0},
}
CUSTOM = "Custom (use sliders)"


# ------------------------------------------------- shared run history
# One store shared by every browser session of this app process, also appended to a file so it
# survives app restarts. Set RUN_HISTORY_FILE to a durable path if /tmp is not persistent.
HISTORY_FILE = os.getenv("RUN_HISTORY_FILE", "/tmp/coldstart_run_history.jsonl")


@st.cache_resource
def _store():
    rows = []
    try:
        with open(HISTORY_FILE) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return {"lock": threading.Lock(), "rows": rows}


def add_history(row):
    st_ = _store()
    with st_["lock"]:
        row["id"] = len(st_["rows"]) + 1
        st_["rows"].append(row)
        try:
            with open(HISTORY_FILE, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass


def get_history():
    st_ = _store()
    with st_["lock"]:
        return list(st_["rows"])


@st.cache_resource
def clients():
    return boto3.client("lambda", region_name=REGION), boto3.client("sns", region_name=REGION)


# ---------------------------------------------------------------- helpers
def decode(resp):
    raw = resp["Payload"].read()
    try:
        data = json.loads(raw.decode()) if raw else {}
    except Exception:
        return {"raw_response": raw.decode(errors="replace")}
    if isinstance(data, dict) and "body" in data:
        body = data["body"]
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                return data
        if isinstance(body, dict):
            body.setdefault("_http_status_code", data.get("statusCode"))
            return body
    return data if isinstance(data, dict) else {"result": data}


def invoke(client, name, payload):
    r = client.invoke(FunctionName=name, InvocationType="RequestResponse",
                      Payload=json.dumps(payload).encode())
    result = decode(r)
    if r.get("FunctionError"):
        raise RuntimeError(f"{name} Lambda error: {json.dumps(result, default=str)}")
    return result


def get_rec(result):
    rec = result.get("recommendation", {})
    if isinstance(rec, str):
        try:
            rec = json.loads(rec)
        except Exception:
            rec = {"action": rec}
    return rec if isinstance(rec, dict) else {}


def condition_of(probability, threshold):
    """State of the workload, derived from the ML cold-start probability:
    'cold' = cold start is likely (idle / low traffic), 'warm' = environment is active (cold start unlikely).
    This is separate from the controller's ACTION (whether a warm-up gets triggered)."""
    try:
        return "cold" if float(probability) >= float(threshold) else "warm"
    except (TypeError, ValueError):
        return "unknown"


def ensure_topic(sns):
    if SNS_TOPIC:
        return SNS_TOPIC
    return sns.create_topic(Name=TOPIC_NAME)["TopicArn"]  # idempotent


def ensure_subscription(sns, topic, address):
    """Returns 'confirmed' or 'pending'. Subscribes the address if not subscribed yet."""
    token = None
    while True:
        kw = {"TopicArn": topic}
        if token:
            kw["NextToken"] = token
        resp = sns.list_subscriptions_by_topic(**kw)
        for sub in resp.get("Subscriptions", []):
            if sub.get("Protocol") == "email" and sub.get("Endpoint", "").lower() == address.lower():
                return "pending" if sub.get("SubscriptionArn") == "PendingConfirmation" else "confirmed"
        token = resp.get("NextToken")
        if not token:
            break
    sns.subscribe(TopicArn=topic, Protocol="email", Endpoint=address)
    return "pending"


def email(sns, scenario, result, target, features, recipient, triggered, condition="unknown"):
    if not recipient:
        return False, "Enter a recipient email address to receive notifications."
    topic = ensure_topic(sns)
    state = ensure_subscription(sns, topic, recipient)
    if state == "pending":
        return False, (f"A subscription confirmation email was sent to {recipient}. Click 'Confirm subscription' "
                       "in that email (check spam) - SNS only delivers to confirmed addresses.")
    rec = get_rec(result)
    seen = target.get("warmup_received_in_environment")
    reuse = "VERIFIED" if seen is True else "NOT VERIFIED"
    body = f'''ColdStart warm-up event
Scenario: {scenario}
Timestamp UTC: {datetime.now(timezone.utc).isoformat()}
Controller: {CONTROLLER}
Target: {TARGET}
Simulated feature values (not actual request volume):
{json.dumps(features, indent=2)}
Condition: {condition.upper()}
Action: {rec.get("action", "not returned")}
Reason: {rec.get("decision_reason", "not returned")}
Cold-start probability: {result.get("cold_start_probability", "not returned")}
Model: {result.get("model", "not returned")}
Warm-up triggered: {triggered}
Target environment ID: {target.get("environment_id", "not returned")}
Application init duration ms: {target.get("application_init_duration_ms", "not returned")}
Cold start observed: {target.get("cold_start", "not returned")}
Warm-up received in same environment: {seen}
Environment reuse: {reuse}
An accepted asynchronous Lambda request does not guarantee that a later request reuses the same environment.'''
    sns.publish(TopicArn=topic, Subject=f"[ColdStart] Warm-up: {scenario}"[:100], Message=body)
    return True, f"Email published via SNS to topic subscribers (including {recipient})."


def looks_cold(f):
    """True when the inputs describe an idle / no-warm-capacity / no-budget workload."""
    return (f["invocations_last_5m"] <= 2 and f["warm_capacity"] == 0
            and f["provisioned_capacity"] == 0 and f["budget_ratio"] <= 0.2)


def scenario_label(f):
    """Name of the preset that matches the current values, otherwise 'Custom'."""
    for name, preset in SCENARIOS.items():
        if all(f.get(k) == v for k, v in preset.items()):
            return name
    return CUSTOM


# ------------------------------------------------- widget state / scenarios
DEFAULTS = SCENARIOS["Warm-up condition"]
for k, v in DEFAULTS.items():
    st.session_state.setdefault(f"w_{k}", v)
st.session_state.setdefault("scenario", CUSTOM)
st.session_state.setdefault("auto_run", True)
st.session_state.setdefault("send_email", False)
st.session_state.setdefault("threshold", 0.5)


def load_scenario():
    preset = SCENARIOS.get(st.session_state["scenario"])
    if preset:  # callbacks run before widgets are drawn, so this is safe
        for k, v in preset.items():
            st.session_state[f"w_{k}"] = v


def get_features():
    return {k: st.session_state[f"w_{k}"] for k in DEFAULTS}


with st.sidebar:
    st.subheader("AWS configuration")
    st.code(f"Region: {REGION or 'boto3 default'}\nController: {CONTROLLER}\nTarget: {TARGET}")
    st.caption("Uses standard boto3 credentials. No AWS secrets are collected in the UI.")
    if SNS_TOPIC:
        st.success("SNS topic configured")
    else:
        st.info("No SNS topic set - one will be created automatically when an email is sent.")
    st.subheader("Warm decision")
    st.slider("COLD if cold-start probability >=", 0.0, 1.0, 0.05, key="threshold",
              help="Condition is COLD (idle, on-demand) when the ML cold-start probability is at or above this "
                   "value, otherwise WARM (active traffic -> warm pool, if warm environments and budget are "
                   "available).")

st.info("Pick a scenario (or use the sliders). These are simulated feature values sent to the ML "
        "controller; they do not generate that amount of real traffic. The controller is called "
        "automatically every 10 seconds and again as soon as you change any parameter.")

top1, top2 = st.columns([3, 1])
with top1:
    st.selectbox("Scenario", [CUSTOM] + list(SCENARIOS), key="scenario", on_change=load_scenario,
                 help="Choosing a scenario fills all sliders below. Moving a slider afterwards makes it custom.")
with top2:
    st.write("")
    st.toggle("Auto-run every 10 s", key="auto_run",
              help="When off, use the 'Run now' button. Dashboard still refreshes every 10 s.")

with st.expander("Workload parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("User preference", ["auto", "cost_saving", "warm"], key="w_user_preference",
                     help="Preference passed to the controller.")
        st.slider("Invocations in last 5 minutes", 0, 200, 1, key="w_invocations_last_5m")
        st.slider("Concurrency", 1, 50, 1, key="w_concurrency")
    with col2:
        st.slider("Seconds since last invocation", 0, 7200, 30, key="w_seconds_since_last_invocation")
        st.slider("Burstiness", 0.0, 3.0, 0.1, key="w_burstiness")
        st.select_slider("Memory allocation (MB)",
                         options=[128, 256, 512, 1024, 1536, 2048, 3008, 4096, 8192, 10240],
                         key="w_memory_mb")
    with col3:
        st.slider("Package size (MB)", 1, 250, 1, key="w_package_mb")
        st.slider("Available warm environments", 0, 20, 1, key="w_warm_capacity")
        st.slider("Provisioned concurrency", 0, 50, 1, key="w_provisioned_capacity")
    st.slider("Normalized budget remaining", 0.0, 1.0, 0.05, key="w_budget_remaining")
    st.slider("Budget ratio", 0.0, 1.0, 0.05, key="w_budget_ratio")

with st.expander("Exact workload payload", expanded=False):
    st.json(get_features())

em1, em2 = st.columns([3, 1])
with em1:
    st.text_input("Recipient email", placeholder="you@example.com", key="recipient")
with em2:
    st.write("")
    st.checkbox("Email me when a warm-up is triggered", key="send_email",
                help="Sends one email each time a warm-up starts being triggered (not every 10 s).")


# ---------------------------------------------------------------- the run
def run_once(features, trigger, recipient, send_email, changed="-"):
    """Call the controller + target once with the current parameters, record the result, return it."""
    scenario_name = scenario_label(features)
    summary = {"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
               "trigger": trigger, "scenario": scenario_name, "condition": "unknown",
               "action": "Not returned", "warmup_triggered": False,
               "controller_status": "not_run", "decision_reason": "Not returned",
               "probability": "Not returned", "cold_start_observed": "Not returned",
               "environment_id": "Not returned",
               "preference": features["user_preference"],
               "invocations_last_5m": features["invocations_last_5m"],
               "seconds_since_last_invocation": features["seconds_since_last_invocation"],
               "concurrency": features["concurrency"],
               "burstiness": features["burstiness"],
               "memory_mb": features["memory_mb"],
               "package_mb": features["package_mb"],
               "warm_capacity": features["warm_capacity"],
               "provisioned_capacity": features["provisioned_capacity"],
               "budget_remaining": features["budget_remaining"],
               "budget_ratio": features["budget_ratio"],
               "threshold": st.session_state.get("threshold", 0.5),
               "changed_parameters": changed}
    out = {"summary": summary, "features": features, "result": None, "rec": {}, "target": {},
           "target_error": None, "error": None, "email": None, "triggered": False, "action": "Not returned"}
    try:
        lam, sns = clients()
        # use_live_state=False: otherwise the controller overwrites idle-time / invocations / concurrency
        # with DynamoDB telemetry and moving those sliders would change nothing.
        # 'Available warm environments' is a capacity limit (can we warm?), not the model's
        # active_warm_containers feature, so it is NOT mapped to that feature.
        event = {"mode": "predict", "function_id": TARGET_ID, "use_live_state": False,
                 "warm_probability_threshold": st.session_state.get("threshold", 0.5), **features}
        try:
            result = invoke(lam, CONTROLLER, event)
        except (BotoCoreError, ClientError, RuntimeError, KeyError, ValueError) as exc:
            summary.update(controller_status="invocation_failed", decision_reason=str(exc))
            out["error"] = f"Controller invocation failed: {exc}"
            return out
        out["result"] = result
        if result.get("status") == "configuration_error":
            msg = result.get("error", "Controller configuration error")
            summary.update(controller_status="configuration_error", decision_reason=msg)
            out["error"] = msg
            return out

        rec = get_rec(result)
        action = rec.get("action", "unknown")
        triggered = result.get("warmup_triggered") is True
        thr = rec.get("probability_threshold", st.session_state.get("threshold", 0.5))
        condition = condition_of(result.get("cold_start_probability"), thr)
        try:
            prob = f"{float(result.get('cold_start_probability')) * 100:.1f}%"
        except (TypeError, ValueError):
            prob = "Not returned"
        summary.update(condition=condition, action=action, warmup_triggered=triggered,
                       controller_status=result.get("status", "Not returned"),
                       decision_reason=rec.get("decision_reason", "Not returned"), probability=prob)
        out.update(rec=rec, triggered=triggered, action=action)

        if triggered:
            time.sleep(2)  # give the asynchronous warmer a moment before the follow-up request
        try:
            target = invoke(lam, TARGET, {"mode": "normal", "work_ms": 100})
            out["target"] = target
            summary.update(cold_start_observed=target.get("cold_start", "Not returned"),
                           environment_id=str(target.get("environment_id", "Not returned")))
        except (BotoCoreError, ClientError, RuntimeError, KeyError, ValueError) as exc:
            out["target_error"] = str(exc)

        # Email once when a warm-up gets triggered (COLD workload), not on every 10 s tick.
        prev = st.session_state.get("prev_triggered", False)
        if triggered and not prev and send_email:
            try:
                out["email"] = email(sns, scenario_name, result, out["target"], features, recipient,
                                     triggered, condition)
            except (BotoCoreError, ClientError) as exc:
                out["email"] = (False, f"Email failed: {exc}. The IAM identity needs sns:CreateTopic, "
                                       "sns:Subscribe, sns:ListSubscriptionsByTopic and sns:Publish.")
        st.session_state.prev_triggered = triggered
        return out
    finally:
        # Always record the run so the dashboard shows every entry (even errors).
        st.session_state.last_run = out
        add_history(summary)


def render_run(out):
    s, f = out["summary"], out["features"]
    st.markdown("## Current result")
    if out["error"]:
        st.markdown(f'<div class="cond cond-err">⚠️ Run failed<small>{out["error"]}</small></div>',
                    unsafe_allow_html=True)
        if out["result"]:
            st.json(out["result"])
        return

    cond, prob, action = s["condition"], s["probability"], out["action"]
    reason = s["decision_reason"]
    
    act = "warm pool warm-up triggered" if out["triggered"] else "on-demand, no warm-up"
    if cond == "warm":
        st.markdown(f'<div class="cond cond-warm"> WARM condition<small>Active workload · '
                    f'Cold-start probability {prob} · {act} ({reason})</small></div>', unsafe_allow_html=True)
    elif cond == "cold":
        st.markdown(f'<div class="cond cond-cold"> COLD condition<small>Idle / low traffic · '
                    f'Cold-start probability {prob} · {act} ({reason})</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="cond cond-err">❔ Condition unknown<small>The controller did not return a '
                    'cold-start probability.</small></div>', unsafe_allow_html=True)

    result = out["result"]
    x, y, z = st.columns(3)
    x.metric("Cold-start probability", prob)
    y.metric("Controller action", str(action).replace("_", " ").title())
    z.metric("Warm-up accepted", "Yes" if out["triggered"] else "No")
    st.write("**Model:**", result.get("model", "Not returned"),
             " · **Controller status:**", result.get("status", "Not returned"))
    if result.get("warmup_error"):
        st.warning(f"Warm-up error: {result['warmup_error']}")

    # Only displays what the controller returned; point at the controller if it looks inconsistent.
    if looks_cold(f) and cond == "warm":
        st.warning("These inputs describe an idle, low-traffic workload, but the model returned a low cold-start "
                   "probability (WARM). Check the model or lower the threshold slider.")

    st.markdown("#### Normal traffic invocation")
    if out["target_error"]:
        st.error(f"Target invocation failed: {out['target_error']}")
        if "AccessDeniedException" in out["target_error"] and "lambda:InvokeFunction" in out["target_error"]:
            st.warning("Fix: grant the IAM identity running this frontend lambda:InvokeFunction "
                       "permission on the exact target Lambda ARN shown in the error.")
    else:
        target = out["target"]
        env = target.get("environment_id", "Not returned")
        init = target.get("application_init_duration_ms", "Not returned")
        a, b, c = st.columns(3)
        a.metric("Environment ID", str(env))
        b.metric("Application init duration", f"{init} ms" if isinstance(init, (int, float)) else str(init))
        c.metric("Cold start observed", str(target.get("cold_start", "Not returned")))
        if out["triggered"] and target.get("warmup_received_in_environment") is True:
            st.success("Verified: target reports that this environment received the warm-up.")
        elif out["triggered"]:
            st.warning("Environment reuse was not verified by this follow-up request.")

    if out["email"]:
        (st.success if out["email"][0] else st.warning)(out["email"][1])
    with st.expander("Full controller / target response"):
        st.json({"controller": result, "target": out["target"]})


def render_dashboard():
    st.divider()
    st.subheader("Dashboard - all entries")
    rows = get_history()
    if not rows:
        st.info("No entries yet. The first one is recorded as soon as the controller responds.")
        return
    n_warm = sum(1 for r in rows if r.get("condition") == "warm")
    n_cold = sum(1 for r in rows if r.get("condition") == "cold")
    last = rows[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total entries", len(rows))
    m2.metric("WARM conditions", n_warm)
    m3.metric("COLD conditions", n_cold)
    m4.metric("Warm-ups triggered", sum(1 for r in rows if r.get("warmup_triggered")))
    m5.metric("Last condition", str(last.get("condition", "-")).upper())
    st.caption(f"Last entry: {last.get('timestamp', 'Not recorded')} · Scenario: {last.get('scenario', '-')} "
               f"· Dashboard refreshed {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    st.markdown("#### Request history (newest first)")
    cols = ["id", "timestamp", "trigger", "scenario", "condition", "action", "probability",
            "warmup_triggered", "controller_status", "cold_start_observed", "environment_id",
            "preference", "invocations_last_5m", "seconds_since_last_invocation", "concurrency",
            "burstiness", "memory_mb", "package_mb", "warm_capacity", "provisioned_capacity",
            "budget_remaining", "budget_ratio", "threshold", "changed_parameters", "decision_reason"]
    table = [{c: r.get(c, "") for c in cols} for r in reversed(rows)]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download history (JSON)", json.dumps(rows, indent=2, default=str),
                       file_name="coldstart_run_history.json", mime="application/json", key="dl_history")


@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def live_panel():
    """Runs on every full rerun (= a parameter changed) and on a 10 s timer."""
    features = get_features()
    full = {**features, "threshold": st.session_state.get("threshold", 0.5)}
    sig = json.dumps(full, sort_keys=True)
    prev_full = st.session_state.get("prev_full")
    last_sig = st.session_state.get("last_sig")
    last_ts = st.session_state.get("last_ts", 0.0)
    auto = st.session_state.auto_run

    manual = False if auto else st.button("▶ Run now", type="primary", key="run_now")

    trigger = None
    if manual:
        trigger = "manual"
    elif auto and last_sig is None:
        trigger = "first run"
    elif auto and sig != last_sig:
        trigger = "parameters changed"
    elif auto and time.time() - last_ts >= REFRESH_SECONDS - 1:
        trigger = f"auto ({REFRESH_SECONDS}s)"

    if trigger:
        st.session_state.last_sig = sig
        st.session_state.last_ts = time.time()
        changed = ", ".join(f"{k}: {prev_full[k]} -> {full[k]}" for k in full
                            if prev_full and prev_full.get(k) != full[k]) or "-"
        st.session_state.prev_full = dict(full)
        with st.spinner("Calling ML controller with the current parameters..."):
            run_once(features, trigger, st.session_state.get("recipient", "").strip(),
                     st.session_state.get("send_email", False), changed)

    last = st.session_state.get("last_run")
    if last:
        s = last["summary"]
        status = (f"Auto-run every {REFRESH_SECONDS} s" if auto else "Auto-run paused")
        st.caption(f"{status} · Last calculated {s['timestamp']} · Trigger: {s['trigger']} · "
                   f"Scenario: {s['scenario']}")
        render_run(last)
    else:
        st.info("Auto-run is paused. Click 'Run now' to calculate." if not auto else "Calculating...")
    render_dashboard()


live_panel()

st.divider()
st.caption("Lambda environment reuse is best-effort. Reuse is only marked verified when the target "
           "response explicitly reports warm-up receipt.")
