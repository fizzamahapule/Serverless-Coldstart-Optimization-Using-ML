import os, json, time
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
</style><div class="hero"><h1>⚡ Serverless Cold Start Optimization</h1>
<p>ML-driven Lambda warm-up simulation · Run a scenario and inspect actual results.</p></div>''', unsafe_allow_html=True)

REGION=os.getenv("AWS_REGION") or None
CONTROLLER=os.getenv("CONTROLLER_FUNCTION_NAME","coldstart-ml-controller")
TARGET=os.getenv("TARGET_FUNCTION_NAME","coldstart-live-target")
TARGET_ID=os.getenv("TARGET_FUNCTION_ID","coldstart-live-target")
SNS_TOPIC=os.getenv("WARMUP_NOTIFICATION_TOPIC_ARN","")

SCENARIOS={
"Warm-up condition":{"user_preference":"warm","seconds_since_last_invocation":900,"invocations_last_5m":18,"concurrency":6,"burstiness":2.8,"memory_mb":512,"package_mb":35,"warm_capacity":2,"budget_remaining":1.0,"budget_ratio":0.9,"provisioned_capacity":0},
"Cold / cost-saving":{"user_preference":"cost_saving","seconds_since_last_invocation":3600,"invocations_last_5m":0,"concurrency":1,"burstiness":0.0,"memory_mb":128,"package_mb":20,"warm_capacity":0,"budget_remaining":0.0,"budget_ratio":0.1,"provisioned_capacity":0},
"Heavy simulated traffic":{"user_preference":"auto","seconds_since_last_invocation":30,"invocations_last_5m":120,"concurrency":20,"burstiness":3.0,"memory_mb":1024,"package_mb":45,"warm_capacity":2,"budget_remaining":1.0,"budget_ratio":0.9,"provisioned_capacity":0},
"Light simulated traffic":{"user_preference":"cost_saving","seconds_since_last_invocation":1200,"invocations_last_5m":2,"concurrency":1,"burstiness":0.2,"memory_mb":256,"package_mb":20,"warm_capacity":1,"budget_remaining":0.25,"budget_ratio":0.3,"provisioned_capacity":0},
"User prefers warming":{"user_preference":"warm","seconds_since_last_invocation":180,"invocations_last_5m":35,"concurrency":8,"burstiness":2.4,"memory_mb":512,"package_mb":30,"warm_capacity":2,"budget_remaining":1.0,"budget_ratio":0.9,"provisioned_capacity":0}}
COST={"cold_start_cost":0.02,"warm_action_cost":0.001,"warm_effectiveness":0.9}

def decode(resp):
 raw=resp["Payload"].read()
 try: data=json.loads(raw.decode()) if raw else {}
 except Exception: return {"raw_response":raw.decode(errors="replace")}
 if isinstance(data,dict) and "body" in data:
  body=data["body"]
  if isinstance(body,str):
   try: body=json.loads(body)
   except Exception: return data
  if isinstance(body,dict):
   body.setdefault("_http_status_code",data.get("statusCode")); return body
 return data if isinstance(data,dict) else {"result":data}

def invoke(client,name,payload):
 r=client.invoke(FunctionName=name,InvocationType="RequestResponse",Payload=json.dumps(payload).encode())
 result=decode(r)
 if r.get("FunctionError"): raise RuntimeError(f"{name} Lambda error: {json.dumps(result,default=str)}")
 return result

def get_rec(result):
 rec=result.get("recommendation",{})
 if isinstance(rec,str):
  try: rec=json.loads(rec)
  except Exception: rec={"action":rec}
 return rec if isinstance(rec,dict) else {}

def email(sns,scenario,result,target,features):
 if not SNS_TOPIC: return False,"Warm-up triggered, but WARMUP_NOTIFICATION_TOPIC_ARN is not configured."
 rec=get_rec(result); seen=target.get("warmup_received_in_environment")
 reuse="VERIFIED" if seen is True else "NOT VERIFIED"
 body=f'''ColdStart warm-up event
Scenario: {scenario}
Timestamp UTC: {datetime.now(timezone.utc).isoformat()}
Controller: {CONTROLLER}
Target: {TARGET}
Simulated feature values (not actual request volume):
{json.dumps(features,indent=2)}
Action: {rec.get("action","not returned")}
Reason: {rec.get("decision_reason","not returned")}
Cold-start probability: {result.get("cold_start_probability","not returned")}
Model: {result.get("model","not returned")}
Warm-up request accepted: {result.get("warmup_triggered",False)}
Target environment ID: {target.get("environment_id","not returned")}
Application init duration ms: {target.get("application_init_duration_ms","not returned")}
Cold start observed: {target.get("cold_start","not returned")}
Warm-up received in same environment: {seen}
Environment reuse: {reuse}
An accepted asynchronous Lambda request does not guarantee that a later request reuses the same environment.'''
 sns.publish(TopicArn=SNS_TOPIC,Subject=f"[ColdStart] Warm-up: {scenario}"[:100],Message=body)
 return True,"Email notification published to SNS."

with st.sidebar:
 st.subheader("AWS configuration")
 st.code(f"Region: {REGION or 'boto3 default'}\nController: {CONTROLLER}\nTarget: {TARGET}")
 st.caption("Uses standard boto3 credentials. No AWS secrets are collected in the UI.")
 st.success("SNS email configured") if SNS_TOPIC else st.warning("SNS email topic not configured")

st.info("Traffic values are simulated feature inputs. 120 invocations is a model feature value, not 120 real Lambda requests.")
scenario=st.selectbox("Choose a scenario",list(SCENARIOS))
features=SCENARIOS[scenario]
st.write({"Warm-up condition":"Warm preference; controller makes final decision.","Cold / cost-saving":"Cost-saving preference and no warm capacity.","Heavy simulated traffic":"High simulated volume and burstiness.","Light simulated traffic":"Low simulated demand.","User prefers warming":"Explicit warm preference, subject to controller constraints."}[scenario])
with st.expander("Workload features sent to controller",expanded=True):
 a,b,c,d=st.columns(4)
 a.metric("Invocations / 5 min",features["invocations_last_5m"]); b.metric("Concurrency",features["concurrency"])
 c.metric("Burstiness",features["burstiness"]); d.metric("Preference",features["user_preference"])
 st.json(features)

if st.button("▶ Run scenario",type="primary",use_container_width=True):
 lam=boto3.client("lambda",region_name=REGION); sns=boto3.client("sns",region_name=REGION)
 event={"mode":"predict","function_id":TARGET_ID,**features,**COST}
 st.markdown("## Run results")
 with st.spinner("Invoking ML controller..."):
  try: result=invoke(lam,CONTROLLER,event)
  except (BotoCoreError,ClientError,RuntimeError,KeyError,ValueError) as exc:
   st.error(f"Controller invocation failed: {exc}"); st.stop()
 if result.get("status")=="configuration_error":
  st.error(result.get("error","Controller configuration error")); st.json(result); st.stop()
 rec=get_rec(result); action=rec.get("action","unknown"); triggered=result.get("warmup_triggered") is True
 try: prob=f"{float(result.get('cold_start_probability'))*100:.1f}%"
 except (TypeError,ValueError): prob="Not returned"
 x,y,z=st.columns(3); x.metric("Cold-start probability",prob); y.metric("Controller action",str(action).replace("_"," ").title()); z.metric("Warm-up accepted","Yes" if triggered else "No")
 st.write("**Decision reason:**",rec.get("decision_reason","Not returned"))
 st.write("**Model:**",result.get("model","Not returned"))
 st.write("**Controller status:**",result.get("status","Not returned"))
 if triggered:
  st.success("Warm-up request triggered. Waiting briefly before one normal follow-up request.")
  time.sleep(2)
  st.markdown("### Warm-up invocation")
  st.success("AWS accepted the asynchronous warmer invocation.")
  st.caption("Acceptance does not prove completion or guarantee environment reuse.")
 else:
  st.info("No warm-up invocation was triggered. No warm-up section or email will be shown.")
 target={}; target_error=None
 with st.spinner("Sending one normal request to target Lambda..."):
  try: target=invoke(lam,TARGET,{"mode":"normal","work_ms":100})
  except (BotoCoreError,ClientError,RuntimeError,KeyError,ValueError) as exc: target_error=str(exc)
 st.markdown("### Normal traffic invocation")
 if target_error:
  st.error(f"Target invocation failed: {target_error}")
  if "AccessDeniedException" in target_error and "lambda:InvokeFunction" in target_error:
   st.warning(
    "Fix: grant the IAM identity running this frontend lambda:InvokeFunction "
    "permission on the exact target Lambda ARN shown in the error. "
    "This is separate from the controller's execution-role permissions."
   )
 else:
  env=target.get("environment_id","Not returned"); init=target.get("application_init_duration_ms","Not returned"); cold=target.get("cold_start","Not returned")
  x,y,z=st.columns(3); x.metric("Environment ID",str(env)); y.metric("Application init duration",f"{init} ms" if isinstance(init,(int,float)) else str(init)); z.metric("Cold start observed",str(cold))
  if triggered and target.get("warmup_received_in_environment") is True: st.success("Verified: target reports that this environment received the warm-up.")
  elif triggered: st.warning("Environment reuse was not verified by this follow-up request.")
  st.json(target)
 if triggered:
  try:
   sent,msg=email(sns,scenario,result,target,features)
   if sent:
    st.success(msg)
   else:
    st.warning(msg)
  except (BotoCoreError,ClientError) as exc:
   st.error(f"Warm-up triggered, but email failed: {exc}")
 else: st.caption("No email sent because the controller did not trigger a warm-up.")
 with st.expander("Full controller response"): st.json(result)

st.divider()
st.caption("Lambda environment reuse is best-effort. Reuse is only marked verified when the target response explicitly reports warm-up receipt.")
