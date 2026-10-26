⚡ Serverless Cold Start Optimization
An ML-driven system that predicts when an AWS Lambda function is likely to suffer a cold start and decides whether to proactively warm it up or leave it on-demand. This repository contains the Streamlit frontend used to run simulated workload scenarios, inspect the controller's decision, verify what actually happened on the target Lambda, and receive an email when a warm-up is triggered.
---
1. The problem
A Lambda cold start happens when AWS has to create a new execution environment (download code, start the runtime, run init code) before handling a request. This adds latency, often hundreds of milliseconds to several seconds. Keeping functions warm avoids it, but warming costs money. The goal is to warm only when the expected benefit outweighs the cost.
2. Architecture
```
┌──────────────────┐   1. predict event    ┌─────────────────────────┐
│  Streamlit app   │ ────────────────────► │  ML controller Lambda   │
│  (app.py)        │ ◄──────────────────── │  coldstart-ml-controller│
│                  │   probability,        └───────────┬─────────────┘
│                  │   action, reason,                 │ 2. if warm-up: async invoke
│                  │   warmup_triggered                ▼
│                  │                          ┌────────────────────┐
│                  │  3. one normal request   │  Target Lambda     │
│                  │ ───────────────────────► │  coldstart-live-   │
│                  │ ◄─────────────────────── │  target            │
│                  │   env ID, init time,     └────────────────────┘
│                  │   cold_start flag
│                  │  4. if warmup_triggered
│                  │ ───────────────────────► Amazon SNS ──► your email
└──────────────────┘
```
Component	Role
Streamlit app (`app.py`)	UI. Sends scenario features, displays the decision and diagnostics, sends the email.
ML controller Lambda	Computes the cold-start probability, chooses an action, and triggers the warmer when warranted. Deployed separately.
Target Lambda	The function being protected. Reports its environment ID, init duration and whether it received a warm-up. Deployed separately.
Amazon SNS	Delivers the warm-up notification email.
> The controller and target Lambdas are **not** included in this repository. The frontend only invokes them.
3. Request flow
The user picks a scenario and clicks Run scenario.
The app invokes the controller with `{"mode": "predict", "function_id": ..., <features>, <cost parameters>}`.
The controller returns `cold_start_probability`, a `recommendation` (`action`, `decision_reason`), the `model` used, and `warmup_triggered`.
The app waits about 2 seconds, then sends one normal request to the target Lambda and reads back `environment_id`, `application_init_duration_ms`, `cold_start`, and `warmup_received_in_environment`.
Only if `warmup_triggered` is true, the app publishes an email through SNS. If the controller chooses on-demand, no warm-up section is shown and no email is sent.
4. Input features
These are simulated feature values sent to the controller. For example, `invocations_last_5m = 120` is a model input, not 120 real requests.
Feature	Meaning
`user_preference`	`warm`, `cost_saving`, or `auto`. Hint to the controller; it makes the final call.
`seconds_since_last_invocation`	Idle time. Longer idle time means a higher chance the environment was reclaimed.
`invocations_last_5m`	Recent request volume.
`concurrency`	Number of concurrent executions.
`burstiness`	How spiky traffic is (higher means more sudden spikes).
`memory_mb`	Configured Lambda memory.
`package_mb`	Deployment package size. Larger packages mean slower init.
`warm_capacity`	Warm environments currently available.
`budget_remaining` / `budget_ratio`	Remaining warm-up budget (absolute and as a fraction).
`provisioned_capacity`	Provisioned concurrency already configured.
Cost parameters sent with every request
Parameter	Value	Meaning
`cold_start_cost`	0.02	Cost (latency/penalty units) of one cold start
`warm_action_cost`	0.001	Cost of one warm-up action
`warm_effectiveness`	0.9	Probability that a warm-up actually prevents the cold start
5. Built-in scenarios
Scenario	Preference	Idle (s)	Inv / 5 min	Concurrency	Burstiness	Intent
Warm-up condition	warm	900	18	6	2.8	Warm preference; controller decides
Cold / cost-saving	cost_saving	3600	0	1	0.0	No budget, no demand, expect on-demand
Heavy simulated traffic	auto	30	120	20	3.0	High volume and burstiness
Light simulated traffic	cost_saving	1200	2	1	0.2	Low demand
User prefers warming	warm	180	35	8	2.4	Explicit warm preference
6. Cold-start probability and the warm-up decision
6.1 Where the probability comes from
`cold_start_probability` is computed by the ML controller Lambda, not by this frontend. The frontend only displays it (as a percentage, `probability × 100`). The `model` field in the response names the model that produced it.
Conceptually, the controller estimates the probability that the next request will land on a fresh environment:
```
P(cold) = P(cold start on next invocation | x)
```
where `x` is the feature vector from section 4. For a logistic-type classifier, which is the usual form for this kind of binary "cold / not cold" prediction, this is:
```
z       = b + w1·x1 + w2·x2 + … + wn·xn
P(cold) = 1 / (1 + e^(−z))
```
`x1…xn` are the features (idle time, recent invocations, concurrency, burstiness, memory, package size, warm capacity, …).
`w1…wn` and `b` are learned weights and intercept from training data.
Expected direction of influence: idle time, burstiness, concurrency and package size push the probability up; recent invocations, warm capacity and provisioned capacity push it down.
> **Fill in before publishing:** the exact model type, coefficients/trees, and feature scaling live in the controller's code and are not part of this repository. Replace this subsection with the real formula from the deployed controller (look at the `model` field returned in the response and the controller's `FEATURES` schema).
6.2 Cost-aware warm-up decision
The probability is then combined with the cost parameters to decide whether warming pays off. The standard expected-cost rule for these inputs is:
```
Expected cold-start loss avoided = P(cold) × cold_start_cost × warm_effectiveness
Net benefit                      = P(cold) × cold_start_cost × warm_effectiveness − warm_action_cost

Warm up        if Net benefit > 0
Stay on-demand otherwise
```
Equivalently, warm up when:
```
P(cold) > warm_action_cost / (cold_start_cost × warm_effectiveness)
```
With the default costs the break-even threshold is:
```
0.001 / (0.02 × 0.9) = 0.001 / 0.018 ≈ 0.0556  (≈ 5.6 %)
```
Worked example (illustrative probability): if the model returns `P(cold) = 0.70`:
```
Net benefit = 0.70 × 0.02 × 0.9 − 0.001 = 0.0126 − 0.001 = 0.0116 > 0  →  warm up
```
The controller may also apply constraints on top of this rule (remaining budget, `user_preference`, existing warm or provisioned capacity). That is why a "warm" preference does not guarantee a warm-up, and a cost-saving preference with a zero budget results in on-demand. The reason is always returned in `decision_reason`.
> The rule above is the expected-cost formulation matching the cost inputs the app sends. Confirm it against the controller's actual decision logic.
7. Setup
Requirements
Python 3.10+
AWS credentials available to boto3 (environment variables, `~/.aws`, or an IAM role)
The controller and target Lambdas already deployed
Install and run
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
PowerShell example:
```powershell
$env:AWS_REGION="ap-south-1"
$env:CONTROLLER_FUNCTION_NAME="coldstart-ml-controller"
$env:TARGET_FUNCTION_NAME="coldstart-live-target"
$env:TARGET_FUNCTION_ID="coldstart-live-target"
streamlit run app.py
```
Environment variables
Variable	Default	Purpose
`AWS_REGION`	boto3 default	Region of the Lambdas and SNS
`CONTROLLER_FUNCTION_NAME`	`coldstart-ml-controller`	Controller Lambda
`TARGET_FUNCTION_NAME`	`coldstart-live-target`	Target Lambda
`TARGET_FUNCTION_ID`	`coldstart-live-target`	`function_id` sent to the controller
`WARMUP_NOTIFICATION_TOPIC_ARN`	(empty)	Existing SNS topic. If empty, the app uses or creates one
`WARMUP_NOTIFICATION_TOPIC_NAME`	`coldstart-warmup-notifications`	Name of the auto-created topic
8. Email notifications
Enter your address in Recipient email in the app.
On the first run that triggers a warm-up, the app subscribes your address to the SNS topic (creating the topic if needed) and shows a message saying a confirmation email was sent.
Open the email from AWS (check spam) and click Confirm subscription. This is a one-time step; SNS does not deliver to unconfirmed addresses.
Run the scenario again. When the controller triggers a warm-up, you receive an email with the scenario, features, action, reason, probability, model, and target diagnostics.
No email is sent when the controller chooses on-demand.
9. IAM permissions
The identity running Streamlit needs:
`lambda:InvokeFunction` on the controller and target function ARNs
`sns:CreateTopic`, `sns:Subscribe`, `sns:ListSubscriptionsByTopic`, `sns:Publish` (for the notification topic)
The controller's own execution role separately needs permission to invoke the warmer. Use least-privilege policies and never hardcode credentials.
10. Understanding the results
Field	Meaning
Cold-start probability	Controller's predicted chance of a cold start
Controller action	Chosen action (for example warm up or on-demand)
Warm-up accepted	AWS accepted the asynchronous warmer invocation
Environment ID	Execution environment that served the follow-up request
Application init duration	Init time, reported on cold starts
Cold start observed	Whether the follow-up request hit a cold start
Verified (green banner)	Target explicitly reported it received the warm-up in the same environment
11. Limitations
Scenario values are simulated inputs, not generated traffic.
An accepted async invoke does not guarantee the warm-up completed or that a later request reuses the same environment. Lambda environment reuse is best-effort, so reuse is marked verified only when the target says so.
Confirm feature names and optional cost fields against the deployed controller's `FEATURES` schema.
12. Troubleshooting
Symptom	Fix
`AccessDeniedException ... lambda:InvokeFunction`	Grant the frontend's IAM identity invoke permission on the exact ARN in the error
Controller returns `configuration_error`	Check the controller's environment variables and model files
No email arrives	Confirm the SNS subscription, check spam, and note that email is only sent when a warm-up is triggered
"Email failed" error	Add the SNS permissions listed in section 9
13. Project structure
```
serverless_cold_start_frontend/
├── app.py             # Streamlit frontend
├── requirements.txt   # streamlit, boto3
└── README.md
```
