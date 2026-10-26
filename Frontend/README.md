# Serverless Cold Start Optimization frontend

## Run locally
Requires Python 3.10+, AWS credentials configured for boto3, and deployed Lambda functions.

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

PowerShell example (replace names with your deployed resources):

```powershell
$env:AWS_REGION="ap-south-1"
$env:CONTROLLER_FUNCTION_NAME="coldstart-ml-controller"
$env:TARGET_FUNCTION_NAME="coldstart-live-target"
$env:TARGET_FUNCTION_ID="coldstart-live-target"
streamlit run app.py
```

## Email setup
1. Create an Amazon SNS Standard topic.
2. Subscribe the recipient's email and confirm the subscription.
3. Set `WARMUP_NOTIFICATION_TOPIC_ARN` to that topic's ARN before running the app.
4. The AWS identity running Streamlit needs `lambda:InvokeFunction` for the controller and target, plus `sns:Publish` for the topic. Use least-privilege policies; never hardcode credentials.

## Behavior
- The controller's actual `warmup_triggered` response controls whether warm-up details and email appear.
- The app sends one separate normal invocation to the target for diagnostic fields.
- Scenario counts are simulated feature values, not real generated requests.
- An async invoke accepted by AWS does not guarantee successful completion or reuse of the same Lambda environment.
- Confirm feature names and optional cost fields against the deployed controller's `FEATURES` schema before use.

### IAM permissions needed

The AWS identity used by Streamlit must have `lambda:InvokeFunction` on both the controller and target function ARNs. The controller's own execution role separately needs permission to invoke the warmer function. SNS email is optional; configure `WARMUP_NOTIFICATION_TOPIC_ARN` only after creating an SNS topic and confirming the recipient subscription.
