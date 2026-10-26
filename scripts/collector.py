import base64, gzip, json, os, re
from datetime import datetime, timezone
import boto3

REGION = os.environ.get('AWS_REGION_NAME') or os.environ.get('AWS_REGION')
HISTORY_TABLE = os.environ.get('HISTORY_TABLE', 'coldstart-invocation-history')
LIVE_STATE_TABLE = os.environ.get('LIVE_STATE_TABLE', 'coldstart-live-state')
STREAM_STATE_TABLE = os.environ.get('STREAM_STATE_TABLE', 'coldstart-log-stream-state')
ddb = boto3.resource('dynamodb', region_name=REGION)
history = ddb.Table(HISTORY_TABLE)
live = ddb.Table(LIVE_STATE_TABLE)
streams = ddb.Table(STREAM_STATE_TABLE)
REPORT_RE = re.compile(r'REPORT RequestId:\s*([0-9a-fA-F-]+).*?Duration:\s*([\d.]+)\s*ms.*?Billed Duration:\s*(\d+)\s*ms.*?Memory Size:\s*(\d+)\s*MB.*?Max Memory Used:\s*(\d+)\s*MB', re.S)
INIT_RE = re.compile(r'Init Duration:\s*([\d.]+)\s*ms')

def handler(event, context):
    data = event.get('awslogs', {}).get('data')
    if not data:
        raise ValueError('Missing awslogs.data')
    payload = json.loads(gzip.decompress(base64.b64decode(data)).decode('utf-8'))
    if payload.get('messageType') == 'CONTROL_MESSAGE':
        return {'statusCode': 200, 'body': 'control message ignored'}
    group = payload.get('logGroup', '')
    prefix = '/aws/lambda/'
    function_id = group[len(prefix):].strip('/') if group.startswith(prefix) else ''
    if not function_id or function_id == os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
        return {'statusCode': 200, 'body': 'unsupported/self log group ignored'}
    stream = payload.get('logStream', 'unknown-stream')
    written = 0
    for ev in payload.get('logEvents', []):
        msg = ev.get('message', '')
        m = REPORT_RE.search(msg)
        if not m:
            continue
        request_id, duration, billed, mem_size, max_mem = m.groups()
        init_match = INIT_RE.search(msg)
        init_ms = float(init_match.group(1)) if init_match else None
        previous = streams.get_item(Key={'log_stream': stream}, ConsistentRead=True).get('Item')
        if init_ms is not None:
            status, cold = 'cold', True
        elif previous and previous.get('has_report') is True:
            status, cold = 'warm', False
        else:
            status, cold = 'unknown', None
        ts = int(ev.get('timestamp', 0))
        observed_at = datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat()
        oid = f'{ts:013d}#{request_id}'
        item = {'function_id': function_id, 'observation_id': oid, 'request_id': request_id,
                'log_stream': stream, 'observed_at': observed_at, 'event_timestamp_ms': ts,
                'cold_start_status': status, 'duration_ms': float(duration),
                'billed_duration_ms': int(billed), 'memory_size_mb': int(mem_size),
                'max_memory_used_mb': int(max_mem)}
        if cold is not None:
            item['cold_start_observed'] = cold
        if init_ms is not None:
            item['init_duration_ms'] = init_ms
        history.put_item(Item=item)
        streams.put_item(Item={'log_stream': stream, 'function_id': function_id,
                               'has_report': True, 'last_request_id': request_id,
                               'last_report_timestamp_ms': ts, 'last_cold_start_status': status})
        expr = ('SET last_observation_id=:oid, last_observed_at=:at, last_duration_ms=:dur, '
                'last_billed_duration_ms=:bill, last_memory_size_mb=:mem, last_max_memory_used_mb=:maxmem, '
                'last_cold_start_status=:status, last_log_stream=:stream, last_request_id=:rid')
        vals = {':oid': oid, ':at': observed_at, ':dur': float(duration), ':bill': int(billed),
                ':mem': int(mem_size), ':maxmem': int(max_mem), ':status': status,
                ':stream': stream, ':rid': request_id}
        if init_ms is not None:
            expr += ', last_init_duration_ms=:init'
            vals[':init'] = init_ms
        if cold is not None:
            expr += ', last_cold_start_observed=:cold'
            vals[':cold'] = cold
        live.update_item(Key={'function_id': function_id}, UpdateExpression=expr, ExpressionAttributeValues=vals)
        written += 1
    print(json.dumps({'event': 'telemetry_batch_processed', 'function_id': function_id,
                      'log_stream': stream, 'records_written': written}))
    return {'statusCode': 200, 'function_id': function_id, 'records_written': written}

# AWS Lambda handler setting: collector.handler
