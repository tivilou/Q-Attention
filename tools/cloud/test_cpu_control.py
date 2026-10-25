import copy, json
from datetime import timedelta
import pytest
from cpu_control import Controller, initialize, utcnow, iso, check_price, validate_spec, write_json
from api import error_fields

SPEC = {'InstanceChargeType': 'POSTPAID_BY_HOUR', 'InstanceCount': 1,
        'Placement': {'Zone': 'ap-beijing-6'}, 'InstanceType': 'BF1.MEDIUM4', 'ImageId': 'img-public',
        'SystemDisk': {'DiskType': 'CLOUD_PREMIUM', 'DiskSize': 50},
        'InternetAccessible': {'InternetChargeType': 'TRAFFIC_POSTPAID_BY_HOUR', 'InternetMaxBandwidthOut': 5, 'PublicIpAssigned': True}}
QUOTE = {'Price': {'InstancePrice': {'UnitPrice': .26, 'UnitPriceDiscount': .22, 'ChargeUnit': 'HOUR'},
                   'BandwidthPrice': {'UnitPrice': .8, 'UnitPriceDiscount': .8, 'ChargeUnit': 'GB'}}, 'RequestId': 'quote'}

class Fake:
    region = 'ap-beijing'
    def __init__(self):
        self.calls = []
        self.instance = None
        self.price = copy.deepcopy(QUOTE)
        self.lose_create_response = False
        self.timer = None
        self.sg_policies = {'Ingress': [], 'Egress': []}
        self.fail_egress_once = False
    def call(self, service, action, params):
        self.calls.append((service, action, copy.deepcopy(params)))
        if action == 'DescribeInstanceTypeConfigs':
            return {'InstanceTypeConfigSet': [{'CPU': 2, 'Memory': 4, 'GPU': 0, 'FPGA': 0, 'GpuCount': 0}]}
        if action == 'DescribeImages':
            return {'ImageSet': [{'Architecture': 'x86_64', 'ImageType': 'PUBLIC_IMAGE', 'IsSupportCloudinit': True}]}
        if action == 'InquiryPriceRunInstances':
            return self.price
        if action == 'RunInstances':
            self.instance = {'InstanceId': 'ins-owned', 'InstanceName': params['InstanceName'],
                             'InstanceType': params['InstanceType'], 'Placement': params['Placement'],
                             'InstanceChargeType': 'POSTPAID_BY_HOUR', 'CPU': 2, 'Memory': 4,
                             'GPUInfo': {'GPUCount': 0}, 'VirtualPrivateCloud': params['VirtualPrivateCloud'],
                             'SystemDisk': params['SystemDisk'],
                             'InstanceState': 'RUNNING'}
            self.timer = {**params['ActionTimer'], 'InstanceId': 'ins-owned', 'Status': 'UNDO'}
            if self.lose_create_response:
                raise TimeoutError('simulated lost response')
            return {'InstanceIdSet': ['ins-owned'], 'RequestId': 'create'}
        if action == 'DescribeInstances':
            return {'InstanceSet': [self.instance] if self.instance else []}
        if action == 'DescribeInstancesActionTimer':
            return {'ActionTimers': [self.timer] if self.timer else []}
        if action == 'DescribeSecurityGroupPolicies':
            result = copy.deepcopy(self.sg_policies)
            for rows in result.values():
                for row in rows:
                    row['Protocol'] = row['Protocol'].lower()
            return {'SecurityGroupPolicySet': result}
        if action == 'CreateSecurityGroupPolicies':
            policies = params['SecurityGroupPolicySet']
            assert len(policies) == 1
            direction = next(iter(policies))
            if direction == 'Egress':
                assert 'Port' not in policies[direction][0]
                if self.fail_egress_once:
                    self.fail_egress_once = False
                    raise TimeoutError('injected egress failure')
            self.sg_policies[direction].extend(copy.deepcopy(policies[direction]))
            return {'RequestId': 'rules'}
        if action == 'ImportKeyPair':
            import re
            assert params['ProjectId'] == 0
            assert re.fullmatch(r'[A-Za-z0-9_]{1,25}', params['KeyName'])
            assert params['PublicKey'].startswith('ssh-rsa ')
            return {'KeyId': 'skey-new', 'RequestId': 'import'}
        if action in ('StartInstances', 'StopInstances', 'RebootInstances', 'TerminateInstances'):
            return {'RequestId': 'mutation'}
        raise AssertionError(action)

@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr('cpu_control.time.sleep', lambda seconds: None)
    path = tmp_path / 'ledger.json'
    initialize(path, SPEC, 'ap-beijing', '8.8.8.8/32', 'research-cpu-test-12345678', 4)
    api = Fake()
    c = Controller(api, path)
    c.data['prepared'] = True
    c.data['resources'] = {kind: {'status': 'created', 'id': value} for kind, value in
                            [('vpc', 'vpc-owned'), ('subnet', 'subnet-owned'),
                             ('security_group', 'sg-owned'), ('key_pair', 'skey-owned')]}
    c.save()
    return c, api

def test_eventually_visible_timer_does_not_repeat_paid_create(configured):
    c, api = configured
    original = api.call
    reads = []
    def delayed(service, action, params):
        if action == 'DescribeInstancesActionTimer':
            reads.append(action)
            if len(reads) < 3:
                return {'ActionTimers': []}
        return original(service, action, params)
    api.call = delayed
    c.create(b'cloud-config')
    assert len(reads) == 3
    assert sum(action == 'RunInstances' for _, action, _ in api.calls) == 1

def test_conflicting_timer_fails_without_polling(configured):
    c, api = configured
    c.create(b'cloud-config')
    api.calls.clear()
    api.timer['ActionTime'] = '2000-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='timer'):
        c.verify_timer()
    assert [action for _, action, _ in api.calls] == ['DescribeInstancesActionTimer']

def test_real_quote_and_timer_frozen_before_paid_call(configured):
    c, api = configured
    original_token = c.data['client_token']
    c.create(b'private-host-key-canary')
    request = [p for _, a, p in api.calls if a == 'RunInstances'][0]
    assert request['ClientToken'] == original_token
    assert request['ActionTimer']['TimerAction'] == 'TerminateInstances'
    assert request['InstanceCount'] == 1
    assert request['InstanceChargeType'] == 'POSTPAID_BY_HOUR'
    assert c.data['launch_attempted']
    assert 'UserData' not in c.path.read_text()
    assert 'private-host-key-canary' not in c.path.read_text()
    assert [a for _, a, _ in api.calls][-3:] == ['InquiryPriceRunInstances', 'RunInstances', 'DescribeInstancesActionTimer']
    with pytest.raises(ValueError, match='already attempted'):
        c.create(b'private-host-key-canary')

def test_lost_create_response_reconciles_without_second_launch(configured):
    c, api = configured
    api.lose_create_response = True
    with pytest.raises(TimeoutError):
        c.create(b'key')
    c = Controller(api, c.path)
    with pytest.raises(ValueError, match='already attempted'):
        c.create(b'key')
    c.reconcile()
    assert c.data['instance_id'] == 'ins-owned'
    assert sum(a == 'RunInstances' for _, a, _ in api.calls) == 1

def test_explicit_replay_reconciles_visible_instance_without_launch(configured):
    c, api = configured; api.lose_create_response = True
    with pytest.raises(TimeoutError): c.create(b'key')
    c = Controller(api, c.path)
    assert c.retry_create(b'key') == 'ins-owned'
    assert sum(a == 'RunInstances' for _, a, _ in api.calls) == 1

def test_explicit_replay_preserves_token_body_userdata_and_deadline(configured):
    c, api = configured; api.lose_create_response = True
    with pytest.raises(TimeoutError): c.create(b'key')
    original = copy.deepcopy([p for _, a, p in api.calls if a == 'RunInstances'][0])
    api.instance = None; api.lose_create_response = False
    c = Controller(api, c.path)
    with pytest.raises(ValueError, match='identity'):
        c.retry_create(b'different-key')
    assert c.retry_create(b'key') == 'ins-owned'
    requests = [p for _, a, p in api.calls if a == 'RunInstances']
    assert requests == [original, original]

def test_price_increase_prevents_paid_call(configured):
    c, api = configured
    api.price['Price']['InstancePrice']['UnitPriceDiscount'] = 100
    with pytest.raises(ValueError, match='ceiling'):
        c.create(b'key')
    assert not any(a == 'RunInstances' for _, a, _ in api.calls)
    assert not c.data.get('launch_attempted')

@pytest.mark.parametrize('unit', ['MONTH', 'DAY', None])
def test_price_units_are_not_guessed(unit):
    quote = copy.deepcopy(QUOTE)
    quote['Price']['InstancePrice']['ChargeUnit'] = unit
    with pytest.raises(ValueError):
        check_price(quote, .5, .8)

@pytest.mark.parametrize('bad_price', [-1, 'NaN', 'Infinity'])
def test_invalid_prices_block(bad_price):
    quote = copy.deepcopy(QUOTE)
    quote['Price']['InstancePrice']['UnitPriceDiscount'] = bad_price
    with pytest.raises(ValueError):
        check_price(quote, .5, .8)

@pytest.mark.parametrize('field,value', [('InstanceCount', 2), ('InstanceChargeType', 'PREPAID'), ('DataDisks', [{}])])
def test_purchase_contract_rejects_expansion(field, value):
    spec = copy.deepcopy(SPEC); spec[field] = value
    with pytest.raises(ValueError):
        validate_spec(spec, 'ap-beijing')

def test_cross_region_and_wrong_instance_cannot_mutate(configured):
    c, api = configured
    api.region = 'ap-shanghai'
    with pytest.raises(ValueError, match='region'):
        Controller(api, c.path)
    api.region = 'ap-beijing'
    c.create(b'key')
    api.instance['InstanceName'] = 'unrelated'
    with pytest.raises(ValueError, match='ownership'):
        c.power('StopInstances')
    with pytest.raises(ValueError, match='ownership'):
        c.terminate()
    assert not any(a in ('StopInstances', 'TerminateInstances') for _, a, _ in api.calls)

def test_expired_node_cannot_be_started(configured):
    c, api = configured; c.create(b'key')
    api.instance['InstanceState'] = 'STOPPED'
    c.data['deadline'] = iso(utcnow() - timedelta(seconds=1))
    with pytest.raises(ValueError, match='expired'):
        c.power('StartInstances')

def test_cleanup_requires_confirmed_termination(configured):
    c, api = configured; c.create(b'key')
    c.terminate()
    assert not c.data.get('termination_verified')
    with pytest.raises(ValueError, match='termination'):
        c.cleanup_aux()
    api.instance = None
    c.terminate()
    assert c.data['termination_verified']

def test_uncertain_aux_create_is_not_blindly_retried(configured):
    c, api = configured
    c.data['resources']['new_aux'] = {'status': 'pending'}
    with pytest.raises(ValueError, match='reconcile'):
        c.once_aux('new_aux', 'vpc', 'CreateVpc', {}, ['Vpc', 'VpcId'])
    assert not api.calls

def test_unsupported_provider_key_fails_before_any_cloud_mutation(configured):
    c, api = configured
    with pytest.raises(ValueError, match='RSA'):
        c.prepare('ssh-ed25519 invalid-for-provider')
    assert not api.calls

def test_permission_hint_omits_raw_request_and_secret():
    class SdkError:
        def get_code(self): return 'UnauthorizedOperation'
        def get_request_id(self): return 'request-123'
        def get_message(self): return 'operation (cvm:DescribeSecurityGroupPolicys) request-secret-canary'
    error = SdkError()
    result = error_fields(error, 'DescribeSecurityGroupPolicies')
    assert result['permission_action'] == 'cvm:DescribeSecurityGroupPolicys'
    assert 'request-secret-canary' not in json.dumps(result)

def test_security_group_partial_failure_retry_reuses_confirmed_direction(configured):
    c, api = configured
    api.fail_egress_once = True
    with pytest.raises(TimeoutError):
        c.ensure_security_group_policies('sg-owned')
    assert len(api.sg_policies['Ingress']) == 1 and api.sg_policies['Egress'] == []
    c = Controller(api, c.path)
    c.ensure_security_group_policies('sg-owned')
    assert len(api.sg_policies['Ingress']) == len(api.sg_policies['Egress']) == 1
    assert sum('Ingress' in p['SecurityGroupPolicySet'] for _, a, p in api.calls if a == 'CreateSecurityGroupPolicies') == 1

def test_modified_security_group_is_not_reconciled_as_ours(configured):
    c, api = configured
    api.sg_policies['Ingress'] = [{'Protocol': 'ALL', 'CidrBlock': '0.0.0.0/0', 'Action': 'ACCEPT'}]
    with pytest.raises(ValueError, match='Unexpected security'):
        c.ensure_security_group_policies('sg-owned')
    assert not any(a == 'CreateSecurityGroupPolicies' for _, a, _ in api.calls)

def test_prepare_satisfies_provider_key_import_contract(configured):
    c, api = configured
    del c.data['resources']['key_pair']
    c.prepare('ssh-rsa synthetic-public-key')
    assert c.data['resources']['key_pair']['id'] == 'skey-new'
    assert c.data['prepared']

def test_wrong_disk_blocks_mutation(configured):
    c, api = configured; c.create(b'key')
    api.instance['SystemDisk'] = {'DiskType': 'CLOUD_SSD', 'DiskSize': 500}
    with pytest.raises(ValueError, match='ownership'):
        c.terminate()

def test_missing_timer_fails_closed_with_owned_id_preserved(configured):
    c, api = configured; c.create(b'key')
    api.timer = None
    with pytest.raises(ValueError, match='timer'):
        c.status()
    assert c.data['instance_id'] == 'ins-owned'
    with pytest.raises(ValueError, match='already attempted'):
        c.create(b'key')

@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows AppData virtualization regression')
def test_actual_appdata_atomic_write():
    import os, uuid
    from pathlib import Path
    root = Path(os.environ['LOCALAPPDATA']) / 'research-infrastructure/secrets'
    path = root / ('atomic-test-' + uuid.uuid4().hex + '.json')
    try:
        write_json(path, {'value': 1})
        write_json(path, {'value': 2})
        assert json.loads(path.read_text()) == {'value': 2}
    finally:
        path.resolve().unlink(missing_ok=True)

def test_name_cidr_and_lifetime_limits(tmp_path):
    for name, cidr, hours in [('research-cpu-unique-1234', '0.0.0.0/0', 4),
                               ('ordinary', '8.8.8.8/32', 4),
                               ('research-cpu-unique-1234', '8.8.8.8/32', 24)]:
        with pytest.raises(ValueError):
            initialize(tmp_path / 'ledger.json', SPEC, 'ap-beijing', cidr, name, hours)
