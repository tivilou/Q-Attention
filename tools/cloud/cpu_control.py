"""Single-node Tencent CPU test controller. Not an autoscaler or scheduler.

Persist the ledger on local NTFS. Keep UserData and private keys separately in an
ACL-protected directory. The API credential must remain on the operator machine.
"""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import time
import uuid
from api import Tencent, ApiError

def utcnow():
    return datetime.now(timezone.utc)

def iso(value):
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write_json(path, data):
    # Desktop AppData virtualization may redirect C: to a package cache on D:.
    # Resolve before both sides of MoveFileEx; mixed virtual paths can fail EXDEV.
    path = Path(path).resolve()
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x', encoding='utf-8') as out:
        json.dump(data, out, indent=2)
        out.flush(); os.fsync(out.fileno())
    os.replace(temp, path)

@contextmanager
def locked(path):
    lock = Path(str(path) + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        yield
    finally:
        lock.unlink()

def validate_spec(spec, region):
    allowed = {'InstanceChargeType', 'InstanceCount', 'Placement', 'InstanceType',
               'ImageId', 'SystemDisk', 'InternetAccessible'}
    if set(spec) != allowed or spec['InstanceChargeType'] != 'POSTPAID_BY_HOUR' or spec['InstanceCount'] != 1:
        raise ValueError('Only one explicitly specified hourly instance is allowed')
    if set(spec['Placement']) != {'Zone'} or not spec['Placement']['Zone'].startswith(region + '-'):
        raise ValueError('Region/zone mismatch')
    if spec['SystemDisk'] != {'DiskType': 'CLOUD_PREMIUM', 'DiskSize': 50}:
        raise ValueError('Expected one 50 GiB system disk')
    if spec['InternetAccessible'] != {'InternetChargeType': 'TRAFFIC_POSTPAID_BY_HOUR',
                                       'InternetMaxBandwidthOut': 5, 'PublicIpAssigned': True}:
        raise ValueError('Unexpected network charging or bandwidth')

def check_price(quote, max_hourly, max_gb):
    price = quote['Price']
    hourly, traffic = price['InstancePrice'], price['BandwidthPrice']
    if hourly['ChargeUnit'] != 'HOUR' or traffic['ChargeUnit'] != 'GB':
        raise ValueError('Unexpected price units')
    def amount(item):
        raw = item.get('UnitPriceDiscount')
        if raw is None:
            raw = item['UnitPrice']
        number = Decimal(str(raw))
        if not number.is_finite() or number < 0:
            raise ValueError('Invalid price')
        return number
    h, gb = amount(hourly), amount(traffic)
    if h > Decimal(str(max_hourly)) or gb > Decimal(str(max_gb)):
        raise ValueError('Quote exceeds price ceiling')
    return {'hourly_cny': str(h), 'egress_cny_per_gb': str(gb)}

def policies_match(actual, wanted):
    if len(actual) != len(wanted):
        return False
    for row, expected in zip(actual, wanted):
        for key, value in expected.items():
            if key == 'PolicyDescription':
                continue
            observed = row.get(key)
            if key == 'Protocol' and isinstance(observed, str):
                observed = observed.upper()  # API canonicalizes TCP to tcp.
            if observed != value:
                return False
    return True

class Controller:
    def __init__(self, api, path):
        self.api, self.path = api, Path(path).resolve()
        self.data = json.loads(self.path.read_text(encoding='utf-8'))
        if self.data['region'] != api.region:
            raise ValueError('API region differs from ledger')
        validate_spec(self.data['spec'], api.region)

    def save(self):
        write_json(self.path, self.data)

    def event(self, action, **fields):
        self.data['events'].append({'at': iso(utcnow()), 'action': action, **fields})
        self.save()

    def check_hardware(self):
        spec = self.data['spec']
        configs = self.api.call('cvm', 'DescribeInstanceTypeConfigs', {
            'Filters': [{'Name': 'instance-type', 'Values': [spec['InstanceType']]},
                        {'Name': 'zone', 'Values': [spec['Placement']['Zone']]}]})['InstanceTypeConfigSet']
        if len(configs) != 1 or configs[0]['CPU'] != 2 or configs[0]['Memory'] != 4 or configs[0].get('GpuCount') or configs[0].get('GPU') or configs[0].get('FPGA'):
            raise ValueError('Expected exactly 2 CPU / 4 GiB / no GPU or FPGA')
        images = self.api.call('cvm', 'DescribeImages', {'ImageIds': [spec['ImageId']],
                                                       'InstanceType': spec['InstanceType']})['ImageSet']
        if len(images) != 1 or images[0]['Architecture'] != 'x86_64' or images[0]['ImageType'] != 'PUBLIC_IMAGE' or not images[0]['IsSupportCloudinit']:
            raise ValueError('Expected compatible public x86_64 cloud-init image')

    def quote(self, request=None):
        quote = self.api.call('cvm', 'InquiryPriceRunInstances', request or self.data['spec'])
        prices = check_price(quote, self.data['max_hourly_cny'], self.data['max_egress_cny_per_gb'])
        self.data['last_quote'] = {'at': iso(utcnow()), 'prices': prices, 'response': quote}
        self.save()
        return prices

    def once_aux(self, kind, service, action, params, field):
        previous = self.data['resources'].get(kind)
        if previous:
            if previous['status'] == 'created':
                return previous['id']
            raise ValueError('Uncertain auxiliary create; reconcile by unique name before continuing')
        self.data['resources'][kind] = {'status': 'pending', 'action': action}
        self.save()  # Do not retry a non-idempotent call after ambiguous transport failure.
        response = self.api.call(service, action, params)
        value = response
        for key in field:
            value = value[key]
        self.data['resources'][kind] = {'status': 'created', 'id': value}
        self.event(action, resource_kind=kind, resource_id=value, request_id=response['RequestId'])
        return value

    def prepare(self, public_key):
        if self.data.get('instance_request'):
            raise ValueError('Instance launch is already frozen')
        if not public_key.startswith('ssh-rsa '):
            raise ValueError('Provider ImportKeyPair requires OpenSSH RSA public key')
        self.check_hardware()
        self.quote()
        name = self.data['name']
        vpc = self.once_aux('vpc', 'vpc', 'CreateVpc', {'VpcName': name, 'CidrBlock': '10.66.0.0/24'}, ['Vpc', 'VpcId'])
        subnet = self.once_aux('subnet', 'vpc', 'CreateSubnet', {
            'VpcId': vpc, 'SubnetName': name, 'Zone': self.data['spec']['Placement']['Zone'],
            'CidrBlock': '10.66.0.0/26'}, ['Subnet', 'SubnetId'])
        sg = self.once_aux('security_group', 'vpc', 'CreateSecurityGroup', {
            'GroupName': name, 'GroupDescription': 'Temporary bounded research CPU recovery test'}, ['SecurityGroup', 'SecurityGroupId'])
        self.ensure_security_group_policies(sg)
        key_name = 'rscpu_' + hashlib.sha256(name.encode()).hexdigest()[:16]
        self.data['key_pair_name'] = key_name
        self.save()
        key = self.once_aux('key_pair', 'cvm', 'ImportKeyPair', {'KeyName': key_name, 'ProjectId': 0, 'PublicKey': public_key.strip()}, ['KeyId'])
        self.data['prepared'] = True
        self.event('PrepareComplete', vpc_id=vpc, subnet_id=subnet, security_group_id=sg, key_id=key)

    def ensure_security_group_policies(self, sg):
        if self.data['resources']['security_group']['id'] != sg:
            raise ValueError('Unknown security group')
        policies = {'Ingress': [{'Protocol': 'TCP', 'Port': '22', 'CidrBlock': self.data['ssh_cidr'],
                                 'Action': 'ACCEPT', 'PolicyDescription': self.data['name']}],
                    'Egress': [{'Protocol': 'ALL', 'CidrBlock': '0.0.0.0/0',
                                'Action': 'ACCEPT', 'PolicyDescription': self.data['name']}]}
        for direction in ('Ingress', 'Egress'):
            existing = self.api.call('vpc', 'DescribeSecurityGroupPolicies', {'SecurityGroupId': sg})['SecurityGroupPolicySet']
            rows = existing.get(direction) or []
            if rows and not policies_match(rows, policies[direction]):
                raise ValueError('Unexpected security group policy; reconcile before retry')
            if not rows:
                # Provider requires one direction per request; ALL protocol must omit Port.
                response = self.api.call('vpc', 'CreateSecurityGroupPolicies', {
                    'SecurityGroupId': sg, 'SecurityGroupPolicySet': {direction: policies[direction]}})
                self.event('CreateSecurityGroupPolicies', direction=direction, request_id=response['RequestId'])
            actual = self.api.call('vpc', 'DescribeSecurityGroupPolicies', {'SecurityGroupId': sg})['SecurityGroupPolicySet']
            if not policies_match(actual.get(direction) or [], policies[direction]):
                raise ValueError('Security group readback mismatch')
            # Record confirmed partial progress too. Retry reuses exact provider readback.
            self.data.setdefault('security_group_policies', {})[direction] = policies[direction]
            self.event('SecurityGroupPolicyVerified', direction=direction)

    def create(self, userdata):
        if not self.data.get('prepared'):
            raise ValueError('Resources are not prepared')
        if self.data.get('instance_id') or self.data.get('launch_attempted'):
            raise ValueError('Launch already attempted; use reconcile, never a new ClientToken')
        self.check_hardware()
        deadline = utcnow() + timedelta(hours=self.data['max_hours'])
        resources = self.data['resources']
        request = {**self.data['spec'], 'InstanceName': self.data['name'],
                   'ClientToken': self.data['client_token'],
                   'LoginSettings': {'KeyIds': [resources['key_pair']['id']]},
                   'SecurityGroupIds': [resources['security_group']['id']],
                   'VirtualPrivateCloud': {'VpcId': resources['vpc']['id'], 'SubnetId': resources['subnet']['id']},
                   'ActionTimer': {'TimerAction': 'TerminateInstances', 'ActionTime': iso(deadline)},
                   'DisableApiTermination': False}
        self.quote(request)  # Re-quote the actual placement/network/login before paid operation.
        self.data['instance_request'] = request
        self.data['userdata_sha256'] = hashlib.sha256(userdata).hexdigest()
        self.data['deadline'] = iso(deadline)
        self.data['launch_attempted'] = True
        self.event('RunInstancesPending')
        return self.submit_frozen(userdata)

    def submit_frozen(self, userdata):
        request = self.data['instance_request']
        if request['ClientToken'] != self.data['client_token'] or hashlib.sha256(userdata).hexdigest() != self.data['userdata_sha256']:
            raise ValueError('Frozen token or UserData identity mismatch')
        response = self.api.call('cvm', 'RunInstances', {**request, 'UserData': base64.b64encode(userdata).decode()})
        ids = response['InstanceIdSet']
        if len(ids) != 1:
            raise ValueError('Unexpected instance count; reconcile immediately')
        self.data['instance_id'] = ids[0]
        self.event('RunInstances', instance_id=ids[0], request_id=response['RequestId'])
        self.verify_timer()
        self.event('AutomaticTerminationVerified')
        return ids[0]

    def retry_create(self, userdata):
        """Explicit operator replay of the identical token/body after reconciliation."""
        if not self.data.get('launch_attempted') or self.data.get('instance_id'):
            raise ValueError('Replay requires unresolved recorded launch')
        if hashlib.sha256(userdata).hexdigest() != self.data['userdata_sha256']:
            raise ValueError('UserData identity mismatch')
        remaining = datetime.fromisoformat(self.data['deadline'].replace('Z', '+00:00')) - utcnow()
        if not timedelta(minutes=5) < remaining <= timedelta(hours=4):
            raise ValueError('Original deadline expired or outside replay window')
        instances = self.api.call('cvm', 'DescribeInstances', {
            'Filters': [{'Name': 'instance-name', 'Values': [self.data['name']]}], 'Limit': 100})['InstanceSet']
        if instances:
            self.reconcile()
            self.verify_timer()
            return self.data['instance_id']
        # Same absolute deadline, count, token and payload; never generate a new token.
        self.quote(self.data['instance_request'])
        self.event('RunInstancesReplayPending', frozen_request_sha256=digest(self.data['instance_request']))
        return self.submit_frozen(userdata)

    def reconcile(self):
        instances = self.api.call('cvm', 'DescribeInstances', {
            'Filters': [{'Name': 'instance-name', 'Values': [self.data['name']]}], 'Limit': 100})['InstanceSet']
        exact = [x for x in instances if x['InstanceName'] == self.data['name']]
        if len(exact) != 1:
            raise ValueError('Reconciliation requires exactly one matching instance; no automatic relaunch')
        if not self.data.get('launch_attempted'):
            raise ValueError('No recorded launch intent')
        self.data['instance_id'] = exact[0]['InstanceId']
        self.verify_instance()
        self.event('ReconcileLaunch', instance_id=self.data['instance_id'])

    def verify_instance(self):
        instance_id = self.data.get('instance_id')
        if not instance_id or not self.data.get('launch_attempted'):
            raise ValueError('No owned instance ID')
        instances = self.api.call('cvm', 'DescribeInstances', {'InstanceIds': [instance_id]})['InstanceSet']
        if not instances:
            return None
        if len(instances) != 1:
            raise ValueError('Unexpected instance count')
        instance = instances[0]
        if (instance['InstanceId'] != instance_id or instance['InstanceName'] != self.data['name']
            or instance['InstanceType'] != self.data['spec']['InstanceType']
            or instance['Placement']['Zone'] != self.data['spec']['Placement']['Zone']
            or instance['InstanceChargeType'] != 'POSTPAID_BY_HOUR'
            or instance['VirtualPrivateCloud']['VpcId'] != self.data['resources']['vpc']['id']
            or any(instance['SystemDisk'].get(k) != v for k, v in self.data['spec']['SystemDisk'].items())
            or instance['CPU'] != 2 or instance['Memory'] != 4 or (instance.get('GPUInfo') or {}).get('GPUCount', 0)):
            raise ValueError('Instance ownership/spec mismatch')
        return instance

    def verify_timer(self):
        for attempt in range(10):
            timer = self.api.call('cvm', 'DescribeInstancesActionTimer', {'InstanceIds': [self.data['instance_id']]})
            self.data['last_timer'] = timer
            self.save()
            rows = timer.get('ActionTimers') or []
            matching = [t for t in rows if t.get('InstanceId') == self.data['instance_id']
                        and t.get('TimerAction') == 'TerminateInstances' and t.get('ActionTime') == self.data['deadline']
                        and t.get('Status') == 'UNDO']
            if len(matching) == 1 and len(rows) == 1:
                return timer
            if rows or attempt == 9:
                break  # Conflicting timers fail immediately; only empty readback is retried.
            time.sleep(2)
        # Owned ID is already durable. Never create again to repair timer visibility.
        raise ValueError('Automatic termination timer not verified; terminate owned node')

    def status(self):
        instance = self.verify_instance()
        self.data['last_instance'] = instance
        timer = self.verify_timer() if instance and instance['InstanceState'] in ('RUNNING', 'STOPPED') else None
        self.event('DescribeInstances')
        return {'instance': instance, 'timer': timer}

    def power(self, action):
        if action not in ('StartInstances', 'StopInstances', 'RebootInstances'):
            raise ValueError('Unsupported power action')
        instance = self.verify_instance()
        state = {'StartInstances': 'STOPPED', 'StopInstances': 'RUNNING', 'RebootInstances': 'RUNNING'}[action]
        if instance is None or instance['InstanceState'] != state or utcnow() >= datetime.fromisoformat(self.data['deadline'].replace('Z', '+00:00')):
            raise ValueError('Unexpected instance state or expired deadline')
        params = {'InstanceIds': [self.data['instance_id']]}
        if action == 'StopInstances':
            params.update(StopType='SOFT', StoppedMode='KEEP_CHARGING')
        response = self.api.call('cvm', action, params)
        self.event(action, request_id=response['RequestId'])

    def terminate(self):
        instance = self.verify_instance()
        if instance is None or instance['InstanceState'] == 'TERMINATED':
            self.data['termination_verified'] = True
            self.event('TerminationVerified')
            return
        response = self.api.call('cvm', 'TerminateInstances', {'InstanceIds': [self.data['instance_id']]})
        self.event('TerminateInstances', request_id=response['RequestId'])
        # A successful asynchronous request is not proof of termination. Invoke again after polling.

    def cleanup_aux(self):
        if not self.data.get('termination_verified'):
            if self.data.get('launch_attempted'):
                raise ValueError('Verify instance termination before network/key cleanup')
            existing = self.api.call('cvm', 'DescribeInstances', {'Filters': [{'Name': 'instance-name', 'Values': [self.data['name']]}], 'Limit': 100})['InstanceSet']
            if existing:
                raise ValueError('Unexpected instance exists before launch')
        for kind, service, describe, delete, params, collection, id_field, name_field in (
            ('security_group', 'vpc', 'DescribeSecurityGroups', 'DeleteSecurityGroup', 'SecurityGroupId', 'SecurityGroupSet', 'SecurityGroupId', 'SecurityGroupName'),
            ('subnet', 'vpc', 'DescribeSubnets', 'DeleteSubnet', 'SubnetId', 'SubnetSet', 'SubnetId', 'SubnetName'),
            ('vpc', 'vpc', 'DescribeVpcs', 'DeleteVpc', 'VpcId', 'VpcSet', 'VpcId', 'VpcName'),
            ('key_pair', 'cvm', 'DescribeKeyPairs', 'DeleteKeyPairs', 'KeyIds', 'KeyPairSet', 'KeyId', 'KeyName')):
            resource = self.data['resources'].get(kind)
            if not resource or resource['status'] == 'deleted':
                continue
            if resource['status'] != 'created':
                raise ValueError('Unknown auxiliary resource; reconcile first')
            resource_id = resource['id']
            query = {'KeyIds': [resource_id]} if kind == 'key_pair' else {id_field + 's': [resource_id]}
            def read_owned_rows():
                try:
                    return self.api.call(service, describe, query)[collection]
                except ApiError as error:
                    try:
                        code = json.loads(str(error)).get('code')
                    except (ValueError, AttributeError):
                        raise error
                    # Actual VPC SG describe returns this after successful deletion,
                    # rather than an empty set. Only this exact owned-ID query maps
                    # the observed absence code; authorization/transport errors fail.
                    if describe == 'DescribeSecurityGroups' and code == 'ResourceNotFound':
                        return []
                    raise
            rows = read_owned_rows()
            expected_name = self.data.get('key_pair_name', self.data['name']) if kind == 'key_pair' else self.data['name']
            if rows and (len(rows) != 1 or rows[0][id_field] != resource_id or rows[0][name_field] != expected_name):
                raise ValueError('Auxiliary resource ownership mismatch')
            if rows:
                if kind == 'security_group':
                    actual = self.api.call('vpc', 'DescribeSecurityGroupPolicies', {'SecurityGroupId': resource_id})['SecurityGroupPolicySet']
                    expected = self.data.get('security_group_policies')
                    if expected is None:
                        expected = {'Ingress': [], 'Egress': []}
                    for direction in ('Ingress', 'Egress'):
                        if not policies_match(actual.get(direction) or [], expected.get(direction) or []):
                            raise ValueError('Security group policies changed; reconcile before deletion')
                response = self.api.call(service, delete, {params: [resource_id] if kind == 'key_pair' else resource_id})
                self.event(delete, request_id=response['RequestId'], resource_id=resource_id)
            if read_owned_rows():
                raise ValueError('Deletion not yet verified; retry cleanup later')
            resource['status'] = 'deleted'
            self.event('AuxiliaryDeletionVerified', resource_kind=kind, resource_id=resource_id)

def initialize(path, spec, region, cidr, name, hours):
    if not re.fullmatch(r'research-cpu-[a-z0-9-]{8,40}', name):
        raise ValueError('Unique research-cpu- name required')
    network = ipaddress.ip_network(cidr, strict=True)
    if network.version != 4 or network.prefixlen != 32 or not network.network_address.is_global:
        raise ValueError('SSH source must be one public IPv4 /32')
    if not 0 < hours <= 4:
        raise ValueError('Test lifetime must be <= 4 hours')
    validate_spec(spec, region)
    path = Path(path)
    if path.exists():
        raise ValueError('Ledger exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {'schema_version': 1, 'region': region, 'name': name, 'spec': spec,
                      'ssh_cidr': str(network), 'max_hours': hours,
                      'max_hourly_cny': '0.50', 'max_egress_cny_per_gb': '0.80',
                      'client_token': uuid.uuid4().hex, 'resources': {}, 'events': []})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['init', 'quote', 'prepare', 'create', 'retry', 'reconcile', 'status', 'start', 'stop', 'reboot', 'terminate', 'cleanup'])
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--credential')
    parser.add_argument('--region')
    parser.add_argument('--spec')
    parser.add_argument('--ssh-cidr')
    parser.add_argument('--name')
    parser.add_argument('--hours', type=float, default=4)
    parser.add_argument('--public-key')
    parser.add_argument('--userdata')
    args = parser.parse_args()
    if args.operation == 'init':
        initialize(args.ledger, json.loads(Path(args.spec).read_text(encoding='utf-8')), args.region, args.ssh_cidr, args.name, args.hours)
        print(json.dumps({'initialized': True})); return
    if not args.region or not args.credential:
        raise ValueError('Explicit region and local credential required')
    with locked(args.ledger):
        controller = Controller(Tencent(args.region, args.credential), args.ledger)
        operation = args.operation
        if operation == 'prepare':
            result = controller.prepare(Path(args.public_key).read_text(encoding='ascii'))
        elif operation == 'create':
            result = controller.create(Path(args.userdata).read_bytes())
        elif operation == 'retry':
            result = controller.retry_create(Path(args.userdata).read_bytes())
        elif operation in ('start', 'stop', 'reboot'):
            result = controller.power({'start': 'StartInstances', 'stop': 'StopInstances', 'reboot': 'RebootInstances'}[operation])
        elif operation == 'cleanup':
            result = controller.cleanup_aux()
        else:
            result = getattr(controller, operation)()
        print(json.dumps({'operation': operation, 'result': result}, ensure_ascii=False))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, ApiError, FileExistsError) as error:
        print(json.dumps({'error': str(error)})); raise SystemExit(1)
