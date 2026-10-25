"""Local-only official SDK bridge; never logs credentials or raw SDK errors."""
import json
import re
from pathlib import Path

class ApiError(RuntimeError):
    pass

def error_fields(error, action):
    fields = {'action': action, 'code': error.get_code(), 'request_id': error.get_request_id()}
    # Permission aliases can differ from modern SDK names. Expose only a strict
    # operation identifier, never the raw message or request (possibly UserData).
    if error.get_code() == 'UnauthorizedOperation':
        match = re.search(r'operation \(([A-Za-z][A-Za-z0-9]*:[A-Za-z][A-Za-z0-9]*)\)', error.get_message())
        if match:
            fields['permission_action'] = match.group(1)
    return fields

class Tencent:
    def __init__(self, region, credential_path):
        from tencentcloud.common.credential import Credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.cvm.v20170312.cvm_client import CvmClient
        from tencentcloud.vpc.v20170312.vpc_client import VpcClient
        from tencentcloud.cvm.v20170312 import models as cm
        from tencentcloud.vpc.v20170312 import models as vm
        values = json.loads(Path(credential_path).read_text(encoding='utf-8'))
        credential = Credential(values['SecretId'], values['SecretKey'])
        profile = ClientProfile(httpProfile=HttpProfile(reqTimeout=45))
        self.region = region
        self.clients = {'cvm': CvmClient(credential, region, profile),
                        'vpc': VpcClient(credential, region, profile)}
        self.models = {'cvm': cm, 'vpc': vm}

    def call(self, service, action, params):
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
        request = getattr(self.models[service], action + 'Request')()
        request.from_json_string(json.dumps(params))
        try:
            return json.loads(getattr(self.clients[service], action)(request).to_json_string())
        except TencentCloudSDKException as error:
            raise ApiError(json.dumps(error_fields(error, action))) from None
