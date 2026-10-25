"""Version-pinned Tencent COS transport for qpersist snapshots.

Requires bucket versioning Enabled and cos-python-sdk-v5==1.9.44.
Credentials are injected through environment variables, never command arguments.
"""
import hashlib
import os
import re
import tempfile
from pathlib import Path


class CosError(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class CosStore:
    def __init__(self, bucket, region, prefix, receipt=None, client=None):
        if not isinstance(bucket, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}-[0-9]+', bucket):
            raise CosError('Invalid bucket')
        if not isinstance(region, str) or not re.fullmatch(r'[a-z]+-[a-z]+(?:-[a-z]+)?', region):
            raise CosError('Invalid region')
        if not isinstance(prefix, str) or not re.fullmatch(r'projects/[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9/-]*/', prefix) or '//' in prefix:
            raise CosError('Explicit project/subdirectory prefix required')
        self.bucket, self.region, self.prefix = bucket, region, prefix
        self.versions = {}
        self.read_only = receipt is not None
        if receipt is not None:
            location = receipt.get('cos', {})
            if any(location.get(k) != v for k, v in [('bucket', bucket), ('region', region), ('prefix', prefix)]):
                raise CosError('Receipt location mismatch')
            versions = location.get('object_versions')
            if not isinstance(versions, dict) or not versions:
                raise CosError('Missing version map')
            for name, version in versions.items():
                self.key(name)
                self.version(version)
            self.versions = dict(versions)
        if client is None:
            try:
                from qcloud_cos import CosConfig, CosS3Client
                sid = os.environ['TENCENTCLOUD_SECRET_ID']
                secret = os.environ['TENCENTCLOUD_SECRET_KEY']
                if not sid or not secret:
                    raise ValueError()
                client = CosS3Client(CosConfig(
                    Region=region, SecretId=sid, SecretKey=secret,
                    Token=os.environ.get('TENCENTCLOUD_SESSION_TOKEN'),
                    Scheme='https', Timeout=60, AllowRedirects=False,
                    VerifySSL=True, AutoSwitchDomainOnRetry=False))
            except Exception:
                raise CosError('COS SDK or credentials unavailable') from None
        self.client = client

    @staticmethod
    def version(value):
        if not isinstance(value, str) or not value or value == 'null' or len(value) > 1024 or any(ord(c) < 33 for c in value):
            raise CosError('Versioned object required; check bucket versioning')
        return value

    def key(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[0-9a-f]{32}--(?:manifest\.json|chunk-[0-9]{6}\.bin)', name):
            raise CosError('Invalid snapshot object name')
        return self.prefix + name

    @staticmethod
    def response_version(response):
        headers = {k.lower(): v for k, v in response.items() if isinstance(k, str)}
        return CosStore.version(headers.get('x-cos-version-id'))

    def _read(self, name, destination, version=None, missing_ok=False):
        body = None
        try:
            kwargs = {'VersionId': version} if version is not None else {}
            result = self.client.get_object(Bucket=self.bucket, Key=self.key(name), **kwargs)
            body = result['Body'].get_raw_stream()
            actual_version = self.response_version(result)
            if version is not None and actual_version != version:
                raise CosError('Object version mismatch')
            with open(destination, 'wb') as target:
                while True:
                    block = body.read(4 * 1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            return actual_version
        except CosError:
            raise
        except Exception as exc:
            if missing_ok and callable(getattr(exc, 'get_status_code', None)) and exc.get_status_code() == 404:
                return None
            raise CosError('COS read failed') from None
        finally:
            if body is not None:
                body.close()

    def get(self, name, destination):
        self.key(name)
        if name not in self.versions:
            raise CosError('Object not pinned by receipt or this publication')
        self._read(name, destination, self.versions[name])

    def put(self, name, source):
        if self.read_only:
            raise CosError('Restore credentials context cannot publish')
        key = self.key(name)
        with tempfile.TemporaryDirectory(prefix='qpersist-cos-') as tmp:
            check = Path(tmp) / 'existing'
            version = self._read(name, check, self.versions.get(name), missing_ok=name not in self.versions)
            if version is not None:
                if sha(check) != sha(source):
                    raise CosError('Immutable snapshot conflict')
                self.versions[name] = version
                return
        try:
            with open(source, 'rb') as payload:
                response = self.client.put_object(
                    Bucket=self.bucket, Key=key, Body=payload, EnableMD5=True,
                    StorageClass='STANDARD', ServerSideEncryption='AES256')
            self.versions[name] = self.response_version(response)
        except CosError:
            raise
        except Exception:
            raise CosError('COS write failed; no success receipt') from None

    def receipt_metadata(self):
        return {'cos': {'bucket': self.bucket, 'region': self.region,
                        'prefix': self.prefix, 'object_versions': dict(self.versions)}}
