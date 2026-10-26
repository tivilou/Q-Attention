"""Real HTTPS subprocess regression for the private exchange accept loop."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collab/server/qattn_exchange_server.py"


@unittest.skipUnless(shutil.which("openssl"), "TLS regression needs openssl for an ephemeral local certificate")
class ExchangeTLSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.token = "local-test-token-" + "x" * 40
        (cls.root / "token").write_text(cls.token)
        (cls.root / "web").mkdir()
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(cls.root / "key.pem"), "-out", str(cls.root / "cert.pem"),
                        "-days", "1", "-subj", "/CN=localhost"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        # Use the production main/handler with a shorter idle deadline in this
        # fixture. No accept/handshake or authentication code is replaced.
        launcher = ("import importlib.util,sys;"
                    "s=importlib.util.spec_from_file_location('exchange',sys.argv[1]);"
                    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                    "m.UploadHandler.timeout=0.5;sys.argv=sys.argv[1:];m.main()")
        cls.child = subprocess.Popen([sys.executable, "-B", "-c", launcher, str(SCRIPT),
                                     "--bind", "127.0.0.1", "--port", str(cls.port),
                                     "--root", str(cls.root / "storage"), "--web-root", str(cls.root / "web"),
                                     "--token-file", str(cls.root / "token"),
                                     "--tls-cert", str(cls.root / "cert.pem"), "--tls-key", str(cls.root / "key.pem")],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.context = ssl.create_default_context(cafile=str(cls.root / "cert.pem"))
        until = time.monotonic() + 5
        while time.monotonic() < until:
            try:
                status, _, _ = cls.request("GET", "/healthz")
                if status == 200:
                    return
            except OSError:
                time.sleep(0.02)
        cls.child.kill(); cls.child.wait()
        cls.temp.cleanup()
        raise AssertionError("Production HTTPS main did not become healthy")

    @classmethod
    def tearDownClass(cls):
        cls.child.terminate()
        try:
            cls.child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            cls.child.kill(); cls.child.wait()
        cls.temp.cleanup()

    @classmethod
    def request(cls, method, path, body=None, token=None):
        client = http.client.HTTPSConnection("localhost", cls.port, context=cls.context, timeout=2)
        try:
            headers = {"Connection": "close"}
            if token is not None:
                headers["Authorization"] = "Bearer " + token
            client.request(method, path, body=body, headers=headers)
            response = client.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            client.close()

    def test_stalled_clients_do_not_block_accept(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as empty:
            with socket.create_connection(("127.0.0.1", self.port), timeout=2) as partial:
                partial.sendall(b"\x16")
                # Also hold a valid TLS connection before any HTTP headers.
                with socket.create_connection(("127.0.0.1", self.port), timeout=2) as raw:
                    with self.context.wrap_socket(raw, server_hostname="localhost"):
                        status, _, payload = self.request("GET", "/healthz")
                        self.assertEqual(status, 200)
                        self.assertEqual(json.loads(payload)["status"], "ok")

    def test_stalled_tls_connection_expires(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as raw:
            raw.settimeout(2)
            self.assertEqual(raw.recv(1), b"")
        self.assertEqual(self.request("GET", "/healthz")[0], 200)

    def test_auth_namespace_and_immutable_upload(self):
        data = b"closed-local-test-artifact\n"
        path = "/upload/q-attention/tls-regression/payload.bin"
        self.assertEqual(self.request("PUT", path, data, "wrong-token")[0], 401)
        self.assertEqual(self.request("PUT", "/upload/other-project/payload.bin", data, self.token)[0], 400)
        status, _, body = self.request("PUT", path, data, self.token)
        self.assertEqual(status, 201)
        self.assertEqual(json.loads(body)["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(self.request("PUT", path, data, self.token)[0], 200)
        self.assertEqual(self.request("PUT", path, b"different", self.token)[0], 409)
        self.assertEqual(self.request("HEAD", "/files/q-attention/tls-regression/payload.bin", token=self.token)[0], 200)
        self.assertEqual(self.request("GET", "/files/q-attention/tls-regression/payload.bin", token=self.token)[2], data)
        self.assertFalse(list((self.root / "storage").rglob("*.uploading")))


if __name__ == "__main__":
    unittest.main()
