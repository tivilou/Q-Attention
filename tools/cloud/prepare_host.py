"""Create dedicated client/host SSH keys and cloud-init in a protected directory.

Never print private keys or UserData. Host public key is trusted because the
corresponding private key is provisioned over the authenticated cloud API.
"""
import argparse, json, os, subprocess
from pathlib import Path

def create(root):
    root = Path(root)
    root.mkdir(exist_ok=False)
    client, host = root / 'client_rsa', root / 'host_ed25519'
    for path, kind, extra in ((client, 'rsa', ['-b', '2048']), (host, 'ed25519', [])):
        subprocess.run(['ssh-keygen', '-q', '-t', kind, *extra, '-N', '', '-f', str(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    public_client = client.with_suffix('.pub').read_text().strip()
    public_host = host.with_suffix('.pub').read_text().strip()
    private_host = host.read_text().strip()
    content = ('#cloud-config\n'
               'ssh_deletekeys: true\nssh_genkeytypes: []\n'
               'ssh_keys:\n  ed25519_private: |\n' + ''.join('    '+line+'\n' for line in private_host.splitlines())
               + '  ed25519_public: '+public_host+'\n'
               'ssh_publish_hostkeys:\n  enabled: false\n'
               'ssh_pwauth: false\nusers:\n  - default\n  - name: research\n'
               '    groups: sudo\n    shell: /bin/bash\n'
               '    sudo: ALL=(ALL) NOPASSWD:ALL\n    ssh_authorized_keys:\n      - '+public_client+'\n')
    with (root / 'cloud-init.yaml').open('x') as out:
        out.write(content); out.flush(); os.fsync(out.fileno())
    fingerprint = subprocess.check_output(['ssh-keygen', '-lf', str(host)+'.pub', '-E', 'sha256'], text=True).split()[1]
    return {'host_key_fingerprint': fingerprint, 'ssh_user': 'research', 'key_paths_created': True}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protected-directory', required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.protected_directory)))
