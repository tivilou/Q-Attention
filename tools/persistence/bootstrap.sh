#!/usr/bin/env bash
# Usage: bash bootstrap.sh FULL_COMMIT NEW_DIRECTORY [VERIFIED_LOCAL_GIT_BUNDLE]
set -euo pipefail
commit=${1:?full source commit required}
destination=${2:?new destination required}
bundle=${3:-}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'Need full 40-character commit' >&2; exit 2; }
[[ ! -e "$destination" && ! -L "$destination" ]] || { echo 'Destination exists' >&2; exit 2; }
python_bin=${PYTHON_BIN:-python3}
"$python_bin" -c 'import sys,platform; assert sys.version_info[:2] == (3,10), "Python 3.10 required"; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Linux x86_64 required"'
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lock="$script_dir/requirements-linux-py310.lock"
[[ -f "$lock" ]] || { echo 'Missing pinned environment definition' >&2; exit 2; }
expected_env=${EXPECTED_ENV_SHA256:?trusted environment SHA256 from private recovery index required}
[[ "$expected_env" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$(sha256sum -- "$lock" | cut -d ' ' -f1)" == "$expected_env" ]] || { echo 'Environment hash mismatch' >&2; exit 2; }
if [[ -n "$bundle" ]]; then
  expected_bundle=${EXPECTED_BUNDLE_SHA256:?trusted bundle SHA256 required}
  [[ "$expected_bundle" =~ ^[0-9a-f]{64}$ ]] || exit 2
  [[ "$(sha256sum -- "$bundle" | cut -d ' ' -f1)" == "$expected_bundle" ]] || { echo 'Bundle hash mismatch' >&2; exit 2; }
fi
mkdir -- "$destination"
# Failure leaves this explicitly incomplete directory for inspection; never delete user files.
touch "$destination/DEPLOYMENT_INCOMPLETE"
git clone --no-checkout https://github.com/tivilou/Q-Attention.git "$destination/code"
if [[ -n "$bundle" ]]; then
  git -C "$destination/code" bundle verify "$bundle"
  git -C "$destination/code" fetch "$bundle" 'refs/heads/*:refs/remotes/recovery/*'
fi
git -C "$destination/code" checkout --detach "$commit"
[[ "$(git -C "$destination/code" rev-parse HEAD)" == "$commit" ]]
"$python_bin" -m venv "$destination/venv"
"$destination/venv/bin/python" -m pip install -r "$lock"
"$destination/venv/bin/python" -m pip install --no-deps --no-build-isolation -e "$destination/code"
"$destination/venv/bin/python" -m pip check
"$destination/venv/bin/python" -c 'import torch,numpy; assert torch.__version__.split("+")[0]=="2.7.1"; assert numpy.__version__=="1.26.4"; print("CUDA visible:",torch.cuda.is_available()); print("Deployment does not certify checkpoint resume or authorize training.")'
rm -- "$destination/DEPLOYMENT_INCOMPLETE"
touch "$destination/ENVIRONMENT_INSTALLED"
printf '%s\n' 'Restore the selected snapshot, then run the original runner preflight. No experiment was launched.'
