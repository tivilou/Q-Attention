"""Read selected private coordinates locally on the experiment server."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--representation', required=True)
    parser.add_argument('--coordinates', required=True)
    args = parser.parse_args()
    from baselines.retacred.rmea_protected_context_trace import read_value
    payload = json.loads((args.run_dir/'sample-trace.json').read_bytes())
    sample = next(s for s in payload['samples'] if s['sample_id'] == args.sample)
    print(json.dumps(read_value(args.run_dir, sample, args.representation,
                               json.loads(args.coordinates)), allow_nan=False))


if __name__ == '__main__':
    main()
