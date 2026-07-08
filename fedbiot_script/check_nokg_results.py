import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='results/nokg_eval/summary.json')
    parser.add_argument('--expected-seeds', type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f'Missing summary file: {args.summary}', file=sys.stderr)
        return 1

    with open(args.summary, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    expected = {(m, d) for m in ['fedbiot', 'fedot']
                for d in ['cwq', 'graphquestions', 'kqapro', 'openbookqa']}
    ok = True
    seen = {}
    for item in summary.get('summary', []):
        key = (item['method'], item['dataset'])
        seen[key] = item
        if item['n'] != args.expected_seeds:
            ok = False
            print(f'Incomplete seeds for {key}: {item["n"]}')
        if not (0.0 <= item['mean'] <= 1.0):
            ok = False
            print(f'Invalid metric range for {key}: {item["mean"]}')

    missing = expected - set(seen)
    if missing:
        ok = False
        for key in sorted(missing):
            print(f'Missing result group: {key}')

    if ok:
        print('NoKG evaluation results look complete.')
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
