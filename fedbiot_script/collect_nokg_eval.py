import argparse
import glob
import json
import os
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='results/nokg_eval')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    result_files = glob.glob(
        os.path.join(args.indir, '*', '*', 'seed*', 'result.json'))
    groups = {}
    rows = []
    for path in sorted(result_files):
        with open(path, 'r', encoding='utf-8') as f:
            item = json.load(f)
        parts = os.path.normpath(path).split(os.sep)
        method, dataset, seed_dir = parts[-4], parts[-3], parts[-2]
        row = {
            'method': method,
            'dataset': dataset,
            'seed': seed_dir.replace('seed', ''),
            'metric': item['metric'],
            'value': item['value'],
            'total': item['total'],
            'path': path,
        }
        rows.append(row)
        groups.setdefault((method, dataset, item['metric']), []).append(
            item['value'])

    summary = []
    for (method, dataset, metric), values in sorted(groups.items()):
        summary.append({
            'method': method,
            'dataset': dataset,
            'metric': metric,
            'n': len(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        })

    output = {'runs': rows, 'summary': summary}
    out_path = args.out or os.path.join(args.indir, 'summary.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    for item in summary:
        print('{method}\t{dataset}\t{metric}\tn={n}\t'
              'mean={mean:.4f}\tstd={std:.4f}'.format(**item))


if __name__ == '__main__':
    main()
