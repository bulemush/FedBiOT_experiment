import argparse
import json
import os


METHODS = ['fedbiot', 'fedot']
DATASETS = ['cwq', 'graphquestions', 'kqapro', 'openbookqa']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1,2,3')
    parser.add_argument('--out', default='fedbiot_script/nokg_manifest.json')
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.replace(' ', ',').split(',') if x]
    runs = []
    for method in METHODS:
        for dataset in DATASETS:
            for seed in seeds:
                runs.append({
                    'method':
                    method,
                    'dataset':
                    dataset,
                    'seed':
                    seed,
                    'cfg':
                    f'fedbiot_script/{method}_nokg/{dataset}.yaml',
                    'checkpoint':
                    f'checkpoints/nokg/{method}/{dataset}_seed{seed}.ckpt',
                    'result':
                    f'results/nokg_eval/{method}/{dataset}/seed{seed}/'
                    'result.json',
                })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'runs': runs}, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(runs)} runs to {args.out}')


if __name__ == '__main__':
    main()
