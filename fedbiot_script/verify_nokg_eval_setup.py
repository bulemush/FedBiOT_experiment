import glob
import os
import sys

import yaml


METHODS = ['fedbiot', 'fedot']
DATASETS = ['cwq', 'graphquestions', 'kqapro', 'openbookqa']


def _get(cfg, path):
    node = cfg
    for key in path.split('.'):
        node = node[key]
    return node


def main():
    ok = True
    expected_files = [
        'fedbiot_script/eval_nokg_tasks.py',
        'fedbiot_script/eval_nokg_all.sh',
        'fedbiot_script/collect_nokg_eval.py',
        'fedbiot_script/check_nokg_results.py',
        'fedbiot_script/generate_nokg_manifest.py',
        'fedbiot_script/preflight_nokg_eval.py',
    ]
    for path in expected_files:
        if not os.path.exists(path):
            print(f'Missing file: {path}')
            ok = False

    for method in METHODS:
        for dataset in DATASETS:
            path = f'fedbiot_script/{method}_nokg/{dataset}.yaml'
            if not os.path.exists(path):
                print(f'Missing YAML: {path}')
                ok = False
                continue
            with open(path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            checks = [
                ('llm.kg_adapter.use', False),
                ('llm.offsite_tuning.use', True),
                ('llm.offsite_tuning.emu_align.use', True),
                ('dataloader.batch_size', 2),
                ('federate.total_round_num', 200),
                ('llm.model_parallel.use', True),
                ('llm.model_parallel.device_map', 'balanced_layers'),
                ('llm.model_parallel.max_memory.0', '14GiB'),
                ('llm.model_parallel.max_memory.1', '22GiB'),
            ]
            for key, expected in checks:
                actual = _get(cfg, key)
                if actual != expected:
                    print(f'{path}: {key}={actual}, expected {expected}')
                    ok = False
            initial_only = _get(cfg, 'llm.offsite_tuning.emu_align.initial_only')
            if method == 'fedbiot' and initial_only is not True:
                print(f'{path}: FedBiOT must set initial_only=True')
                ok = False
            if method == 'fedot' and initial_only is not False:
                print(f'{path}: FedOT must set initial_only=False')
                ok = False
            data_type = _get(cfg, 'data.type')
            if 'kg' in str(data_type).lower() and dataset != 'kqapro':
                print(f'{path}: suspicious data.type={data_type}')
                ok = False

    yaml_count = len(
        glob.glob('fedbiot_script/fedbiot_nokg/*.yaml')) + len(
            glob.glob('fedbiot_script/fedot_nokg/*.yaml'))
    if yaml_count != 8:
        print(f'Expected 8 NoKG YAML files, found {yaml_count}')
        ok = False

    if ok:
        print('NoKG setup verification passed.')
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
