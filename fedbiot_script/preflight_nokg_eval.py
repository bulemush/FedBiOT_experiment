import argparse
import glob
import json
import os
import sys

import yaml


def _first_present(item, keys, default=None):
    for key in keys:
        if key in item and item[key] not in [None, '']:
            return item[key]
    return default


def _stringify_answer(answer):
    if answer is None:
        return ''
    if isinstance(answer, list):
        answers = [_stringify_answer(x) for x in answer]
        answers = [x for x in answers if x]
        return answers[0] if answers else ''
    if isinstance(answer, dict):
        value = _first_present(answer, [
            'answer', 'text', 'name', 'label', 'answer_text',
            'answerArgument', 'argument'
        ])
        return _stringify_answer(value)
    return str(answer).strip()


def _candidate_dataset_dirs(root, dataset_name):
    aliases = {
        'cwq': ['cwq', 'CWQ', 'complexwebquestions',
                'ComplexWebQuestions'],
        'graphquestions': ['graphquestions', 'GraphQuestions',
                           'graph_questions', 'Graph_Questions'],
        'kqa_pro': ['kqa_pro', 'kqapro', 'KQAPro', 'KQA_PRO', 'kqa-pro'],
        'openbookqa_mcqa': ['openbookQA/main', 'openbookqa/main',
                            'OpenBookQA/main', 'openbookqa',
                            'OpenBookQA', 'openbookQA'],
    }
    return [os.path.join(root, name)
            for name in aliases.get(dataset_name, [dataset_name])]


def _read_records_from_file(path):
    lower_path = path.lower()
    if lower_path.endswith('.jsonl'):
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if lower_path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ['data', 'examples', 'questions', 'records']:
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        return data
    if lower_path.endswith('.parquet'):
        import pandas as pd
        return pd.read_parquet(path).to_dict('records')
    return []


def _normalize_path_list(paths):
    if paths is None:
        return []
    if isinstance(paths, (list, tuple)):
        return list(paths)
    return [paths]


def _read_split_files(root, split_files):
    split_records = {}
    for split, paths in split_files.items():
        records = []
        for raw_path in _normalize_path_list(paths):
            path = os.fspath(raw_path)
            if not os.path.isabs(path):
                path = os.path.join(root, path)
            matched_paths = glob.glob(path)
            if not matched_paths and os.path.exists(path):
                matched_paths = [path]
            if not matched_paths:
                raise FileNotFoundError(f'Cannot find data file: {path}')
            for matched_path in sorted(matched_paths):
                records.extend(_read_records_from_file(matched_path))
        split_records[split] = records
    return split_records


def _split_files_from_args(data_args):
    if not data_args:
        return {}
    args = data_args[0] if isinstance(data_args, list) else data_args
    if not hasattr(args, 'get'):
        return {}
    key_groups = {
        'train': ['train_file', 'train_files'],
        'validation': [
            'validation_file', 'validation_files', 'val_file', 'val_files',
            'valid_file', 'valid_files', 'dev_file', 'dev_files'
        ],
        'test': ['test_file', 'test_files'],
    }
    split_files = {}
    for split, keys in key_groups.items():
        files = []
        for key in keys:
            files.extend(_normalize_path_list(args.get(key)))
        if files:
            split_files[split] = files
    return split_files


def _default_split_patterns(dataset_name):
    return {
        'cwq': {
            'train': ['ComplexWebQuestions_train.*'],
            'validation': [
                'ComplexWebQuestions_dev.*',
                'ComplexWebQuestions_validation.*',
            ],
            'test': ['ComplexWebQuestions_test.*'],
        },
        'graphquestions': {
            'train': ['graphquestions.training.*', 'graphquestions_train.*'],
            'validation': [
                'graphquestions.validation.*', 'graphquestions.dev.*',
                'graphquestions_val.*'
            ],
            'test': ['graphquestions.testing.*', 'graphquestions_test.*'],
        },
        'kqa_pro': {
            'train': ['train.*'],
            'validation': ['val.*', 'valid.*', 'validation.*', 'dev.*'],
            'test': ['test.*'],
        },
        'openbookqa_mcqa': {
            'train': ['train-*.*', 'train.*'],
            'validation': ['validation-*.*', 'validation.*', 'valid.*'],
            'test': ['test-*.*', 'test.*'],
        },
    }.get(dataset_name, {})


def _records_to_list(split):
    if split is None:
        return []
    if hasattr(split, 'to_list'):
        return split.to_list()
    return [dict(item) for item in split]


def _load_split_records(root, dataset_name, hf_name=None, data_args=None):
    split_files = _split_files_from_args(data_args)
    if split_files:
        return _read_split_files(root, split_files)

    split_aliases = {
        'train': ['train'],
        'validation': ['validation', 'valid', 'val', 'dev'],
        'test': ['test'],
    }
    extensions = ['jsonl', 'json', 'parquet']
    for data_dir in _candidate_dataset_dirs(root, dataset_name):
        if not os.path.exists(data_dir):
            continue
        try:
            import datasets
            disk_dataset = datasets.load_from_disk(data_dir)
            if hasattr(disk_dataset, 'keys'):
                return {
                    'train': _records_to_list(disk_dataset.get('train')),
                    'validation': _records_to_list(
                        disk_dataset.get('validation')
                        or disk_dataset.get('valid')
                        or disk_dataset.get('val')
                        or disk_dataset.get('dev')),
                    'test': _records_to_list(disk_dataset.get('test')),
                }
        except Exception:
            pass

        split_records = {}
        dataset_patterns = _default_split_patterns(dataset_name)
        for split, patterns in dataset_patterns.items():
            files = []
            for pattern in patterns:
                files.extend(glob.glob(os.path.join(data_dir, pattern)))
            if files:
                records = []
                for path in sorted(files):
                    records.extend(_read_records_from_file(path))
                split_records[split] = records

        for split, aliases in split_aliases.items():
            if split in split_records:
                continue
            files = []
            for alias in aliases:
                for ext in extensions:
                    files.extend(
                        glob.glob(os.path.join(data_dir, f'{alias}.{ext}')))
                    files.extend(
                        glob.glob(os.path.join(data_dir, alias,
                                               f'*.{ext}')))
            if files:
                records = []
                for path in sorted(files):
                    records.extend(_read_records_from_file(path))
                split_records[split] = records
        if split_records:
            return split_records

    if hf_name:
        import datasets
        hf_dataset = datasets.load_dataset(hf_name)
        return {
            'train': _records_to_list(hf_dataset.get('train')),
            'validation': _records_to_list(
                hf_dataset.get('validation') or hf_dataset.get('valid')
                or hf_dataset.get('val') or hf_dataset.get('dev')),
            'test': _records_to_list(hf_dataset.get('test')),
        }
    raise FileNotFoundError(f'Cannot find local data for {dataset_name}')


def _format_text_qa_records(records, category):
    formatted = []
    for item in records:
        question = _first_present(item, [
            'question', 'Question', 'question_text', 'questionText',
            'utterance', 'machine_question', 'paraphrased_question'
        ])
        answer = _first_present(item, [
            'answer', 'answers', 'Answer', 'answer_text', 'answerText',
            'target', 'output', 'gold', 'gold_answer'
        ])
        if question and _stringify_answer(answer):
            formatted.append({'category': category})
    return formatted


def _format_openbookqa_records(records):
    formatted = []
    for item in records:
        question = _first_present(item,
                                  ['question_stem', 'question', 'Question'])
        choices = item.get('choices', {})
        answer = _first_present(item, ['answerKey', 'answer', 'label'])
        if question and choices and answer not in [None, '']:
            formatted.append({'category': 'openbookqa'})
    return formatted


METHODS = ['fedbiot', 'fedot']
DATASETS = ['cwq', 'graphquestions', 'kqapro', 'openbookqa']
DATASET_LOADERS = {
    'cwq': ('cwq', None, lambda records: _format_text_qa_records(records,
                                                                 'cwq')),
    'graphquestions':
    ('graphquestions', None,
     lambda records: _format_text_qa_records(records, 'graphquestions')),
    'kqapro': ('kqa_pro', None,
               lambda records: _format_text_qa_records(records, 'kqa_pro')),
    'openbookqa': ('openbookqa_mcqa', None, _format_openbookqa_records),
}


def _check_data(dataset, root, data_args=None):
    dataset_name, hf_name, formatter = DATASET_LOADERS[dataset]
    split_records = _load_split_records(root,
                                        dataset_name,
                                        hf_name=hf_name,
                                        data_args=data_args)
    counts = {}
    ok = True
    for split in ['train', 'validation', 'test']:
        records = split_records.get(split) or []
        formatted = formatter(records)
        counts[split] = len(formatted)
    if counts['train'] == 0:
        ok = False
    if counts['validation'] == 0 and counts['test'] == 0:
        ok = False
    return ok, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-checkpoints', action='store_true')
    parser.add_argument('--seeds', default='1,2,3')
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.replace(' ', ',').split(',') if x]
    ok = True
    checked_data = {}

    for method in METHODS:
        for dataset in DATASETS:
            cfg_path = f'fedbiot_script/{method}_nokg/{dataset}.yaml'
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            if dataset not in checked_data:
                try:
                    data_ok, counts = _check_data(
                        dataset, cfg['data']['root'], cfg['data'].get('args'))
                    checked_data[dataset] = (data_ok, counts)
                except Exception as error:
                    data_ok, counts = False, {'error': str(error)}
                    checked_data[dataset] = (data_ok, counts)
                print(f'[DATA] {dataset}: {counts}')
                ok = ok and data_ok

            if args.skip_checkpoints:
                continue
            for seed in seeds:
                ckpt = f'checkpoints/nokg/{method}/{dataset}_seed{seed}.ckpt'
                if not os.path.exists(ckpt):
                    print(f'[CKPT MISSING] {ckpt}')
                    ok = False

    if ok:
        print('NoKG preflight passed.')
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
