import os
import gzip
import glob
import json
import pickle
import random
import logging
import torch
import datasets
import transformers
from transformers import GenerationConfig
from tqdm import tqdm

from dataclasses import dataclass
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset, PROMPT_DICT
from federatedscope.core.data.utils import download_url
from federatedscope.llm.model.model_builder import get_llm

logger = logging.getLogger(__name__)


def _parse_model_type(model_type):
    if '@' in model_type:
        return model_type.split('@', 1)
    return model_type, 'huggingface_llm'


@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class LLMRewardCollator():
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        win_data, lose_data = \
            tuple([instance[key] for instance in instances]
                  for key in ("win_data", "lose_data"))

        # Form a concatenated dataset
        concat_data = win_data + lose_data
        concat_data_collator = LLMDataCollator(tokenizer=self.tokenizer)
        concat_data_dict = concat_data_collator(concat_data)

        return dict(
            win_input_ids=concat_data_dict["input_ids"][:len(win_data)],
            win_labels=concat_data_dict["labels"][:len(win_data)],
            win_attention_mask=concat_data_dict["attention_mask"]
            [:len(win_data)],
            lose_input_ids=concat_data_dict["input_ids"][len(win_data):],
            lose_labels=concat_data_dict["labels"][len(win_data):],
            lose_attention_mask=concat_data_dict["attention_mask"]
            [len(win_data):])


class Generator:
    """Generate the output from the original LLM model"""
    def __init__(self, config, tokenizer, generate_kwargs=None):
        self.device = f'cuda:{config.device}'

        # self.model = get_llm(config).to(self.device)
        self.add_special_tokens = True
        self.tokenizer = tokenizer

        if generate_kwargs is not None:
            self.generate_kwargs = generate_kwargs
        else:
            self.generate_kwargs = {
                'temperature': 0.0,
                'top_p': 1.0,
                'max_new_tokens': config.llm.chat.max_len,
            }
            self.generate_kwargs = {
                'max_new_tokens': config.llm.chat.max_len,
                'num_beams': 4,
                'no_repeat_ngram_size': 2,
                'early_stopping': True,
                'temperature': 0.0
            }

    def __call__(self, input_text, model):
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).to(self.device)
        response = model.generate(input_ids=input_ids, **self.generate_kwargs)
        response_tokens = \
            self.tokenizer.decode(response[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
        if response_tokens == "":
            print('INPUT:', input_text)
            print(len(input_text))
            print('===============================\n\n')
        return response_tokens


def get_tokenizer(model_name, cache_dir, tok_len=128, padding_side="right"):
    from transformers import AutoTokenizer, GPT2Tokenizer

    if model_name == 'CarperAI/openai_summarize_tldr_sft':
        tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2',
            cache_dir=cache_dir,
            model_max_length=tok_len,
            padding_side=padding_side,
            use_fast=False,
        )
    else:
        local_files_only = os.path.isdir(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            model_max_length=tok_len,
            padding_side=padding_side,
            use_fast=False,
            local_files_only=local_files_only,
        )

    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    return tokenizer, num_new_tokens


class new_dict(dict):
    """
    Create a new_dict to ensure we can access the dictionary with
    one bracket only
    e.g., dict[key1][key2][key3] --> dict[key1.key2.key3]
    """
    def __init__(self, init_dict: dict):
        self.dict = init_dict
        for key in self.dict.keys():
            if type(self.dict[key]) is dict:
                self.dict[key] = new_dict(self.dict[key])
            if type(self.dict[key]) is list:
                self.dict[key] = new_dict({
                    str(idx): value
                    for idx, value in enumerate(self.dict[key])
                })

    def __getitem__(self, __key):
        try:
            if '.' not in __key:
                return self.dict[__key]
            else:
                prefix, suffix = __key.split('.', 1)
                return self.dict[prefix][suffix]
        except:
            return None

    def __setitem__(self, __key, __value):
        if type(__value) is dict:
            self.dict[__key] = new_dict(__value)
        else:
            if '.' not in __key:
                self.dict[__key] = __value
            else:
                prefix, suffix = __key.split('.', 1)
                if prefix not in self:
                    self.dict[prefix] = new_dict({})
                self.dict[prefix][suffix] = __value


def load_json(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category',
              **kwargs):
    # Format: [{'instruction': ..., 'input': ..., 'output':...}]
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Replace key
    new_list_data_dict = []
    for item in list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output] if output in item else None,
            category=item[category] if category in item else None)
        for key, value in kwargs.items():
            new_item[key] = item[value]
        new_list_data_dict.append(new_item)
    return new_list_data_dict


def load_jsonl(file_path,
               is_gzip=False,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               **kwargs):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = new_dict(json.loads(line))
            new_item = dict(instruction=item[instruction],
                            input=item[input],
                            output=item[output],
                            category=item[category])
            for key, value in kwargs.items():
                new_item[key] = item[value]
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def load_jsonls(file_paths,
                is_gzip=False,
                instruction='instruction',
                input='input',
                output='output',
                category='category',
                **kwargs):
    list_data_dict = []
    for path in file_paths:
        list_data_dict.extend(
            load_jsonl(path, is_gzip, instruction, input, output, category,
                       **kwargs))
    return list_data_dict


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
                            'OpenBookQA', 'openbookQA']
    }
    names = aliases.get(dataset_name, [dataset_name])
    return [os.path.join(root, name) for name in names]


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
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError('Reading parquet data requires pandas.') \
                from error
        return pd.read_parquet(path).to_dict('records')
    raise ValueError(f'Unsupported data file type: {path}')


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


def _load_split_records(root,
                        dataset_name,
                        hf_name=None,
                        hf_config=None,
                        data_args=None):
    split_files = _split_files_from_args(data_args)
    if split_files:
        return _read_split_files(root, split_files)

    split_aliases = {
        'train': ['train'],
        'validation': ['validation', 'valid', 'val', 'dev'],
        'test': ['test']
    }
    extensions = ['jsonl', 'json', 'parquet']

    for data_dir in _candidate_dataset_dirs(root, dataset_name):
        if not os.path.exists(data_dir):
            continue

        try:
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

        single_files = []
        for ext in extensions:
            single_files.extend(glob.glob(os.path.join(data_dir, f'*.{ext}')))
        if single_files:
            records = []
            for path in sorted(single_files):
                records.extend(_read_records_from_file(path))
            grouped = {'train': [], 'validation': [], 'test': []}
            for item in records:
                split = str(_first_present(item, ['split', 'set'], 'train'))
                split = {'valid': 'validation',
                         'val': 'validation',
                         'dev': 'validation'}.get(split.lower(),
                                                  split.lower())
                grouped.setdefault(split, []).append(item)
            if any(grouped.values()):
                return grouped

    if hf_name is None:
        raise FileNotFoundError(
            f'Cannot find local data for {dataset_name} under {root}.')

    if hf_config is None:
        hf_dataset = datasets.load_dataset(hf_name)
    else:
        hf_dataset = datasets.load_dataset(hf_name, hf_config)
    return {
        'train': _records_to_list(hf_dataset.get('train')),
        'validation': _records_to_list(
            hf_dataset.get('validation') or hf_dataset.get('valid')
            or hf_dataset.get('val') or hf_dataset.get('dev')),
        'test': _records_to_list(hf_dataset.get('test')),
    }


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
        question = '' if question is None else str(question).strip()
        answer = _stringify_answer(answer)
        if question and answer:
            formatted.append(
                dict(instruction=question,
                     input=None,
                     output=answer,
                     category=category))
    return formatted


def _format_openbookqa_records(records):
    formatted = []
    for item in records:
        question = _first_present(item,
                                  ['question_stem', 'question', 'Question'])
        choices = item.get('choices', {})
        answer = _first_present(item, ['answerKey', 'answer', 'label'])
        if not question or not choices or answer in [None, '']:
            continue

        labels = choices.get('label') if isinstance(choices, dict) else None
        texts = choices.get('text') if isinstance(choices, dict) else None
        if labels is None or texts is None:
            if isinstance(choices, list):
                labels = [chr(ord('A') + idx) for idx in range(len(choices))]
                texts = [
                    _stringify_answer(choice.get('text', choice))
                    if isinstance(choice, dict) else _stringify_answer(choice)
                    for choice in choices
                ]
            else:
                continue

        options = []
        for label, text in zip(labels, texts):
            options.append(f'{str(label).strip()}. {str(text).strip()}')
        instruction = (
            f'Question: {str(question).strip()}\n'
            f'Choices:\n' + '\n'.join(options) +
            '\nAnswer with the option letter.')
        formatted.append(
            dict(instruction=instruction,
                 input=None,
                 output=str(answer).strip(),
                 category='openbookqa'))
    return formatted


def _build_llm_split_dataset(split_records, tokenizer, formatter):
    train_records = split_records.get('train') or []
    val_records = split_records.get('validation') or []
    test_records = split_records.get('test') or []

    if val_records or test_records:
        if not val_records:
            val_size = max(1, int(0.01 * len(train_records)))
            val_records, train_records = train_records[:val_size], \
                train_records[val_size:]
        if not test_records:
            test_size = max(1, int(0.01 * len(train_records)))
            test_records, train_records = train_records[:test_size], \
                train_records[test_size:]
        return (LLMDataset(formatter(train_records), tokenizer),
                LLMDataset(formatter(val_records), tokenizer),
                LLMDataset(formatter(test_records), tokenizer))

    return LLMDataset(formatter(train_records), tokenizer)


def load_llm_dataset(config=None, **kwargs):
    model_name, _ = _parse_model_type(config.model.type)
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)

    dataset_name, _ = config.data.type.split('@')

    if dataset_name.endswith('.json'):
        fp = os.path.join(config.data.root, dataset_name)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.endswith('.jsonl'):
        fp = os.path.join(config.data.root, dataset_name)
        list_data_dict = load_jsonl(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'alpaca':
        fp = os.path.join(config.data.root, 'alpaca_data.json')
        download_url(
            'https://raw.githubusercontent.com/tatsu-lab'
            '/stanford_alpaca/'
            '761dc5bfbdeeffa89b8bff5d038781a4055f796a/'
            'alpaca_data.json', config.data.root)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'alpaca_cleaned':
        fp = os.path.join(config.data.root, 'alpaca_data_cleaned.json')
        download_url(
            'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/'
            'a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/'
            'alpaca_data_cleaned.json', config.data.root)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'dolly-15k':
        fp = os.path.join(config.data.root, 'databricks-dolly-15k.jsonl')
        download_url(
            'https://raw.githubusercontent.com/databrickslabs'
            '/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64'
            '/data/databricks-dolly-15k.jsonl', config.data.root)
        list_data_dict = load_jsonl(fp,
                                    instruction='instruction',
                                    input='context',
                                    output='response',
                                    category='category')
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'gsm8k':
        fp = os.path.join(config.data.root, 'gsm8k_train.jsonl')
        if not os.path.exists(fp):
            download_url(
                'https://raw.githubusercontent.com/openai/grade-school-math'
                '/3101c7d5072418e28b9008a6636bde82a006892c/'
                'grade_school_math/data/train.jsonl', config.data.root)
            os.rename(os.path.join(config.data.root, 'train.jsonl'), fp)
        list_data_dict = load_jsonl(fp,
                                    instruction='question',
                                    output='answer')
        for i in range(len(list_data_dict)):
            list_data_dict[i]['output'] = \
                list_data_dict[i]['output'].replace('####', 'The answer is')
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'code_search_net':
        from federatedscope.llm.dataset.code_search_net import \
            CSN_FILE_NUM_DICT

        list_data_dict = []
        logger.info('Loading code search net data file...')
        try:
            for language in tqdm(CSN_FILE_NUM_DICT.keys()):
                sub_list_data_dict = []
                for file_index in range(CSN_FILE_NUM_DICT[language]['train']):
                    fp = \
                        os.path.join(config.data.root, language,
                                     'final', 'jsonl', 'train',
                                     f'{language}_train_{file_index}.jsonl.gz')
                    tmp_list_data_dict = load_jsonl(
                        fp,
                        instruction='docstring',
                        input='language',
                        output='code',
                        category='language',
                        is_gzip=True,
                    )
                    sub_list_data_dict += tmp_list_data_dict
                # Subsample
                raw_size = len(sub_list_data_dict)
                num_subsample = int(raw_size * config.data.subsample)
                list_data_dict += random.sample(sub_list_data_dict,
                                                num_subsample)
                logger.info(f"Subsample "
                            f"{sub_list_data_dict[0]['category']} with "
                            f"rate {config.data.subsample}: "
                            f"the sample size is # {num_subsample} "
                            f"(the raw size is {raw_size}).")
            # Modify instruction with specific language
            for sample in list_data_dict:
                sample['instruction'] = \
                    sample['category'] + ' ' + sample['instruction']
        except FileNotFoundError:
            raise FileNotFoundError(
                'Data not found! Please run `python '
                'federatedscope/llm/dataset/code_search_net.py` '
                'to download data.')
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'rosetta_alpaca':
        fp = os.path.join(config.data.root, 'rosetta_alpaca.json')
        download_url(
            'https://raw.githubusercontent.com/'
            'sahil280114/codealpaca/'
            'd269da106a579a623a654529b3cb91b5dfa9c72f/'
            'data/rosetta_alpaca.json', config.data.root)
        list_data_dict = load_json(fp,
                                   instruction='instruction',
                                   input='input',
                                   output='output',
                                   category='input')

        # Remove 'x86-64 Assembl' if splitter is `meta` due to the number of
        # samples is too small.
        if config.data.splitter == 'meta':
            list_data_dict = [
                i for i in list_data_dict if i['category'] != 'X86-64 Assembly'
            ]
        # Manually remove \u00a0
        for i in range(len(list_data_dict)):
            list_data_dict[i]['output'] = \
                list_data_dict[i]['output'].replace('\u00a0', '')
            list_data_dict[i]['instruction'] = \
                list_data_dict[i]['instruction'].replace('\u00a0', '')
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'cwq':
        split_records = _load_split_records(config.data.root,
                                            'cwq',
                                            hf_name=None,
                                            data_args=config.data.args)
        dataset = _build_llm_split_dataset(
            split_records,
            tokenizer,
            lambda records: _format_text_qa_records(records, 'cwq'))

    elif dataset_name.lower() == 'graphquestions':
        split_records = _load_split_records(config.data.root,
                                            'graphquestions',
                                            hf_name=None,
                                            data_args=config.data.args)
        dataset = _build_llm_split_dataset(
            split_records,
            tokenizer,
            lambda records: _format_text_qa_records(records,
                                                    'graphquestions'))

    elif dataset_name.lower() in ['kqa_pro', 'kqapro']:
        split_records = _load_split_records(config.data.root,
                                            'kqa_pro',
                                            hf_name=None,
                                            data_args=config.data.args)
        dataset = _build_llm_split_dataset(
            split_records,
            tokenizer,
            lambda records: _format_text_qa_records(records, 'kqa_pro'))

    elif dataset_name.lower() == 'openbookqa_mcqa':
        split_records = _load_split_records(config.data.root,
                                            'openbookqa_mcqa',
                                            hf_name=None,
                                            data_args=config.data.args)
        dataset = _build_llm_split_dataset(split_records, tokenizer,
                                           _format_openbookqa_records)

    elif dataset_name.lower() == 'offsite_tuning':
        from federatedscope.llm.dataloader.offsite_tuning_dataset import \
            PIQA, HellaSwag, OpenBookQA, ARC, SciQ, WebQs, RACE
        # list of dataset
        task_dict = {
            "piqa": PIQA(),
            "hellaswag": HellaSwag(),
            "openbookqa": OpenBookQA(),
            "arc_easy": ARC(name='ARC-Easy'),
            "arc_challenge": ARC(name='ARC-Challenge'),
            "sciq": SciQ(),
            "web_questions": WebQs(),
            "race": RACE(),
        }
        # concat these datasets
        list_train_dict, list_val_dict, list_test_dict = [], [], []
        for dataset in task_dict.values():
            list_train_dict += dataset.get_data_dict(label='train')
            list_val_dict += dataset.get_data_dict(label='validation')
            list_test_dict += dataset.get_data_dict(label='test')

        train_dataset = LLMDataset(list_train_dict,
                                   tokenizer,
                                   prompt_no_input='{context}',
                                   prompt_input='{context}',
                                   output_tag='target')
        val_dataset = LLMDataset(list_val_dict,
                                 tokenizer,
                                 prompt_no_input='{context}',
                                 prompt_input='{context}',
                                 output_tag='target')
        test_dataset = LLMDataset(list_test_dict,
                                  tokenizer,
                                  prompt_no_input='{context}',
                                  prompt_input='{context}',
                                  output_tag='target')

        dataset = (train_dataset, val_dataset, test_dataset)

    elif dataset_name.lower() == 'wikitext-2':
        pass

    elif dataset_name.lower() == 'reddit-tldr':
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_human_annotated_dataset
        dataset = load_human_annotated_dataset(config.data.root, tokenizer)

    elif dataset_name.lower() == 'reddit-tldr-finetuning':
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_human_finetuning_dataset
        data_root = os.path.join(config.data.root, 'reddit-tldr-comparison')
        dataset = load_human_finetuning_dataset(data_root,
                                                tokenizer,
                                                max_num_test=1000)

    elif dataset_name.lower() == 'alpaca_reddit-tldr-finetuning':
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_human_finetuning_dataset
        from torch.utils.data import ConcatDataset
        data_root = os.path.join(config.data.root, 'reddit-tldr-comparison')
        train_dataset, val_dataset, test_dataset = \
            load_human_finetuning_dataset(data_root, tokenizer,
                                          max_num_test=1000)
        fp = os.path.join(config.data.root, 'alpaca_data.json')
        download_url(
            'https://raw.githubusercontent.com/tatsu-lab'
            '/stanford_alpaca/'
            '761dc5bfbdeeffa89b8bff5d038781a4055f796a/'
            'alpaca_data.json', config.data.root)
        list_data_dict = load_json(fp)
        alpaca_dataset = LLMDataset(list_data_dict, tokenizer)
        train_dataset = ConcatDataset([train_dataset, alpaca_dataset])
        dataset = (train_dataset, val_dataset, test_dataset)

    elif dataset_name.lower() == 'reddit-tldr-rlhf':
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_human_finetuning_dataset
        data_root = os.path.join(config.data.root, 'reddit-tldr-comparison')
        dataset = load_human_finetuning_dataset(data_root,
                                                tokenizer,
                                                rlhf=True,
                                                max_num_test=1000)

    elif dataset_name.lower() == 'reddit-tldr-comparison':
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_comparison_dataset
        data_root = os.path.join(config.data.root, 'reddit-tldr-comparison')
        dataset = load_comparison_dataset(data_root,
                                          tokenizer,
                                          max_num_test=1000)

    elif dataset_name.lower() == 'reddit-tldr-best':
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_best_dataset
        data_root = os.path.join(config.data.root, 'reddit-tldr-comparison')
        dataset = load_best_dataset(data_root, tokenizer, max_num_test=1000)

    elif dataset_name.lower() == 'reddit-tldr-comparison-choice':
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_comparison_dataset_by_choice
        data_root = os.path.join(config.data.root, 'reddit-tldr-comparison')
        dataset = load_comparison_dataset_by_choice(data_root,
                                                    tokenizer,
                                                    max_num_test=1000)

    elif dataset_name.lower() == 'shp-comparison':
        from federatedscope.llm.dataloader.shp import \
            load_shp_cmp_dataset_by_choice
        data_root = os.path.join(config.data.root, 'shp')
        dataset = load_shp_cmp_dataset_by_choice(data_root,
                                                 tokenizer,
                                                 config,
                                                 max_num_test=1000)

    elif dataset_name.lower() == 'shp-best':
        from federatedscope.llm.dataloader.shp import \
            load_shp_best_dataset
        data_root = os.path.join(config.data.root, 'shp')
        dataset = load_shp_best_dataset(data_root,
                                        tokenizer,
                                        config,
                                        max_num_test=1000)

    elif dataset_name.lower() == 'shp-comparison-pairs':
        from federatedscope.llm.dataloader.shp import \
            load_comparison_dataset
        data_root = os.path.join(config.data.root, 'shp')
        dataset = load_comparison_dataset(data_root,
                                          tokenizer,
                                          config,
                                          max_num_test=1000)

    else:
        raise ValueError(f'Not support data type {dataset_name}.')

    return dataset, config


if __name__ == '__main__':
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.cmd_args import parse_args, parse_client_cfg
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.auxiliaries.logging import update_logger

    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    load_llm_dataset(init_cfg)
