import argparse
import json
import os
import re
import string
import sys

import torch
import transformers

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from federatedscope.core.cmd_args import parse_client_cfg
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT
from federatedscope.llm.dataloader.dataloader import (
    _first_present,
    _format_openbookqa_records,
    _format_text_qa_records,
    _load_split_records,
    _stringify_answer,
)
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)


QA_DATASETS = {'cwq', 'graphquestions', 'kqapro'}
TASK_TO_DATASET = {
    'cwq': ('cwq', None),
    'graphquestions': ('graphquestions', None),
    'kqapro': ('kqa_pro', None),
    'openbookqa': ('openbookqa_mcqa', 'openbookqa'),
}


def _normalize(text):
    text = str(text).lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def _answers_from_item(item):
    answer = _first_present(item, [
        'answer', 'answers', 'Answer', 'answer_text', 'answerText', 'target',
        'output', 'gold', 'gold_answer'
    ])
    if isinstance(answer, list):
        answers = [_stringify_answer(x) for x in answer]
    else:
        answers = [_stringify_answer(answer)]
    return [x for x in answers if x]


def _question_from_item(item):
    question = _first_present(item, [
        'question', 'Question', 'question_text', 'questionText', 'utterance',
        'machine_question', 'paraphrased_question'
    ])
    return '' if question is None else str(question).strip()


def _select_split(split_records, split):
    candidates = [split]
    if split == 'test':
        candidates.extend(['validation', 'train'])
    elif split == 'validation':
        candidates.extend(['test', 'train'])
    else:
        candidates.extend(['validation', 'test'])
    for name in candidates:
        records = split_records.get(name) or []
        if records:
            return name, records
    return split, []


def _load_eval_records(task, root, split):
    dataset_name, hf_name = TASK_TO_DATASET[task]
    split_records = _load_split_records(root, dataset_name, hf_name=hf_name)
    used_split, records = _select_split(split_records, split)
    return used_split, records


def _build_cfg(cfg_path, seed, checkpoint):
    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(cfg_path)
    opts = []
    if seed is not None:
        opts.extend(['seed', str(seed)])
    if checkpoint:
        opts.extend(['federate.save_to', checkpoint])
    cfg_opt, _ = parse_client_cfg(opts)
    init_cfg.merge_from_list(cfg_opt)
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)
    init_cfg.freeze()
    return init_cfg


def _prompt(instruction):
    return PROMPT_DICT['prompt_no_input'].format_map(
        {'instruction': instruction})


def _first_completion(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)):
        for item in completion:
            text = _first_completion(item)
            if text:
                return text
        return ''
    return str(completion)


def eval_text_qa(bot, task, records, limit, out_jsonl):
    formatted = _format_text_qa_records(
        records, 'kqa_pro' if task == 'kqapro' else task)
    if limit is not None:
        formatted = formatted[:limit]

    correct = 0
    total = 0
    generate_kwargs = {
        'max_new_tokens': bot.config.llm.chat.max_len,
        'num_beams': 1,
        'do_sample': False,
        'temperature': 0.0,
    }

    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for item in formatted:
            completion = bot.generate([_prompt(item['instruction'])],
                                      generate_kwargs=generate_kwargs)
            pred = _first_completion(completion)
            answers = [item['output']]
            norm_pred = _normalize(pred)
            hits = [
                ans for ans in answers
                if _normalize(ans) == norm_pred
                or (_normalize(ans) and _normalize(ans) in norm_pred)
            ]
            is_correct = len(hits) > 0
            correct += int(is_correct)
            total += 1
            f.write(
                json.dumps(
                    {
                        'question': item['instruction'],
                        'answers': answers,
                        'prediction': pred,
                        'correct': is_correct,
                    },
                    ensure_ascii=False) + '\n')
            f.flush()

    return {'metric': 'hit@1', 'correct': correct, 'total': total}


def _openbook_choices(item):
    choices = item.get('choices', {})
    labels = choices.get('label') if isinstance(choices, dict) else None
    texts = choices.get('text') if isinstance(choices, dict) else None
    if labels is None or texts is None:
        if not isinstance(choices, list):
            return []
        labels = [chr(ord('A') + idx) for idx in range(len(choices))]
        texts = [
            _stringify_answer(choice.get('text', choice))
            if isinstance(choice, dict) else _stringify_answer(choice)
            for choice in choices
        ]
    return [(str(label).strip(), str(text).strip())
            for label, text in zip(labels, texts)]


def _score_completion(model, tokenizer, prompt, completion, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + completion,
                         add_special_tokens=False).input_ids
    if len(full_ids) <= len(prompt_ids):
        return float('-inf')
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    score = 0.0
    for pos in range(len(prompt_ids) - 1, len(full_ids) - 1):
        score += float(log_probs[pos, full_ids[pos + 1]].item())
    return score


def eval_openbookqa(bot, records, limit, out_jsonl):
    formatted = _format_openbookqa_records(records)
    if limit is not None:
        records = records[:limit]
        formatted = formatted[:limit]

    device = bot._get_model_input_device()
    correct = 0
    total = 0
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for raw, item in zip(records, formatted):
            choices = _openbook_choices(raw)
            answer = str(_first_present(raw, ['answerKey', 'answer',
                                              'label'])).strip()
            prompt = _prompt(item['instruction'])
            scores = {
                label: _score_completion(bot.model, bot.tokenizer, prompt,
                                         ' ' + label, device)
                for label, _ in choices
            }
            pred = max(scores, key=scores.get) if scores else ''
            is_correct = pred == answer
            correct += int(is_correct)
            total += 1
            f.write(
                json.dumps(
                    {
                        'question': _first_present(raw, [
                            'question_stem', 'question', 'Question'
                        ]),
                        'answer': answer,
                        'prediction': pred,
                        'scores': scores,
                        'correct': is_correct,
                    },
                    ensure_ascii=False) + '\n')
            f.flush()

    return {'metric': 'accuracy', 'correct': correct, 'total': total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--dataset',
                        choices=sorted(QA_DATASETS | {'openbookqa'}),
                        required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = _build_cfg(args.cfg, args.seed, args.checkpoint)
    used_split, records = _load_eval_records(args.dataset, cfg.data.root,
                                             args.split)
    if not records:
        raise RuntimeError(f'No records found for {args.dataset}.')

    bot = FSChatBot(cfg)
    pred_path = os.path.join(args.out, 'predictions.jsonl')
    if args.dataset == 'openbookqa':
        metrics = eval_openbookqa(bot, records, args.limit, pred_path)
    else:
        metrics = eval_text_qa(bot, args.dataset, records, args.limit,
                               pred_path)

    total = metrics['total']
    value = float(metrics['correct']) / total if total else 0.0
    result = {
        'dataset': args.dataset,
        'split': used_split,
        'cfg': args.cfg,
        'checkpoint': cfg.federate.save_to,
        'seed': cfg.seed,
        'metric': metrics['metric'],
        'value': value,
        'correct': metrics['correct'],
        'total': total,
        'predictions': pred_path,
    }
    with open(os.path.join(args.out, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
