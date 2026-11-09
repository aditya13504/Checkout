"""
Builds showcase/data.js from the project's real outputs, for the interactive showcase page.

Run from the project root:  py -3.10 showcase/build_data.py
Nothing here is invented: every number comes from a results, metrics or data file in the repo.
"""
import csv
import itertools
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'report'))
from make_report import bootstrap_mean_ci  # noqa: E402  (the same intervals as the report)

KAGGLE = os.path.join(ROOT, 'kaggle_outputs', 'outputs_full')  # the reported run: all 13 agents
NOTEBOOK_LOG = os.path.join(ROOT, 'kaggle_outputs', 'notebook530ac845ac.log')
LOCAL = os.path.join(ROOT, 'results_cpu')  # validation runs of the value shopper, made on a laptop
HUMAN_INSTRUCTIONS = os.path.join(ROOT, 'baseline_models', 'data', 'items_human_ins.json')

# (key, file, label, group, how actions were chosen)
AGENTS = [
    ('rule_softmax', KAGGLE, 'Rule baseline', 'baseline', 'Fixed rule'),
    ('rl_scratch_softmax', KAGGLE, 'RL from pretrained BERT', 'trained', 'Sampled'),
    ('rl_scratch_greedy', KAGGLE, 'RL from pretrained BERT, greedy', 'trained', 'Greedy'),
    ('il_seed0_softmax', KAGGLE, 'IL', 'trained', 'Sampled'),
    ('il_seed1_softmax', KAGGLE, 'IL, second seed', 'trained', 'Sampled'),
    ('il_seed0_greedy', KAGGLE, 'IL, greedy', 'trained', 'Greedy'),
    ('il_seed0_instruction_softmax', KAGGLE, 'IL, instruction as query', 'trained', 'Sampled'),
    ('il_rl_softmax', KAGGLE, 'IL+RL', 'trained', 'Sampled'),
    ('il_rl_greedy', KAGGLE, 'IL+RL, greedy', 'trained', 'Greedy'),
    ('rule_assist', KAGGLE, 'Rule baseline + assist', 'added', 'Fixed rule'),
    ('il_rl_assist_greedy', KAGGLE, 'IL+RL, greedy + assist', 'added', 'Greedy'),
    ('careful_shopper', KAGGLE, 'Careful shopper', 'added', 'Hand-written score'),
    ('value_shopper', KAGGLE, 'Value shopper', 'added', 'Learned value'),
]


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def agents():
    out, per_goal = [], {}
    for key, folder, label, group, choice in AGENTS:
        run = read_json(os.path.join(folder, 'results', f'{key}.json'))
        rows = sorted(run['results'], key=lambda r: r['goal_idx'])
        assert [r['goal_idx'] for r in rows] == list(range(500)), key
        scores = np.array([r['score'] for r in rows], dtype=float)
        per_goal[key] = scores
        low, high = bootstrap_mean_ci(scores)
        s = run['summary']
        out.append({
            'key': key, 'label': label, 'group': group, 'choice': choice,
            'score': round(s['score'], 1), 'low': round(low, 1), 'high': round(high, 1),
            'success': round(s['success_rate'], 1), 'bought': round(s['bought_rate'], 1),
            'steps': round(s['steps'], 1),
            'parts': {k: round(s[k], 1) for k in ('r_type', 'r_att', 'r_option', 'r_price')},
            # per goal: score, type, attributes, options, price, steps (compact lists)
            'goals': [[round(r['score'], 1), round(r['r_type'], 2), round(r['r_att'], 2), round(r['r_option'], 2),
                       round(float(r['r_price']), 2), r['steps']] for r in rows],
        })
    return out, per_goal


def paired(per_goal):
    """Mean difference per goal (row minus column) with the report's bootstrap interval."""
    out = {}
    for a, b in itertools.permutations(per_goal, 2):
        diff = per_goal[a] - per_goal[b]
        low, high = bootstrap_mean_ci(diff)
        out[f'{a}|{b}'] = [round(float(diff.mean()), 1), round(low, 1), round(high, 1),
                           int((diff > 0).sum()), int((diff < 0).sum())]
    return out


def curves():
    def series(path, x, y, every=1):
        rows = read_csv(path)
        return [[round(float(r[x]), 2), round(float(r[y]), 2)] for r in rows[::every] if r[y] not in ('', 'nan')]

    def smoothed(path, window=15):
        rows = read_csv(path)
        xs = [float(r['elapsed_min']) for r in rows]
        ys = [float(r['episode_score']) for r in rows]
        out = []
        for i in range(0, len(rows), 3):
            part = ys[max(0, i - window + 1):i + 1]
            out.append([round(xs[i], 1), round(sum(part) / len(part), 1)])
        return out

    runs = os.path.join(KAGGLE, 'runs')
    return {
        'il_rl': {'eval': series(os.path.join(runs, 'il_rl', 'eval_metrics.csv'), 'elapsed_min', 'eval_score'),
                  'train': smoothed(os.path.join(runs, 'il_rl', 'train_metrics.csv'))},
        'rl_scratch': {'eval': series(os.path.join(runs, 'rl_scratch', 'eval_metrics.csv'), 'elapsed_min', 'eval_score'),
                       'train': smoothed(os.path.join(runs, 'rl_scratch', 'train_metrics.csv'))},
        'il': {f'seed{s}': [[int(r['epoch']), round(float(r['eval_acc']) * 100, 1), round(float(r['elapsed_min']), 1)]
                            for r in read_csv(os.path.join(runs, f'il_seed{s}', 'il_metrics.csv'))] for s in (0, 1)},
    }


def gpu():
    rows = read_csv(os.path.join(KAGGLE, 'gpu_usage.csv'))
    start = min(int(r['t']) for r in rows)
    bins = defaultdict(list)
    for r in rows:
        bins[((int(r['t']) - start) // 120, int(r['gpu']))].append(float(r['util']))
    last = max(k[0] for k in bins)
    return [[round(i * 2 / 60, 3)] + [round(sum(bins[(i, g)]) / len(bins[(i, g)])) if bins.get((i, g)) else None
                                      for g in (0, 1)] for i in range(last + 1)]


def stages():
    """Minutes taken by each stage of the Kaggle run, in order."""
    folder = os.path.join(KAGGLE, 'stages')
    out = []
    if os.path.isdir(folder):
        for name in ('install', 'data', 'catalog', 'tests', 'train_il', 'train_rl', 'train_value', 'evaluate'):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    out.append([name, round(float(f.read().split()[0]), 1)])
    return out


def eval_times():
    """[agent, minutes] in the order the run evaluated them, one at a time, from the notebook's log."""
    with open(NOTEBOOK_LOG, encoding='utf-8') as f:
        text = ''.join(entry.get('data', '') for entry in json.load(f))
    text = text[text.find('=== evaluate ==='):]
    seconds = lambda hms: sum(int(x) * m for x, m in zip(hms.split(':'), (3600, 60, 1)))  # noqa: E731
    started, out = {}, []
    for clock, name, done in re.findall(r'\[(\d\d:\d\d:\d\d)\] (?:Starting (\w+) on GPU \d|(\w+) finished)', text):
        if name:
            started[name] = seconds(clock)
        elif done in started:
            out.append([done, round(((seconds(clock) - started[done]) % 86400) / 60, 2)])  # the run passed midnight
    return out


def run_details():
    """Logged counters for the pipeline playback: demonstrations, RL episodes and actions over time."""
    runs = os.path.join(KAGGLE, 'runs')
    with open(os.path.join(KAGGLE, 'logs', 'il_seed0.log'), encoding='utf-8', errors='replace') as f:
        demos = re.search(r'train: (\d+) trajectories, (\d+) transitions', f.read())
    out = {'il_demos': int(demos.group(1)), 'il_actions': int(demos.group(2)), 'eval': eval_times(), 'rl': {}}
    for name in ('il_rl', 'rl_scratch'):
        summary = read_json(os.path.join(runs, name, 'train_summary.json'))
        out['rl'][name] = {
            'episodes': summary['episodes'], 'actions': summary['transitions'], 'updates': summary['updates'],
            'envs': summary['args']['num_envs'],
            # [minutes, episodes so far, actions so far]
            'counts': [[round(float(r['elapsed_min']), 2), int(r['episodes']), int(r['transitions'])]
                       for r in read_csv(os.path.join(runs, name, 'train_metrics.csv'))],
        }
    return out


def instructions(n=36, seed=7):
    """Real shopping requests written by crowd workers, for the reward calculator."""
    data = read_json(HUMAN_INSTRUCTIONS)
    pool = [g for goals in data.values() for g in goals
            if 1 <= len(g['instruction_attributes']) <= 3 and 1 <= len(g['instruction_options']) <= 3
            and 40 <= len(g['instruction']) <= 150]
    random.Random(seed).shuffle(pool)
    return [{'text': g['instruction'].strip().rstrip('.'), 'attributes': g['instruction_attributes'],
             'options': g['instruction_options']} for g in pool[:n]]


def main():
    agent_rows, per_goal = agents()
    validation = read_json(os.path.join(LOCAL, 'value_validation.json'))
    data = {
        'agents': agent_rows,
        'paired': paired(per_goal),
        'curves': curves(),
        'gpu': gpu(),
        'stages': stages(),
        'run': run_details(),
        'validation': [{k: (round(r[k], 1) if isinstance(r[k], float) else r[k])
                        for k in ('name', 'label', 'score', 'low', 'high', 'success', 'steps')}
                       for r in validation['rows']],
        'instructions': instructions(),
        'paper': {'rule': [45.6, 9.6], 'il': [59.9, 29.1], 'il_rl': [62.4, 28.7], 'expert': [82.1, 59.6]},
        'meta': {'gpus': read_json(os.path.join(KAGGLE, 'meta.json'))['gpus'], 'catalogue': 1181430, 'goals': 500},
    }
    path = os.path.join(ROOT, 'showcase', 'data.js')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('// Built by showcase/build_data.py from the project outputs. Do not edit by hand.\n')
        f.write('window.WEBSHOP = ' + json.dumps(data, separators=(',', ':')) + ';\n')
    print(f'wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
