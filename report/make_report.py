"""
Builds the project report from the outputs of a run. Every number and figure in the
Results section is computed here from those outputs; nothing is typed in by hand. The
written interpretation in INTERPRETATION.md describes one particular run, and is left
out of a report on any other run.

    python report/make_report.py --outputs <outputs dir> --out_dir <report dir>
        [--previous <outputs dir of an earlier run>] [--laptop <results_cpu dir>]

--previous adds a comparison with an earlier full run; --laptop adds the validation runs and the
earlier test runs of the added agents that were made on a laptop before the Kaggle run.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INTERPRETATION = HERE / 'INTERPRETATION.md'
INTERPRETED_RUN = {'rule_softmax': 46.7, 'il_rl_softmax': 59.2, 'il_rl_assist_greedy': 68.2, 'value_shopper': 80.4}  # its test scores

STAGE_ORDER = ['install', 'data', 'catalog', 'tests', 'train_il', 'train_rl', 'train_value', 'evaluate']
OPTIONAL_STAGES = {'train_value'}  # runs made before the value model was added have no such stage
GPU_STAGES = {'train_il': 'Imitation learning', 'train_rl': 'Reinforcement learning', 'train_value': 'Value model',
              'evaluate': 'Evaluation'}

# Published results (Yao et al., 2022, Figure 4 and Table 2).
PAPER = [
    ('Rule', 45.6, 9.6),
    ('IL', 59.9, 29.1),
    ('IL+RL', 62.4, 28.7),
    ('Human expert', 82.1, 59.6),
]

# Fixed colour per agent, in the validated categorical order.
COLORS = {'il_rl': '#2a78d6', 'il': '#eb6834', 'rl_scratch': '#1baf7a', 'rule': '#eda100',
          'il_instruction': '#e87ba4'}
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e5e4df', '#ffffff'

NAMES = {
    'rule': 'Rule baseline',
    'il': 'IL',
    'il_instruction': 'IL, instruction as query',
    'il_rl': 'IL+RL',
    'rl_scratch': 'RL from pretrained BERT',
    'rule_assist': 'Rule baseline + assist',
    'il_rl_assist': 'IL+RL + assist',
    'careful': 'Careful shopper (rules)',
    'value': 'Value shopper',
}
IMPROVED = ['rule_assist', 'il_rl_assist_greedy', 'careful_shopper', 'value_shopper']  # Section 6.5


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def column(rows, name):
    return np.array([float(r[name]) if r.get(name) not in (None, '') else np.nan for r in rows])


def style(ax):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def bootstrap_mean_ci(values, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_results(results_dir):
    """Results by evaluation name, e.g. 'il_seed0_softmax'."""
    results = {}
    if os.path.isdir(results_dir):
        for name in sorted(os.listdir(results_dir)):
            if name.endswith('.json'):
                results[name[:-5]] = read_json(os.path.join(results_dir, name))
    return results


def agent_of(eval_name):
    for prefix, agent in (('rule_assist', 'rule_assist'), ('il_rl_assist', 'il_rl_assist'), ('careful', 'careful'), ('value', 'value')):
        if eval_name.startswith(prefix):
            return agent
    if eval_name.startswith('rule'):
        return 'rule'
    if eval_name.startswith('il_rl'):
        return 'il_rl'
    if eval_name.startswith('rl_scratch'):
        return 'rl_scratch'
    if 'instruction' in eval_name:
        return 'il_instruction'
    return 'il'


def per_goal(results, name):
    return np.array([r['score'] for r in results[name]['results']])


def pick_il(results):
    """The IL evaluation to compare against: the seed used to start IL+RL."""
    for name in ('il_seed0_softmax', 'il_hub_softmax'):
        if name in results:
            return name
    return None


def fig_test_scores(results, main, out):
    """Score and success rate, as two small multiples on their own scales."""
    labels = [NAMES[agent_of(n)] for n in main][::-1]
    colors = [COLORS[agent_of(n)] for n in main][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 0.55 * len(main) + 1.3), sharey=True)
    for ax, key, title in ((axes[0], 'score', 'Score (0 to 100)'), (axes[1], 'success_rate', 'Success rate (%)')):
        values = [results[n]['summary'][key] for n in main][::-1]
        bars = ax.barh(labels, values, color=colors, height=0.55)
        ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=9, color=INK)
        style(ax)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        ax.grid(axis='y', visible=False)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.set_xlim(0, max(max(values) * 1.2, 1))
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_improved(results, names, out):
    """How far each change moved the score, from the best evaluation of the original agents upward."""
    labels = [NAMES[agent_of(n)] + (' (greedy)' if n.endswith('_greedy') else '') for n in names][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 0.5 * len(names) + 1.3), sharey=True)
    for ax, key, title in ((axes[0], 'score', 'Score (0 to 100)'), (axes[1], 'success_rate', 'Success rate (%)')):
        values = [results[n]['summary'][key] for n in names][::-1]
        bars = ax.barh(labels, values, color=COLORS['il_rl'], height=0.55)
        ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=9, color=INK)
        style(ax)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        ax.grid(axis='y', visible=False)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_breakdown(results, main, out):
    """Each reward component per agent, one panel per component."""
    parts = [('r_att', 'Attributes'), ('r_option', 'Options'), ('r_price', 'Price'), ('r_type', 'Product type')]
    labels = [NAMES[agent_of(n)] for n in main][::-1]
    colors = [COLORS[agent_of(n)] for n in main][::-1]
    fig, axes = plt.subplots(1, 4, figsize=(10, 0.55 * len(main) + 1.3), sharey=True, sharex=True)
    for ax, (key, title) in zip(axes, parts):
        values = [results[n]['summary'][key] for n in main][::-1]
        bars = ax.barh(labels, values, color=colors, height=0.55)
        ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8, color=INK)
        style(ax)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        ax.grid(axis='y', visible=False)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.set_xlim(0, 118)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_learning(runs, out):
    """Evaluation score and training episode score over environment transitions."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    plotted = False
    for key, name in (('il_rl', 'IL+RL'), ('rl_scratch', 'RL from pretrained BERT')):
        evals, trains = runs.get(key, ([], []))
        if evals:
            x, y = column(evals, 'transitions'), column(evals, 'eval_score')
            axes[0].plot(x, y, color=COLORS[key], linewidth=2, marker='o', markersize=5,
                         markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
            plotted = True
        if trains:
            axes[1].plot(column(trains, 'transitions'), column(trains, 'episode_score'),
                         color=COLORS[key], linewidth=2, label=name)
    if not plotted:
        plt.close(fig)
        return False
    axes[0].set_title('Held-out evaluation goals', loc='left', fontsize=10, color=INK)
    axes[1].set_title('Training episodes (mean of last 200)', loc='left', fontsize=10, color=INK)
    axes[0].set_ylabel('Score (0 to 100)')
    for ax in axes:
        style(ax)
        ax.set_xlabel('Environment transitions')
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc='lower right')
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_il(il_runs, out):
    """Validation accuracy of the imitation models by epoch."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    palette = ['#eb6834', '#2a78d6']
    plotted = False
    for (name, rows), color in zip(il_runs.items(), palette):
        if rows:
            ax.plot(column(rows, 'epoch') + 1, column(rows, 'eval_acc') * 100, color=color, linewidth=2,
                    marker='o', markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    style(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Human action predicted (%)')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return True


def stage_windows(stages_dir):
    """Hours from the start of the run at which each GPU stage began and ended.

    Built from the stage timings, so it only applies to a run that was not resumed.
    """
    windows, hours = {}, 0.0
    for name in STAGE_ORDER:
        try:
            with open(os.path.join(stages_dir, name)) as f:
                minutes = float(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            if name in OPTIONAL_STAGES:
                continue
            return {}
        if name in GPU_STAGES:
            windows[GPU_STAGES[name]] = (hours, hours + minutes / 60)
        hours += minutes / 60
    return windows


def rolling_mean(values, width):
    return np.convolve(values, np.ones(width) / width, mode='same')


def fig_gpu(rows, out, windows):
    """GPU utilization and memory over the run, one panel per measure, marked by stage."""
    if not rows:
        return None
    by_gpu = {}
    for r in rows:
        by_gpu.setdefault(r['gpu'], []).append(r)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)
    palette = ['#2a78d6', '#eb6834']
    start = min(float(r['t']) for r in rows)
    width = max(1, round(600 / np.median(np.diff(column(next(iter(by_gpu.values())), 't')))))
    for (gpu, gpu_rows), color in zip(sorted(by_gpu.items()), palette):
        hours = (column(gpu_rows, 't') - start) / 3600
        axes[0].plot(hours, rolling_mean(column(gpu_rows, 'util'), width), color=color, linewidth=2, label=f'GPU {gpu}')
        axes[1].plot(hours, column(gpu_rows, 'mem_used') / column(gpu_rows, 'mem_total') * 100,
                     color=color, linewidth=1.5, label=f'GPU {gpu}')
    axes[0].set_title('Utilization (%, 10-minute average)', loc='left', fontsize=10, color=INK)
    axes[1].set_title('Memory used (%)', loc='left', fontsize=10, color=INK)
    short = {'Imitation learning': 'IL', 'Reinforcement learning': 'RL', 'Value model': 'Value', 'Evaluation': 'Eval.'}
    for ax in axes:
        style(ax)
        ax.set_xlabel('Hours since the start of the run')
        ax.set_ylim(0, 118)
        ax.set_yticks(range(0, 101, 20))
        for stage, (low, high) in windows.items():
            ax.axvline(low, color=INK_2, linewidth=0.8, linestyle=':')
            ax.text((low + high) / 2, 110, short[stage], ha='center', va='center', fontsize=9, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_2, loc='upper right', ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)

    stats = {}
    for stage, (low, high) in windows.items():
        stats[stage] = {}
        for gpu, gpu_rows in sorted(by_gpu.items()):
            hours = (column(gpu_rows, 't') - start) / 3600
            chosen = (hours >= low) & (hours < high)
            util = column(gpu_rows, 'util')[chosen]
            memory = (column(gpu_rows, 'mem_used') / column(gpu_rows, 'mem_total') * 100)[chosen]
            stats[stage][gpu] = {'util': float(util.mean()), 'idle': float((util == 0).mean()) * 100,
                                 'memory': float(memory.mean())}
    return stats or {'': {}}


def label_of(name):
    label = NAMES[agent_of(name)]
    if name.endswith('_greedy'):
        label += ' (greedy)'
    if 'seed1' in name:
        label += ' (seed 1)'
    return label


def results_table(results, names):
    head = '| Agent | Score | 95% CI | Success (%) | Attributes | Options | Price | Type | Steps | Results pages* | Products opened* |\n'
    head += '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    rows = []
    for name in names:
        s = results[name]['summary']
        low, high = bootstrap_mean_ci(per_goal(results, name))
        rows.append(f"| {label_of(name)} | {s['score']:.1f} | {low:.1f} to {high:.1f} | {s['success_rate']:.1f} | "
                    f"{s['r_att']:.1f} | {s['r_option']:.1f} | {s['r_price']:.1f} | {s['r_type']:.1f} | "
                    f"{s['steps']:.1f} | {s['searches']:.2f} | {s['items']:.2f} |")
    return head + '\n'.join(rows)


def compare(results, a, b):
    """Paired comparison of two agents over the same goals (a minus b)."""
    diff = per_goal(results, a) - per_goal(results, b)
    low, high = bootstrap_mean_ci(diff)
    return float(diff.mean()), low, high


def comparison_sentence(results, a, b, a_name=None, b_name=None):
    if a not in results or b not in results:
        return ''
    if len(results[a]['results']) != len(results[b]['results']):
        return ''
    mean, low, high = compare(results, a, b)
    a_name, b_name = a_name or NAMES[agent_of(a)], b_name or NAMES[agent_of(b)]
    if low > 0:
        verdict = f'{a_name} scored higher, and the interval excludes zero'
    elif high < 0:
        verdict = f'{a_name} scored lower, and the interval excludes zero'
    else:
        verdict = 'the interval includes zero, so these goals do not show a reliable difference'
    return (f'- **{a_name} minus {b_name}:** {mean:+.1f} points (paired 95% bootstrap interval '
            f'{low:+.1f} to {high:+.1f}); {verdict}.')


def build_context(outputs, figures_dir, previous=None, laptop=None):
    outputs = str(outputs)
    results = load_results(os.path.join(outputs, 'results'))
    meta = read_json(os.path.join(outputs, 'meta.json')) or {}
    runs_dir = os.path.join(outputs, 'runs')
    catalog = {name: read_json(os.path.join(outputs, f'catalog_{name}.json')) or {} for name in ('train', 'test')}

    order = ['rule_softmax', 'il_seed0_instruction_softmax', pick_il(results), 'rl_scratch_softmax', 'il_rl_softmax']
    main = [n for n in order if n and n in results]
    all_names = main + [n for n in ('il_seed1_softmax', 'il_seed0_greedy', 'rl_scratch_greedy', 'il_rl_greedy')
                        if n in results]
    improved = [n for n in IMPROVED if n in results]

    os.makedirs(figures_dir, exist_ok=True)
    figures = {}
    if improved:
        ladder = ([n for n in ('rule_softmax', 'il_rl_greedy') if n in results]) + improved
        fig_improved(results, ladder, os.path.join(figures_dir, 'improved_scores.png'))
        figures['improved'] = True
    if main:
        fig_test_scores(results, main, os.path.join(figures_dir, 'test_scores.png'))
        fig_breakdown(results, main, os.path.join(figures_dir, 'reward_breakdown.png'))
        figures['test_scores'] = figures['reward_breakdown'] = True

    rl_runs = {
        key: (read_csv(os.path.join(runs_dir, key, 'eval_metrics.csv')),
              read_csv(os.path.join(runs_dir, key, 'train_metrics.csv')))
        for key in ('il_rl', 'rl_scratch')
    }
    figures['learning'] = fig_learning(rl_runs, os.path.join(figures_dir, 'learning_curves.png'))
    il_rows = {f'Seed {s}': read_csv(os.path.join(runs_dir, f'il_seed{s}', 'il_metrics.csv')) for s in (0, 1)}
    figures['il'] = fig_il(il_rows, os.path.join(figures_dir, 'il_accuracy.png'))
    gpu_stats = fig_gpu(read_csv(os.path.join(outputs, 'gpu_usage.csv')), os.path.join(figures_dir, 'gpu_usage.png'),
                        stage_windows(os.path.join(outputs, 'stages')))
    figures['gpu'] = bool(gpu_stats)

    summaries = {key: read_json(os.path.join(runs_dir, key, 'train_summary.json')) for key in rl_runs}
    il_summaries = {s: read_json(os.path.join(runs_dir, f'il_seed{s}', 'all_results.json')) for s in (0, 1)}
    throughput = {}
    for key, (_, trains) in rl_runs.items():
        if trains:
            rates = column(trains, 'transitions_per_sec')
            throughput[key] = float(np.nanmedian(rates))

    # Laptop runs: kept beside the outputs by the Kaggle pipeline, or passed with --laptop.
    if laptop:
        selection = read_json(os.path.join(laptop, 'value_validation.json')) or {}
        before_fix = load_results(os.path.join(laptop, 'before_fix'))
        second_run = load_results(os.path.join(laptop, 'second_run'))
        laptop_final = {n: r for n, r in load_results(laptop).items() if n in IMPROVED}
    else:
        selection = read_json(os.path.join(outputs, 'value_validation.json')) or {}
        before_fix = load_results(os.path.join(outputs, 'results_before_fix'))
        second_run = load_results(os.path.join(outputs, 'results_second_run'))
        laptop_final = {}
    previous_results = load_results(os.path.join(str(previous), 'results')) if previous else {}
    return dict(results=results, meta=meta, main=main, all_names=all_names, improved=improved, selection=selection,
                before_fix=before_fix, second_run=second_run, laptop_final=laptop_final, previous=previous_results,
                figures=figures, summaries=summaries, il_summaries=il_summaries, throughput=throughput,
                gpu_stats=gpu_stats, catalog=catalog, rl_runs=rl_runs, il_rows=il_rows)


def setup_section(c):
    meta, catalog = c['meta'], c['catalog']
    lines = []
    if catalog['train']:
        lines.append(f"- **Training catalogue:** {catalog['train'].get('products', 0):,} products, including all "
                     f"{catalog['train'].get('human_goal_products', 0):,} products with a human instruction. "
                     f"A smaller catalogue makes search faster and saves memory.")
    if catalog['test']:
        lines.append(f"- **Test catalogue:** {catalog['test'].get('products', 0):,} products. Testing on the full "
                     f"catalogue keeps scores comparable with the paper, because a small catalogue makes the task easier.")
    if meta.get('gpus'):
        lines.append(f"- **Hardware:** {meta['gpus']}. Training ran one experiment per GPU at the same time; "
                     f"evaluation on the full catalogue ran one agent at a time.")
    lines.append(f"- **Library versions:** Python {meta.get('python', '3.10')}, PyTorch {meta.get('torch', '2.2.2')}, "
                 f"Transformers {meta.get('transformers', '4.41.2')}, spaCy {meta.get('spacy', '3.3.0')}.")
    for s, summary in c['il_summaries'].items():
        if summary:
            planned = summary['args']['num_train_epochs']
            reached = len(c['il_rows'].get(f'Seed {s}') or [])
            limit = summary['args'].get('max_minutes')
            stopped = (f", stopped by the {limit:.0f}-minute limit during epoch {reached} of {planned}"
                       if limit and summary['minutes'] >= limit and reached < planned else '')
            lines.append(f"- **IL seed {s}:** {summary['train_samples']:,} training samples, "
                         f"best validation accuracy {summary['best_eval_accuracy'] * 100:.1f}%, "
                         f"{summary['minutes']:.0f} minutes{stopped}.")
    labels = {'il_rl': 'IL+RL', 'rl_scratch': 'RL from pretrained BERT'}
    for key, label in labels.items():
        summary = c['summaries'].get(key)
        if summary:
            rate = c['throughput'].get(key)
            rate_text = f"; median {rate:.1f} transitions per second" if rate is not None else ''
            lines.append(f"- **{label} training:** {summary['transitions']:,} transitions, "
                         f"{summary['updates']:,} updates, {summary['episodes']:,} episodes, "
                         f"{summary['minutes'] / 60:.1f} hours{rate_text}.")
    return '\n'.join(lines)


def gpu_section(c):
    if not c['gpu_stats']:
        return 'GPU usage was not recorded for this run.'
    figure = '![GPU utilization and memory](figures/gpu_usage.png)'
    if '' in c['gpu_stats']:
        return figure
    rows = []
    for stage, gpus in c['gpu_stats'].items():
        cell = lambda key: ' / '.join(f"{gpus[gpu][key]:.0f}" for gpu in sorted(gpus))
        rows.append(f"| {stage} | {cell('util')} | {cell('idle')} | {cell('memory')} |")
    table = ('| Stage (GPU 0 / GPU 1) | Mean utilization (%) | Samples at 0% utilization (%) | Mean memory (%) |\n'
             '|---|---:|---:|---:|\n' + '\n'.join(rows))
    return (table + '\n\n*Sampled with `nvidia-smi`. Its memory figure includes the cache PyTorch keeps reserved, '
            'so it is higher than the 80% budget that the micro-batches aim for.*\n\n' + figure)


def findings_section(c):
    results = c['results']
    il = pick_il(results)
    lines = []
    if 'il_rl_softmax' in results and il:
        lines.append(comparison_sentence(results, 'il_rl_softmax', il))
    if 'rl_scratch_softmax' in results and 'il_rl_softmax' in results:
        lines.append(comparison_sentence(results, 'il_rl_softmax', 'rl_scratch_softmax'))
    if il and 'rule_softmax' in results:
        lines.append(comparison_sentence(results, il, 'rule_softmax'))
    if 'rl_scratch_softmax' in results and 'rule_softmax' in results:
        lines.append(comparison_sentence(results, 'rl_scratch_softmax', 'rule_softmax'))
    if il and 'il_seed0_instruction_softmax' in results:
        lines.append(comparison_sentence(results, il, 'il_seed0_instruction_softmax'))
    if 'il_seed0_softmax' in results and 'il_seed1_softmax' in results:
        lines.append(comparison_sentence(results, 'il_seed0_softmax', 'il_seed1_softmax').replace(
            'IL minus IL', 'IL seed 0 minus IL seed 1').replace('IL scored', 'Seed 0 scored'))
    lines.append(comparison_sentence(results, 'il_rl_greedy', 'il_seed0_greedy', 'IL+RL (greedy)', 'IL (greedy)'))
    lines.append(comparison_sentence(results, 'il_rl_greedy', 'il_rl_softmax', 'IL+RL (greedy)', 'IL+RL (sampled)'))
    return '\n'.join(line for line in lines if line) or 'No comparisons are available for this run.'


def improved_comparisons(c):
    results = c['results']
    pairs = [('value_shopper', 'il_rl_greedy', 'Value shopper', 'IL+RL (greedy)'),
             ('value_shopper', 'careful_shopper', 'Value shopper', 'Careful shopper'),
             ('careful_shopper', 'rule_assist', 'Careful shopper', 'Rule baseline + assist'),
             ('il_rl_assist_greedy', 'il_rl_greedy', 'IL+RL + assist (greedy)', 'IL+RL (greedy)'),
             ('rule_assist', 'rule_softmax', 'Rule baseline + assist', 'Rule baseline')]
    lines = [comparison_sentence(results, a, b, a_name, b_name) for a, b, a_name, b_name in pairs]
    return '\n'.join(line for line in lines if line)


def earlier_runs(runs):
    """'Label 71.8 and 48.4%; ...' for each agent of an earlier test run."""
    parts = []
    for name, data in runs.items():
        label = NAMES[agent_of(name)] + (' (greedy)' if name.endswith('_greedy') else '')
        parts.append(f"{label} {data['summary']['score']:.1f} and {data['summary']['success_rate']:.1f}%")
    return '; '.join(parts)


def before_fix_note(c):
    """Earlier test runs of the added agents, made on a laptop before two faults were fixed, are reported too."""
    first, second, final = c['before_fix'], c['second_run'], c['laptop_final']
    if not first:
        return ''
    lead = ('Before this Kaggle run, the added agents were developed on a laptop and evaluated there on the test goals '
            'more than once' if final else 'The added agents were evaluated on the test goals more than once')
    text = (f'*{lead}, because faults were found after the first evaluation. Their settings did not change between '
            'runs. (1) The text view never '
            'showed an option containing a double quote, such as `20"x20"`, as selected, so assist mode and the shoppers '
            'clicked it again and again. The reward was not affected, because the site recorded the click. This was fixed in '
            'the text view, and the agents were changed never to repeat a click on the same visit. The first evaluation, '
            'before this fix, gave ' + earlier_runs(first) + '.')
    if second:
        text += (' (2) A change made for this work let assist mode treat the product buttons on a results page as option '
                 'buttons, so it opened a product on the agent\'s behalf. Only the two agents that use assist mode were '
                 'affected, and the fault was found from their low price credit in the second evaluation, which gave '
                 + earlier_runs(second) + '. They were evaluated a third time after the fix.')
    results = c['results']
    same = [n for n, r in final.items() if n in results and len(r['results']) == len(results[n]['results'])
            and np.array_equal(per_goal(results, n), np.array([x['score'] for x in r['results']]))]
    differ = [n for n in final if n in results and n not in same]
    if same:
        phrases = {'rule_assist': 'the rule baseline with assist', 'careful_shopper': 'the careful shopper',
                   'value_shopper': 'the value shopper', 'il_rl_assist_greedy': 'IL+RL with assist'}
        named = [phrases.get(n, label_of(n)) for n in same]
        listed = named[0] if len(named) == 1 else ', '.join(named[:-1]) + ' and ' + named[-1]
        text += f' The third laptop evaluation and this Kaggle run gave identical scores on every test goal for {listed}.'
    for n in differ:
        text += (f" {label_of(n)} uses the IL+RL weights, which this run trained again, so its score differs "
                 f"({final[n]['summary']['score']:.1f} on the laptop with the first run's weights).")
    return text + (' The validation runs above were made before fix (1); fault (2) did not affect the value shopper or the '
                   'careful shopper.*')


def run_comparison(c):
    """The same agents in an earlier full run, compared goal by goal with this one."""
    previous, results = c['previous'], c['results']
    names = [n for n in c['all_names'] if n in previous and len(previous[n]['results']) == len(results[n]['results'])]
    if not names:
        return 'Each agent was trained and evaluated once.'
    rows = []
    for n in names:
        diff = per_goal(results, n) - per_goal(previous, n)
        low, high = bootstrap_mean_ci(diff)
        limit = lambda r: sum(1 for x in r[n]['results'] if x['steps'] >= 100)  # noqa: E731
        rows.append(f"| {label_of(n)} | {previous[n]['summary']['score']:.1f} | {results[n]['summary']['score']:.1f} | "
                    f"{diff.mean():+.1f} | {low:+.1f} to {high:+.1f} | {limit(previous)} / {limit(results)} |")
    return ('| Agent | First run | This run | Difference | Paired 95% CI | Goals at the step limit (first / this) |\n'
            '|---|---:|---:|---:|---:|---:|\n' + '\n'.join(rows))


def selection_table(selection):
    """The settings compared on validation goals, from which the test configuration was chosen."""
    if not selection:
        return ''
    rows = '\n'.join(f"| {r['label']} | {r['score']:.1f} | {r['low']:.1f} to {r['high']:.1f} | {r['success']:.1f} | {r['steps']:.1f} |"
                     for r in selection['rows'])
    return ('| Value shopper settings (validation) | Score | 95% CI | Success (%) | Steps |\n|---|---:|---:|---:|---:|\n' + rows)


def paper_table():
    rows = '\n'.join(f'| {name} | {score} | {success} |' for name, score, success in PAPER)
    return f'| Agent (published) | Score | Success (%) |\n|---|---:|---:|\n{rows}'


def interpretation(c):
    """The written parts of the report, which only apply to the run they were written about."""
    scores = {name: round(c['results'][name]['summary']['score'], 1) for name in INTERPRETED_RUN if name in c['results']}
    written = c['meta'].get('mode') == 'full' and scores == INTERPRETED_RUN and INTERPRETATION.exists()
    if not written:
        return {'ABSTRACT_RESULT': '',
                'DISCUSSION': ('*[To complete after the run: interpret Section 6 in your own words. Address the three '
                               'questions from Section 1, and say whether the curves had flattened.]*'),
                'CONCLUSION': '*[To complete after the run: two or three sentences on what was learned, based on Section 6.]*'}
    parts = re.split(r'<!-- (\w+) -->\n', INTERPRETATION.read_text(encoding='utf-8'))[1:]
    return {name.upper(): text.strip() for name, text in zip(parts[::2], parts[1::2])}


def fill(template, c):
    results = c['results']
    values = {
        **interpretation(c),
        'SETUP': setup_section(c),
        'RESULTS_TABLE': results_table(results, c['all_names']) if c['all_names'] else 'No evaluation results found.',
        'IMPROVED_TABLE': results_table(results, c['improved']) if c['improved'] else 'These agents were not evaluated in this run.',
        'IMPROVED_COMPARISONS': improved_comparisons(c),
        'BEFORE_FIX_NOTE': before_fix_note(c),
        'RUN_COMPARISON': run_comparison(c),
        'SELECTION_TABLE': selection_table(c['selection']),
        'VALUE_FEATURE_COUNT': str(c['selection'].get('features', 'its')),
        'VALUE_GOAL_COUNT': f"{c['selection'].get('train_goals', 2000):,}",
        'FIG_IMPROVED': '![Score and success rate as each change is added](figures/improved_scores.png)' if c['figures'].get('improved') else '',
        'PAPER_TABLE': paper_table(),
        'COMPARISONS': findings_section(c),
        'GPU_SECTION': gpu_section(c),
        'FIG_TEST': '![Test score and success rate per agent](figures/test_scores.png)' if c['figures'].get('test_scores') else '',
        'FIG_BREAKDOWN': '![Reward components per agent](figures/reward_breakdown.png)' if c['figures'].get('reward_breakdown') else '',
        'FIG_LEARNING': '![Learning curves](figures/learning_curves.png)' if c['figures'].get('learning') else '',
        'FIG_IL': '![Imitation learning validation accuracy](figures/il_accuracy.png)' if c['figures'].get('il') else '',
        'RUN_NOTE': ('**This report comes from a short smoke-test run. Its numbers only show that the pipeline '
                     'works and must not be presented as results.**\n\n') if c['meta'].get('mode') == 'smoke' else '',
    }
    for key, value in values.items():
        template = template.replace('{{' + key + '}}', value)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--template', default=str(HERE / 'REPORT_TEMPLATE.md'))
    parser.add_argument('--previous', default=None, help='outputs folder of an earlier full run, to compare with')
    parser.add_argument('--laptop', default=None, help='results_cpu folder: validation and earlier test runs on a laptop')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    context = build_context(args.outputs, os.path.join(args.out_dir, 'figures'), args.previous, args.laptop)
    with open(args.template, encoding='utf-8') as f:
        report = fill(f.read(), context)
    md_path = os.path.join(args.out_dir, 'REPORT.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print('Wrote', md_path)

    pandoc = shutil.which('pandoc')
    if not pandoc:
        try:  # pip install pypandoc_binary also provides pandoc
            import pypandoc
            pandoc = pypandoc.get_pandoc_path()
        except (ImportError, OSError):
            pandoc = None
    if pandoc:
        subprocess.run([pandoc, 'REPORT.md', '-o', 'REPORT.docx', '--resource-path=.'], cwd=args.out_dir, check=True)
        print('Wrote', os.path.join(args.out_dir, 'REPORT.docx'))
    else:
        print('pandoc not found: skipped the Word version')


if __name__ == '__main__':
    main()
