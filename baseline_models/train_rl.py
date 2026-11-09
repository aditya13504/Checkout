"""
Trains the choice policy with synchronous advantage actor-critic (A2C).

Every environment is stepped between one batched GPU pass and the next. Training
stops at --max_hours as well as --max_steps, and can be resumed with --resume.
"""
import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch

from agent import Agent, Transition
from vec_env import VecWebEnv, run_episodes



class CsvLog:
    """Appends rows to a CSV file, keeping earlier rows when a run is resumed."""

    def __init__(self, path, columns):
        self.columns = columns
        is_new = not os.path.exists(path)
        self.file = open(path, 'a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=columns, extrasaction='ignore')
        if is_new:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_policy(agent, method, search='model'):
    """A batched policy for `run_episodes`.

    `search` is how search pages are handled: 'model' lets the network choose among
    the candidate queries, 'top1' takes the first and 'instruction' the last.
    """
    def act_batch(obs_list, infos):
        actions = [None] * len(obs_list)
        ask_model = []
        for i, info in enumerate(infos):
            valid = info['valid']
            if len(valid) == 1:
                actions[i] = valid[0]
            elif valid[0].startswith('search[') and search == 'top1':
                actions[i] = valid[0]
            elif valid[0].startswith('search[') and search == 'instruction':
                actions[i] = valid[-1]
            else:
                ask_model.append(i)
        if ask_model:
            states = agent.build_states([obs_list[i] for i in ask_model], [infos[i] for i in ask_model])
            valids = [infos[i]['valid'] for i in ask_model]
            chosen, _, _, _ = agent.act(states, valids, method=method)
            for i, action in zip(ask_model, chosen):
                actions[i] = action
        return actions
    return act_batch


def evaluate(agent, vec, env_ids, goal_idxs, method='softmax'):
    results = run_episodes(vec, env_ids, goal_idxs, make_policy(agent, method))
    return {
        'score': float(np.mean([r['score'] for r in results])),
        'success': float(np.mean([r['success'] for r in results])),
        'steps': float(np.mean([r['steps'] for r in results])),
    }


def train(agent, vec, args):
    train_ids = list(range(args.num_envs))
    eval_ids = list(range(args.num_envs, args.num_envs + args.num_eval_envs))
    train_goals = vec.goal_range('train')
    eval_goals = list(vec.goal_range('eval'))
    rng = random.Random(args.seed)
    rng.shuffle(eval_goals)
    eval_goals = eval_goals[:args.eval_episodes]

    os.makedirs(args.output_dir, exist_ok=True)
    train_log = CsvLog(os.path.join(args.output_dir, 'train_metrics.csv'), [
        'step', 'transitions', 'updates', 'elapsed_min', 'episodes', 'episode_score', 'episode_success',
        'episode_steps', 'transitions_per_sec', 'loss', 'loss_pg', 'loss_td', 'loss_en', 'returns',
        'advs', 'grad_norm', 'token_budget', 'gpu_mem_share', 'env_errors',
    ])
    eval_log = CsvLog(os.path.join(args.output_dir, 'eval_metrics.csv'),
                      ['step', 'transitions', 'elapsed_min', 'eval_score', 'eval_success', 'eval_steps'])

    step, transitions_seen, updates, best_eval, previous_elapsed = 0, 0, 0, -1.0, 0.0
    latest = os.path.join(args.output_dir, 'ckpt_latest.pt')
    if args.resume and os.path.exists(latest):
        checkpoint = agent.load_checkpoint(latest)
        step = checkpoint['step']
        transitions_seen = checkpoint['transitions']
        updates = checkpoint.get('updates', 0)
        best_eval = checkpoint['best_eval']
        previous_elapsed = checkpoint['elapsed']
        print(f'Resumed from step {step} ({previous_elapsed / 60:.0f} min done)', flush=True)

    start = time.time()

    def elapsed():
        return previous_elapsed + time.time() - start

    def save_latest():
        agent.save_checkpoint('ckpt_latest.pt', step=step, transitions=transitions_seen,
                              updates=updates, best_eval=best_eval, elapsed=elapsed())

    last_eval_step = None

    def run_eval():
        nonlocal best_eval, last_eval_step
        last_eval_step = step
        metrics = evaluate(agent, vec, eval_ids, eval_goals)
        eval_log.write({'step': step, 'transitions': transitions_seen, 'elapsed_min': elapsed() / 60,
                        'eval_score': metrics['score'], 'eval_success': metrics['success'],
                        'eval_steps': metrics['steps']})
        print(f"[eval] step {step}: score {metrics['score']:.1f}, success {metrics['success']:.3f}", flush=True)
        if metrics['score'] > best_eval:
            best_eval = metrics['score']
            agent.save_weights('model_best.pt')

    if best_eval < 0:
        run_eval()

    resets = vec.reset([(i, rng.choice(train_goals)) for i in train_ids])
    obs = [resets[i][0] for i in train_ids]
    infos = [resets[i][1] for i in train_ids]
    states = agent.build_states(obs, infos)
    valids = [info['valid'] for info in infos]

    episodes = 0
    recent = deque(maxlen=200)
    rollout, last_stats, env_errors = [], {}, 0
    last_eval, last_ckpt, last_log_time = time.time(), time.time(), time.time()
    last_log_transitions = transitions_seen
    max_seconds = args.max_hours * 3600

    while step < args.max_steps and elapsed() < max_seconds:
        step += 1
        action_strs, act_idxs, valid_ids, _ = agent.act(states, valids, method=args.exploration_method)
        outcomes = vec.step(list(zip(train_ids, action_strs)))
        transitions_seen += args.num_envs

        rewards = [outcomes[i][1] for i in train_ids]
        dones = [outcomes[i][2] for i in train_ids]
        rollout.append(Transition(states, act_idxs, rewards, valid_ids, dones))

        obs = [outcomes[i][0] for i in train_ids]
        infos = [outcomes[i][3] for i in train_ids]

        # Bootstrap from the state reached, before done envs are replaced.
        finished = [i for i, done in zip(train_ids, dones) if done]
        for i in finished:
            info = outcomes[i][3]
            episodes += 1
            env_errors += 'env_error' in info
            recent.append((outcomes[i][1], outcomes[i][1] == 10.0, info.get('steps', 0)))
        if finished:
            resets = vec.reset([(i, rng.choice(train_goals)) for i in finished])
            for i in finished:
                obs[i], infos[i] = resets[i]
        states = agent.build_states(obs, infos)
        valids = [info['valid'] for info in infos]

        if len(rollout) >= args.bptt:
            last_values = agent.get_values(states, valids)
            last_stats = agent.update(rollout, last_values)
            updates += 1
            rollout = []

        now = time.time()
        if step % args.log_freq == 0:
            rate = (transitions_seen - last_log_transitions) / max(now - last_log_time, 1e-6)
            last_log_time, last_log_transitions = now, transitions_seen
            row = {
                'step': step, 'transitions': transitions_seen, 'updates': updates,
                'elapsed_min': elapsed() / 60, 'episodes': episodes,
                'episode_score': np.mean([r[0] for r in recent]) * 10 if recent else 0.0,
                'episode_success': np.mean([r[1] for r in recent]) if recent else 0.0,
                'episode_steps': np.mean([r[2] for r in recent]) if recent else 0.0,
                'transitions_per_sec': rate, 'env_errors': env_errors, **last_stats,
            }
            train_log.write(row)
            print(f"step {step} | {elapsed() / 60:.1f} min | {rate:.1f} transitions/s | "
                  f"episode score {row['episode_score']:.1f} | success {row['episode_success']:.3f} | "
                  f"token budget {agent.token_budget}", flush=True)

        if now - last_eval >= args.eval_every_min * 60:
            run_eval()
            last_eval = time.time()
        if now - last_ckpt >= args.ckpt_every_min * 60:
            save_latest()
            last_ckpt = time.time()

    if step != last_eval_step:  # the same weights must not get a second, differently sampled score
        run_eval()
    save_latest()
    agent.save_weights('model_final.pt')
    with open(os.path.join(args.output_dir, 'train_summary.json'), 'w') as f:
        json.dump({'steps': step, 'transitions': transitions_seen, 'updates': updates, 'episodes': episodes,
                   'minutes': elapsed() / 60, 'best_eval_score': best_eval, 'env_errors': env_errors,
                   'args': vars(args)}, f, indent=2)
    print(f'Finished: {step} steps, {transitions_seen} transitions, {updates} updates, '
          f'best eval score {best_eval:.1f}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    # logging
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--ckpt_every_min', default=15, type=float)
    parser.add_argument('--eval_every_min', default=15, type=float)
    parser.add_argument('--eval_episodes', default=100, type=int)
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--resume', default=0, type=int)

    # rl
    parser.add_argument('--num_envs', default=16, type=int)
    parser.add_argument('--num_eval_envs', default=16, type=int)
    parser.add_argument('--num_workers', default=2, type=int, help='env worker processes (0 = in-process)')
    parser.add_argument('--step_limit', default=100, type=int)
    parser.add_argument('--max_steps', default=300000, type=int)
    parser.add_argument('--max_hours', default=24.0, type=float)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--clip', default=10, type=float)
    parser.add_argument('--bptt', default=8, type=int)
    parser.add_argument('--exploration_method', default='softmax', type=str, choices=['eps', 'softmax'])
    parser.add_argument('--w_pg', default=1, type=float)
    parser.add_argument('--w_td', default=1, type=float)
    parser.add_argument('--w_il', default=0, type=float)
    parser.add_argument('--w_en', default=1, type=float)

    # performance
    parser.add_argument('--amp', default=1, type=int, help='16-bit mixed precision on GPU')
    parser.add_argument('--token_budget', default=0, type=int, help='initial GPU micro-batch size in tokens')

    # model
    parser.add_argument('--network', default='bert', type=str, choices=['bert', 'rnn'])
    parser.add_argument('--bert_path', default="", type=str,
                        help="weights to start from; '' = pretrained BERT, 'scratch' = random BERT")
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--get_image', default=0, type=int, help='use image features (needs feat_conv.pt)')
    parser.add_argument('--assist', default=0, type=int, help='1 = hide unaffordable products and click named options')
    parser.add_argument('--truncation', default='right', choices=['left', 'right'], help='which end of a long page to cut')

    # env
    parser.add_argument('--num', default=None, type=int)
    parser.add_argument('--click_item_name', default=1, type=int)
    parser.add_argument('--state_format', default='text_rich', type=str)
    parser.add_argument('--human_goals', default=1, type=int, help='use human goals')
    parser.add_argument('--num_prev_obs', default=0, type=int, help='number of previous observations')
    parser.add_argument('--num_prev_actions', default=0, type=int, help='number of previous actions')
    parser.add_argument('--extra_search_path', default="./data/goal_query_predict.json", type=str,
                        help='path for extra search queries')

    # experimental
    parser.add_argument('--ban_buy', default=0, type=int, help='ban buy action before selecting options')
    parser.add_argument('--go_to_item', default=0, type=int)
    parser.add_argument('--go_to_search', default=0, type=int)
    parser.add_argument('--harsh_reward', default=0, type=int)

    parser.add_argument('--debug', default=0, type=int, help='debug mode')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.num_envs = 2
        args.num_eval_envs = 2
        args.human_goals = 0
        args.num = 100
    print(args, flush=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Fork the env workers before the GPU and the tokenizer are first used.
    vec = VecWebEnv(args, args.num_envs + args.num_eval_envs, args.num_workers, seed=args.seed)
    print(f'{vec.n_goals} goals loaded', flush=True)

    agent = Agent(args)
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        train(agent, vec, args)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
