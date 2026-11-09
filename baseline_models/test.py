"""
Evaluates a policy on every goal of a data split and writes per-goal results.

    python test.py --policy rule --out results/rule.json
    python test.py --policy model --model_path ckpts/web_click/model_best.pth --out results/il.json
    python test.py --policy model --hub_model webshop/il-choice-bert-image_0 --out results/il_hub.json
"""
import argparse
import json
import os

import numpy as np

from shopping_rules import CarefulShopper
from train_rl import make_policy, set_seed
from value_model import ValueShopper, ValueWeights
from vec_env import VecWebEnv, run_episodes


def rule_policy(obs_list, infos):
    """Search the instruction, buy the first result without choosing options."""
    actions = []
    for info in infos:
        valid = info['valid']
        if valid[0].startswith('search['):
            actions.append(valid[-1])
            continue
        items = [a for a in valid if a.startswith('click[item - ')]
        if items:
            actions.append(items[0])
        elif 'click[buy now]' in valid:
            actions.append('click[buy now]')
        else:  # a search with no results: there is nothing left to do but stop trying
            actions.append(next((a for a in valid if a == 'click[back to search]'), valid[0]))
    return actions


def summarize(results):
    return {
        'episodes': len(results),
        'score': float(np.mean([r['score'] for r in results])),
        'success_rate': float(np.mean([r['success'] for r in results])) * 100,
        'bought_rate': float(np.mean([r['bought'] for r in results])) * 100,
        'r_att': float(np.mean([r['r_att'] for r in results])) * 100,
        'r_option': float(np.mean([r['r_option'] for r in results])) * 100,
        'r_price': float(np.mean([r['r_price'] for r in results])) * 100,
        'r_type': float(np.mean([r['r_type'] for r in results])) * 100,
        'steps': float(np.mean([r['steps'] for r in results])),
        'searches': float(np.mean([r['n_search'] for r in results])),
        'items': float(np.mean([r['n_items'] for r in results])),
        'options': float(np.mean([r['n_options'] for r in results])),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a WebShop policy')
    parser.add_argument('--policy', default='model', choices=['model', 'rule', 'shopper', 'value'])
    parser.add_argument('--value_model', default='value_model.json', help='value: the weights train_value_model.py wrote')
    parser.add_argument('--pages', default=2, type=int, help='value: results pages to read per search')
    parser.add_argument('--assist', default=0, type=int, help='1 = hide unaffordable products and click named options')
    parser.add_argument('--truncation', default='right', choices=['left', 'right'],
                        help='must match how the model was trained; the original checkpoints use left')
    parser.add_argument('--inspect', default=5, type=int, help='shopper: how many results to open per search')
    parser.add_argument('--queries', default='top1', help='shopper: searches to try, as a comma list of top1,instruction')
    parser.add_argument('--read_details', default=1, type=int, help='shopper: also read the features and description')
    parser.add_argument('--model_path', default='', help='model weights (.pth or .pt) to evaluate')
    parser.add_argument('--hub_model', default='', help='Hugging Face repo with pytorch_model.bin to evaluate')
    parser.add_argument('--softmax', type=int, default=1, help='1 = sample actions, 0 = always take the best one')
    parser.add_argument('--search', default='top1', choices=['top1', 'instruction', 'model'],
                        help='search pages: first predicted query, the instruction, or let the model pick')
    parser.add_argument('--split', default='test', choices=['test', 'eval', 'train'])
    parser.add_argument('--num_goals', default=0, type=int, help='only use the first N goals (0 = all)')
    parser.add_argument('--out', default='', help='where to write the results as JSON')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_envs', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--amp', default=1, type=int)
    parser.add_argument('--get_image', default=0, type=int)

    parser.add_argument('--num', default=None, type=int)
    parser.add_argument('--human_goals', default=1, type=int)
    parser.add_argument('--step_limit', default=100, type=int)
    parser.add_argument('--extra_search_path', default='./data/goal_query_predict.json')
    return parser.parse_args()


def add_env_defaults(args):
    args.state_format = 'text_rich'
    args.click_item_name = 1
    args.num_prev_obs = 0
    args.num_prev_actions = 0
    args.ban_buy = 0
    args.go_to_item = 0
    args.go_to_search = 0
    args.harsh_reward = 0
    return args


def add_agent_defaults(args):
    args.network = 'bert'
    args.embedding_dim = 128
    args.hidden_dim = 128
    args.output_dir = '.'
    args.clip = 10
    args.w_pg = args.w_td = args.w_en = 1
    args.w_il = 0
    args.learning_rate = 1e-5
    args.gamma = 0.9
    args.token_budget = 0
    args.bert_path = 'scratch'
    return args


def main():
    args = add_env_defaults(parse_args())
    set_seed(args.seed)
    vec = VecWebEnv(args, args.num_envs, args.num_workers, seed=args.seed)
    goals = list(vec.goal_range(args.split))
    if args.num_goals:
        goals = goals[:args.num_goals]
    print(f'Evaluating on {len(goals)} {args.split} goals', flush=True)

    try:
        if args.policy == 'rule':
            act_batch = rule_policy
        elif args.policy == 'shopper':
            act_batch = CarefulShopper(queries=args.queries.split(','), inspect=args.inspect,
                                       read_details=bool(args.read_details))
        elif args.policy == 'value':
            act_batch = ValueShopper(ValueWeights.load(args.value_model), inspect=args.inspect, pages=args.pages,
                                     read_details=bool(args.read_details), step_limit=args.step_limit)
        else:
            from agent import Agent
            agent = Agent(add_agent_defaults(args))
            path = args.model_path
            if args.hub_model:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(args.hub_model, 'pytorch_model.bin')
            agent.load_weights(path)
            act_batch = make_policy(agent, 'softmax' if args.softmax else 'greedy', search=args.search)
        results = run_episodes(vec, list(range(args.num_envs)), goals, act_batch)
    finally:
        vec.close()

    summary = summarize(results)
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'summary': summary, 'results': results,
                       'settings': {'policy': args.policy, 'softmax': args.softmax, 'search': args.search, 'assist': args.assist,
                                    'truncation': args.truncation, 'inspect': args.inspect, 'queries': args.queries,
                                    'read_details': args.read_details, 'pages': args.pages,
                                    'value_model': args.value_model if args.policy == 'value' else '',
                                    'split': args.split, 'model': args.hub_model or args.model_path}},
                      f, indent=2)


if __name__ == '__main__':
    main()
