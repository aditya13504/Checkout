"""
Collects (visible features, reward received) for the products that search returns, by buying each one
in simulation with the options the instruction names.

    python train_value_model.py collect --split train --num_goals 2000 --out data/value_train.npz
    python train_value_model.py fit --data data/value_train.npz --out value_model.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from os.path import abspath, dirname, join

import numpy as np

sys.path.insert(0, join(dirname(abspath(__file__)), '..'))

from env import SEARCH_PRICE_TEXT, goal_range  # noqa: E402
from shopping_rules import best_options, instruction_of, price_range  # noqa: E402
from value_model import FULL_FEATURES, TITLE_FEATURES, Wanted, full_features, goal_context, phrase_text  # noqa: E402
from web_agent_site.engine.engine import get_top_n_product_from_keywords  # noqa: E402
from web_agent_site.engine.goal import get_reward  # noqa: E402
from web_agent_site.envs.web_agent_text_env import SimServer  # noqa: E402
from web_agent_site.utils import DEFAULT_FILE_PATH  # noqa: E402


def product_groups(product):
    return [[value.lower() for value in values] for values in product.get('options', {}).values() if values]


def product_details(product):
    return ' '.join(product.get('BulletPoints') or []) + ' ' + (product.get('Description') or '')


def candidates_of(server, queries, depth):
    """Each product returned by the queries, with its rank in each one."""
    found = {}
    for q, query in enumerate(queries):
        results = get_top_n_product_from_keywords(query.split(), server.search_engine, server.all_products,
                                                  server.product_item_dict, server.attribute_to_asins)
        for rank, product in enumerate(results[:depth]):
            found.setdefault(product['asin'], {})[q] = rank
    return found


def reward_of(server, asin, goal, instruction):
    product = server.product_item_dict[asin]
    chosen = dict(zip(product.get('options', {}), best_options(product_groups(product), instruction)))
    return get_reward(product, goal, price=server.product_prices.get(asin), options=chosen) * 100


def attribute_phrases(goals, min_goals=3):
    """Attribute phrases that several goals ask for. Single words are left to the word features."""
    counts = Counter(phrase_text(attribute).strip() for goal in goals for attribute in goal['attributes'])
    return sorted(p for p, count in counts.items() if count >= min_goals and ' ' in p)


def collect(args):
    server = SimServer('http://localhost', DEFAULT_FILE_PATH, num_products=None, human_goals=args.human_goals)
    with open(args.extra_search_path, encoding='utf-8') as f:
        predicted = {k.strip('.'): v for k, v in json.load(f).items()}
    ids = list(goal_range(len(server.goals), args.split))
    if args.split == 'train':  # an even sample of the training goals
        ids = ids[::max(len(ids) // args.num_goals, 1)]
    ids = ids[:args.num_goals]
    phrases = attribute_phrases(server.goals[i] for i in goal_range(len(server.goals), 'train'))
    print(f'{len(phrases)} attribute phrases from the training goals', flush=True)
    X, y, goal_ids, query_mask, ranks_out = [], [], [], [], []
    for n, idx in enumerate(ids):
        goal = server.goals[idx]
        text = goal['instruction_text']
        instruction, budget = instruction_of(text)
        wanted = Wanted(instruction, budget, phrases)
        queries = list(dict.fromkeys(predicted.get(text.split(SEARCH_PRICE_TEXT)[0], [])[:args.predicted_queries] + [text.lower()]))
        found = candidates_of(server, queries, args.depth)
        titles = [server.product_item_dict[a]['Title'] for a in found]
        context = goal_context(wanted, titles)
        for asin, ranks in found.items():
            product = server.product_item_dict[asin]
            X.append(full_features(wanted, product['Title'], price_range(str(product.get('Price', ''))), ranks, context,
                                   product_details(product), product_groups(product)))
            y.append(reward_of(server, asin, goal, instruction))
            goal_ids.append(idx)
            query_mask.append(sum(1 << q for q in ranks))
            ranks_out.append([ranks.get(q, 999) for q in range(args.predicted_queries + 1)])
        if (n + 1) % 100 == 0:
            print(f'{n + 1} goals, {len(y)} candidates', flush=True)
    os.makedirs(dirname(abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.float32), goal=np.array(goal_ids),
                        query_mask=np.array(query_mask), ranks=np.array(ranks_out), n_title=len(TITLE_FEATURES),
                        names=np.array(FULL_FEATURES), phrases=np.array(phrases))


def fit_linear(X, y, goals, alpha):
    """Ridge regression on differences within a goal, since a shopper only compares a goal's products."""
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale
    order = np.argsort(goals, kind='stable')
    starts = np.nonzero(np.r_[True, goals[order][1:] != goals[order][:-1]])[0]
    Z_within, y_within = Z[order].copy(), y[order].astype(np.float64)
    for a, b in zip(starts, np.r_[starts[1:], len(y)]):
        Z_within[a:b] -= Z_within[a:b].mean(axis=0)
        y_within[a:b] -= y_within[a:b].mean()
    weights = np.linalg.solve(Z_within.T @ Z_within + alpha * np.eye(X.shape[1]), Z_within.T @ y_within)
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist()}


def fit(args):
    data = np.load(args.data, allow_pickle=False)
    X, y, goals = data['X'].astype(np.float64), data['y'], data['goal']
    n_title = int(data['n_title'])
    models = {'title': fit_linear(X[:, :n_title], y, goals, args.alpha), 'full': fit_linear(X, y, goals, args.alpha)}
    for name, feature_names in (('title', TITLE_FEATURES), ('full', FULL_FEATURES)):
        models[name]['names'] = feature_names
    models['phrases'] = data['phrases'].tolist()
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(models, f, indent=1)
    print(f'{len(set(goals.tolist()))} goals, {len(y)} products: wrote {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['collect', 'fit'])
    parser.add_argument('--split', default='train', choices=['train', 'eval'])
    parser.add_argument('--num_goals', default=2000, type=int)
    parser.add_argument('--predicted_queries', default=1, type=int)
    parser.add_argument('--depth', default=20, type=int, help='results kept per query')
    parser.add_argument('--human_goals', default=1, type=int)
    parser.add_argument('--extra_search_path', default='./data/goal_query_predict.json')
    parser.add_argument('--data', help='fit: what collect wrote')
    parser.add_argument('--alpha', default=10.0, type=float)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    collect(args) if args.command == 'collect' else fit(args)


if __name__ == '__main__':
    main()
