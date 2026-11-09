"""
A learned estimate of what buying a product would earn: Q(instruction, buy this product).

The features use only what a shopper can see on the pages (instruction, titles, prices, option
buttons, features and description). The target is the reward actually received, measured on
training goals only (train_value_model.py).
"""
import json
import math

import numpy as np

from shopping_rules import (OPTION_MATCH, STOP_WORDS, instruction_of, is_item_page, is_search_page, item_actions, next_option_click,
                            option_groups, option_match, page_body, result_prices, tokens, within_budget)

PAGE_SIZE = 10  # products per results page
INSPECT_STEPS = 6  # open, features, back, description, back, back to the results
MAX_OPTION_CLICKS = 6  # option clicks allowed for when buying
PHRASE_FILLERS = {'a', 'an', 'and', 'are', 'be', 'for', 'in', 'is', 'of', 'the', 'to', 'with'}

TITLE_FEATURES = [
    'title_cover', 'title_cover_rare', 'title_cover_first', 'title_cover_last', 'title_bigrams', 'title_precision',
    'title_words', 'cover_vs_best', 'rare_vs_best', 'within_budget', 'price_vs_budget', 'has_budget',
    'best_rank', 'log_best_rank', 'query_hits', 'best_query',
]
FULL_FEATURES = TITLE_FEATURES + [
    'text_cover', 'text_cover_rare', 'text_bigrams', 'details_gain', 'log_details', 'all_cover', 'all_cover_rare',
    'n_groups', 'n_named', 'named_share', 'unnamed_groups', 'best_option_match', 'option_cover', 'option_cover_rare',
    'phrases_named', 'phrases_in_title', 'phrases_in_text',
]


class Wanted:
    """The content of an instruction, in a form that is quick to compare with page text.

    `phrases` is a list of attribute phrases learned from the training goals, such as 'long lasting';
    those that the instruction itself contains are looked for in a product's text.
    """

    def __init__(self, instruction, budget, phrases=()):
        self.instruction, self.budget = instruction, budget
        self.words = list(dict.fromkeys(w for w in tokens(instruction) if w not in STOP_WORDS))
        all_words = tokens(instruction)
        self.bigrams = list(dict.fromkeys(zip(all_words, all_words[1:])))
        plain = phrase_text(instruction)
        self.phrases = [p for p in phrases if f' {p}' in plain]


def rarity(titles):
    """How unusual each word is among the titles seen for this goal: common words tell products apart less."""
    n = len(titles)
    counts = {}
    for title in titles:
        for word in set(tokens(title)):
            counts[word] = counts.get(word, 0) + 1
    return {word: math.log((1 + n) / (1 + count)) + 1 for word, count in counts.items()}, math.log(1 + n) + 1


def cover(words, found, compact, weights=None, unseen=1.0):
    """The share of `words` present in a text; `weights` counts unusual words for more."""
    total = hit = 0.0
    for word in words:
        weight = 1.0 if weights is None else weights.get(word, unseen)
        total += weight
        if word in found or (len(word) > 3 and word in compact):
            hit += weight
    return hit / total if total else 0.0


def bigram_cover(bigrams, text_tokens):
    if not bigrams:
        return 0.0
    present = set(zip(text_tokens, text_tokens[1:]))
    return sum(b in present for b in bigrams) / len(bigrams)


def phrase_text(text):
    """A text in the form attribute phrases are written in: 'easy to use' becomes 'easy use'."""
    return ' ' + ' '.join(word for word in tokens(text) if word not in PHRASE_FILLERS)


def phrase_cover(phrases, text):
    """The share of `phrases` in a text. Only the start of a phrase's last word must match: 'long sleeved' has 'long sleeve'."""
    if not phrases:
        return 0.0
    plain = phrase_text(text)
    return sum(f' {p}' in plain for p in phrases) / len(phrases)


def _text_view(text):
    words = tokens(text)
    return words, set(words), ''.join(words)


def title_features(wanted, title, prices, ranks, weights, unseen, best_cover, best_rare):
    """`ranks` maps a query's position in the search order to the product's rank in that query's results."""
    words, found, compact = _text_view(title)
    title_cover = cover(wanted.words, found, compact)
    rare = cover(wanted.words, found, compact, weights, unseen)
    title_content = [w for w in words if w not in STOP_WORDS]
    wanted_set = set(wanted.words)
    budget = wanted.budget
    best_rank = min(ranks.values()) if ranks else 50
    return [
        title_cover, rare,
        cover(wanted.words[:3], found, compact), cover(wanted.words[-3:], found, compact),
        bigram_cover(wanted.bigrams, words),
        sum(w in wanted_set for w in title_content) / len(title_content) if title_content else 0.0,
        min(len(words), 60) / 60,
        title_cover - best_cover, rare - best_rare,
        within_budget(prices, budget),
        min(prices[0] / budget, 3.0) if budget and prices else 1.0,
        float(budget is not None),
        min(best_rank, 50) / 50, math.log1p(best_rank),
        float(len(ranks)),
        float(min(ranks, key=ranks.get)) if ranks else 0.0,
    ]


def goal_context(wanted, titles):
    """What depends on all the titles seen for a goal; compute once per goal."""
    weights, unseen = rarity(titles)
    best_cover = best_rare = 0.0
    for title in titles:
        _, found, compact = _text_view(title)
        best_cover = max(best_cover, cover(wanted.words, found, compact))
        best_rare = max(best_rare, cover(wanted.words, found, compact, weights, unseen))
    return weights, unseen, best_cover, best_rare


def full_features(wanted, title, prices, ranks, context, details, groups):
    """`groups` are the option values of each option group, `details` the features and description text."""
    weights, unseen, best_cover, best_rare = context
    base = title_features(wanted, title, prices, ranks, weights, unseen, best_cover, best_rare)
    words, found, compact = _text_view(title + ' ' + details)
    option_text = ' '.join(value for group in groups for value in group)
    _, option_found, option_compact = _text_view(option_text)
    _, all_found, all_compact = _text_view(title + ' ' + details + ' ' + option_text)
    matches = [max(option_match(value, wanted.instruction) for value in group) for group in groups if group]
    named = sum(score >= OPTION_MATCH for score in matches)
    text_cover = cover(wanted.words, found, compact)
    return base + [
        text_cover, cover(wanted.words, found, compact, weights, unseen),
        bigram_cover(wanted.bigrams, words),
        text_cover - base[0],
        math.log1p(len(details)) / 10,
        cover(wanted.words, all_found, all_compact), cover(wanted.words, all_found, all_compact, weights, unseen),
        float(min(len(groups), 6)), float(min(named, 6)),
        named / len(groups) if groups else 0.0,
        float(min(len(groups) - named, 6)),
        sum(matches) / (100 * len(matches)) if matches else 0.0,
        cover(wanted.words, option_found, option_compact),
        cover(wanted.words, option_found, option_compact, weights, unseen),
        float(min(len(wanted.phrases), 6)),
        phrase_cover(wanted.phrases, title),
        phrase_cover(wanted.phrases, title + ' ' + details),
    ]


class ValueWeights:
    """A linear model of the reward: the title model shortlists products, the full model decides."""

    def __init__(self, models, phrases=()):
        self.phrases = list(phrases)
        self.models = {name: {k: np.array(model[k], dtype=np.float64) for k in ('mean', 'scale', 'weights')}
                       for name, model in models.items()}

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            saved = json.load(f)
        return cls({name: saved[name] for name in ('title', 'full')}, saved.get('phrases', ()))

    def score(self, name, features):
        model = self.models[name]
        return float((np.array(features, dtype=np.float64) - model['mean']) / model['scale'] @ model['weights'])


class Candidate:
    def __init__(self, action, prices, location):
        self.action, self.prices, self.location = action, prices, location  # location = (query number, page)
        self.title = action[len('click[item - '):-1]
        self.ranks, self.value, self.shortlist_value = {}, None, 0.0


class Trip:
    """What a shopper remembers while serving one goal."""

    def __init__(self, goal_idx, valid):
        self.goal_idx, self.steps = goal_idx, 0
        self.searches = list(dict.fromkeys([valid[0], valid[-1]]))
        self.stage, self.scan_query, self.at = 'scan', 0, (0, 1)
        self.found, self.queue, self.best = {}, [], None
        self.opening, self.reading, self.read, self.details = None, False, set(), ''
        self.wanted, self.context, self.tried = None, None, set()


class ValueShopper:
    """Reads the results of two searches, opens the most promising products, buys the most valuable.

    Everything it decides from is on the pages it visits. One decision is made per step and each
    env has its own Trip, so this works with `run_episodes`.
    """

    def __init__(self, weights, inspect=8, pages=2, read_details=True, step_limit=100):
        self.weights, self.inspect, self.pages, self.read_details = weights, inspect, pages, read_details
        self.step_limit = step_limit
        # Leaves room to return to the best product and buy it, whichever that turns out to be.
        self.buy_reserve = 2 + pages + 1 + MAX_OPTION_CLICKS + 1
        self.trips = {}

    def __call__(self, obs_list, infos):
        return [self.decide(ob, info) for ob, info in zip(obs_list, infos)]

    def decide(self, ob, info):
        valid = info['valid']
        trip = self.trips.get(info['env_id'])
        if trip is None or trip.goal_idx != info['goal_idx']:
            trip = self.trips[info['env_id']] = Trip(info['goal_idx'], valid)
        trip.steps += 1
        if trip.wanted is None:
            trip.wanted = Wanted(*instruction_of(info['goal']), self.weights.phrases)
        if is_search_page(valid):
            return self.on_search_page(trip)
        if is_item_page(valid):
            return self.on_item_page(trip, ob, valid)
        if trip.reading:
            trip.details += ' ' + page_body(ob)
            trip.reading = False
            return 'click[< prev]'
        return self.on_results_page(trip, ob, valid)

    def on_search_page(self, trip):
        if trip.stage == 'scan':
            query = trip.scan_query
        else:
            target = self.target(trip)
            query = target.location[0] if target else 0
        trip.at = (query, 1)
        return trip.searches[query]

    def on_results_page(self, trip, ob, valid):
        if trip.stage == 'scan':
            action = self.scan(trip, ob, valid)
            if action:
                return action
        return self.travel(trip, valid)

    def scan(self, trip, ob, valid):
        """Notes the products on a results page, then moves on until every page has been read."""
        actions = item_actions(valid)
        prices = result_prices(ob)
        prices = prices if len(prices) == len(actions) else [None] * len(actions)
        query, page = trip.at
        for position, (action, price) in enumerate(zip(actions, prices)):
            found = trip.found.setdefault(action, Candidate(action, price, trip.at))
            found.ranks.setdefault(query, (page - 1) * PAGE_SIZE + position)
        if page < self.pages and 'click[next >]' in valid and actions:
            trip.at = (query, page + 1)
            return 'click[next >]'
        trip.scan_query += 1
        if trip.scan_query < len(trip.searches):
            return 'click[back to search]'
        self.make_shortlist(trip)
        return None

    def make_shortlist(self, trip):
        titles = [c.title for c in trip.found.values()]
        trip.context = goal_context(trip.wanted, titles)
        weights, unseen, best_cover, best_rare = trip.context
        for candidate in trip.found.values():
            candidate.shortlist_value = self.weights.score('title', title_features(
                trip.wanted, candidate.title, candidate.prices, candidate.ranks, weights, unseen, best_cover, best_rare))
        ranked = sorted(trip.found.values(), key=lambda c: -c.shortlist_value)[:self.inspect]
        trip.queue = sorted(ranked, key=lambda c: (c.location, -c.shortlist_value))  # the shortest way round
        trip.stage = 'inspect' if trip.queue else 'buy'

    def target(self, trip):
        if trip.stage == 'inspect' and not self.can_inspect_another(trip):
            trip.stage = 'buy'
        return trip.queue[0] if trip.stage == 'inspect' else trip.best

    def can_inspect_another(self, trip):
        return bool(trip.queue) and trip.steps + INSPECT_STEPS + 2 + self.pages + self.buy_reserve < self.step_limit

    def travel(self, trip, valid):
        """The next step towards the product in question, from a results page."""
        target = self.target(trip)
        if target is None:  # the searches found nothing, so there is nothing to buy
            return 'click[back to search]'
        if trip.at == target.location:
            if target.action in valid:
                if trip.stage == 'inspect':
                    trip.opening, trip.read, trip.details = trip.queue.pop(0), set(), ''
                return target.action
            if trip.stage == 'inspect':  # no longer here, so move on to the next
                trip.queue.pop(0)
                return self.travel(trip, valid)
            actions = item_actions(valid)
            return actions[0] if actions else 'click[back to search]'
        if trip.at[0] == target.location[0] and trip.at[1] < target.location[1] and 'click[next >]' in valid:
            trip.at = (trip.at[0], trip.at[1] + 1)
            return 'click[next >]'
        return 'click[back to search]'

    def on_item_page(self, trip, ob, valid):
        if trip.opening is None:  # buying
            action = next_option_click(ob, trip.wanted.instruction, skip=trip.tried)
            if action is None or trip.steps >= self.step_limit - 1 or len(trip.tried) >= MAX_OPTION_CLICKS:
                return 'click[buy now]'
            trip.tried.add(action)
            return action
        if self.read_details:
            for page in ('features', 'description'):
                action = f'click[{page}]'
                if page not in trip.read and action in valid:
                    trip.read.add(page)
                    trip.reading = True
                    return action
        candidate, trip.opening = trip.opening, None
        features = full_features(trip.wanted, candidate.title, candidate.prices, candidate.ranks, trip.context,
                                 trip.details, option_groups(ob))
        candidate.value = self.weights.score('full', features)
        if trip.best is None or candidate.value > trip.best.value:
            trip.best = candidate
        return 'click[< prev]'
