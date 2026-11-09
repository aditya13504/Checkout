"""
Live demo: watch an agent shop in the real WebShop store, one step at a time, in the browser.

    data\\venv\\Scripts\\python.exe demo\\live_demo.py        (or double-click demo\\START_DEMO.bat)

Then open http://127.0.0.1:3100 and run the value shopper on a test request, or on a request you type.
Every decision is made live on this machine; nothing is replayed from a file. The screen shows each page
in the store's current design, while the agent reads and acts on the environment's own pages, so what it
sees, and therefore its decisions and scores, are exactly those of the reported runs.

Needs the prepared full catalogue in data/, which run.ps1 or run.sh builds.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PREPARED = DATA / 'prepared_test'
DEMO = Path(__file__).resolve().parent
RUN = ROOT / 'kaggle_outputs' / 'outputs_full'  # the Kaggle run reported in the report
RUNS, RESULTS = RUN / 'runs', RUN / 'results'

# The environment reads these when it is imported, so they are set first.
os.environ.setdefault('WEBSHOP_ITEMS_PATH', str(PREPARED / 'items.json'))
os.environ.setdefault('WEBSHOP_ATTR_PATH', str(PREPARED / 'attrs.json'))
os.environ.setdefault('WEBSHOP_SQLITE_INDEX', str(PREPARED / 'webshop.sqlite'))
os.environ.setdefault('WEBSHOP_HUMAN_PATH', str(DATA / 'items_human_ins.json'))
os.environ.setdefault('WEBSHOP_PRODUCT_STORE', str(DATA / 'products.sqlite'))
os.environ.setdefault('HF_HOME', str(DATA / 'hf'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(ROOT / 'baseline_models'))
os.chdir(ROOT / 'baseline_models')  # the agents' data paths are relative to this folder

# (label, kind, weights, how search pages are handled, recorded test results)
AGENTS = {
    'value_shopper': ('Value shopper (learned value model)', 'value', RUNS / 'value' / 'value_model.json', None,
                      RESULTS / 'value_shopper.json'),
    'il_rl_greedy': ('IL+RL: BERT policy fine-tuned with A2C', 'model', RUNS / 'il_rl' / 'model_best.pt', 'model',
                     RESULTS / 'il_rl_greedy.json'),
    'rl_scratch_greedy': ('RL from pretrained BERT (A2C only)', 'model', RUNS / 'rl_scratch' / 'model_best.pt', 'model',
                          RESULTS / 'rl_scratch_greedy.json'),
    'il_seed0_greedy': ('IL: BERT policy copying people', 'model', RUNS / 'il_seed0' / 'model_best.pth', 'top1',
                        RESULTS / 'il_seed0_greedy.json'),
    'rule_softmax': ('Rule baseline', 'rule', None, None, RESULTS / 'rule_softmax.json'),
}
CUSTOM_GOAL = 999999  # the session number of requests typed by people; they have no answer key
BUDGET = re.compile(r'price lower than ([\d.]+) dollars')
SEARCH_PRICE_TEXT = ', and price lower than'  # as in baseline_models/env.py
# Typed requests say their budget in many ways ("under $70", "max 70 dollars", "$70 or less").
_PRICE_WORDS = (r"(?:under|below|less than|lower than|cheaper than|up to|at most|no more than|not more than|not over|"
                r"max(?:imum)?(?: of)?|within|budget(?: of| is)?|for less than|<=?)")
PRICE_BEFORE = re.compile(rf",?\s*(?:and\s+|with\s+|that\s+)?(?:a\s+)?(?:(?:price|priced|costing|costs?)\s+)?{_PRICE_WORDS}\s*"
                          r"(?:a\s+price\s+of\s+)?(?:\$|usd\s*)?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:\$|dollars?|usd|bucks)?", re.I)
PRICE_AFTER = re.compile(r",?\s*\$\s*(\d[\d,]*(?:\.\d+)?)\s*(?:or less|or under|or below|max(?:imum)?|tops|and under)", re.I)
FILLER = re.compile(r"^(?:hi|hello|please)?[\s,]*(?:i\s*(?:'m|am)?\s*(?:really\s+)?(?:need|want|would like|'d like|"
                    r"looking for|searching for|shopping for)|looking for|find me|get me|show me|buy me|buy|"
                    r"can you find(?: me)?)\s+(?:to buy\s+|to get\s+)?(?:an?\s+|some\s+|the\s+)?", re.I)


# ---------------------------------------------------------------- typed requests: product type and specs
# The value model was learned from the dataset's requests and scores a product mostly by the words its
# title shares with the request. On a typed request that can buy "a headphone case" for "headphones", or
# a 25 W charger for "a 100 watt charger". These two checks are added to its estimate for typed requests
# only; the test requests, and so the reported results, never use them.
UNITS = {'w': 'w', 'watt': 'w', 'watts': 'w', 'oz': 'oz', 'ounce': 'oz', 'ounces': 'oz', 'inch': 'inch', 'inches': 'inch',
         'ft': 'ft', 'feet': 'ft', 'foot': 'ft', 'mah': 'mah', 'gb': 'gb', 'tb': 'tb', 'mm': 'mm', 'cm': 'cm', 'lb': 'lb',
         'lbs': 'lb', 'pound': 'lb', 'pounds': 'lb', 'v': 'v', 'volt': 'v', 'volts': 'v', 'qt': 'qt', 'quart': 'qt',
         'quarts': 'qt', 'liter': 'l', 'liters': 'l', 'litre': 'l', 'litres': 'l', 'l': 'l', 'ml': 'ml', 'pack': 'pack',
         'packs': 'pack', 'pk': 'pack', 'pcs': 'pack', 'piece': 'pack', 'pieces': 'pack', 'count': 'pack', 'ct': 'pack'}
SPEC = re.compile(r'(\d+(?:\.\d+)?)\s*-?\s*(' + '|'.join(sorted(UNITS, key=len, reverse=True)) + r')\b', re.I)
PACK_OF = re.compile(r'\bpack of (\d+)\b', re.I)
NOT_PRODUCT = set(UNITS) | {'size', 'sizes', 'color', 'colour', 'colors', 'pack', 'set', 'piece', 'count', 'inch', 'men',
                            'mens', 'women', 'womens', 'kids', 'large', 'small', 'medium', 'xl', 'xxl', 'black', 'white',
                            'red', 'blue', 'green', 'brown', 'pink', 'grey', 'gray', 'one', 'two', 'new', 'best', 'good'}
ACCESSORIES = (r'case|cases|cover|covers|stand|holder|mount|strap|straps|skin|skins|protector|bag|pouch|sleeve|pads?|'
               r'cushions?|tips|replacement|parts?|accessor(?:y|ies)|remote|filters?|refills?|lid|lids|hooks?|clips?|'
               r'bracket|organizer|rack|decals?|stickers?|hanger|tray|earpads|charm|keychain')


def request_specs(text):
    """Numbers with units in a text, such as (100.0, 'w') for "100 watt" and (2.0, 'pack') for "2-pack"."""
    found = {(float(n), UNITS[u.lower()]) for n, u in SPEC.findall(text)}
    return found | {(float(n), 'pack') for n in PACK_OF.findall(text)}


def request_head(query):
    """The kind of product a request asks for: the last real noun of its first phrase ("charger")."""
    for part in re.split(r'\b(?:with|for|in|of|that|which|to|from|on|without|under|by)\b', query.lower()):
        words = [w for w in re.findall(r"[a-z][a-z'-]*", part) if w not in NOT_PRODUCT and len(w) > 2]
        if words:
            word = words[-1].split('-')[-1]
            if word.endswith(('ches', 'shes', 'sses', 'xes')):
                return word[:-2]
            if word.endswith('ies'):
                return word[:-3] + 'y'
            return word[:-1] if word.endswith('s') and not word.endswith('ss') else word
    return None


def head_role(text, head):
    """'head' if the title uses the word as the product itself at least once, 'modifier' if it only uses it
    to describe another noun ("headphone hat"), None if it never uses it. Uses the store's spaCy tagger."""
    from web_agent_site.engine.goal import nlp
    forms, found = {head, head + 's', head + 'es'}, False
    tokens = list(nlp(text[:400]))
    for i, token in enumerate(tokens):
        if token.text in forms:
            found = True
            after = tokens[i + 1] if i + 1 < len(tokens) else None
            if after is None or after.pos_ not in ('NOUN', 'PROPN') or after.like_num or after.text in UNITS:
                return 'head'
    return 'modifier' if found else None


def typed_check(title, extra, head, specs, request_words):
    """Points added to the value model's estimate of buying a product, for a typed request."""
    text, points = title.lower(), 0.0
    if head:
        role = head_role(text, head)
        if role is None:
            points -= 25  # the title never names the kind of product asked for
        elif role == 'modifier':
            points -= 25  # it only describes another product: a "headphone hat", a "charger hub"
        else:
            after = re.search(rf'\b{re.escape(head)}(?:s|es)?\W+({ACCESSORIES})\b', text)
            before = re.search(rf'\b({ACCESSORIES})\b[^|]{{0,60}}\b(?:for|compatible|fits?)\b[^|]{{0,80}}\b{re.escape(head)}', text)
            accessory = after or before
            if accessory and accessory.group(1) not in request_words:
                points -= 35  # an accessory for that kind of product, such as a case for headphones
    have = request_specs(text + ' ' + extra)
    for amount, unit in specs:
        if (amount, unit) in have:
            points += 12
        elif any(u == unit for _, u in have):
            points -= 15  # the same measure with a different amount, such as 25 W for 100 W
    return points


def typed_request_shopper(base, query):
    """The value shopper, with the product-type and spec checks added to its estimates."""
    head, specs = request_head(query), request_specs(query)
    words = set(re.findall(r'[a-z]+', query.lower()))
    from shopping_rules import option_groups

    class TypedRequestShopper(base):
        def make_shortlist(self, trip):
            super().make_shortlist(trip)
            for candidate in trip.found.values():
                candidate.shortlist_value += typed_check(candidate.title, '', head, specs, words)
            ranked = sorted(trip.found.values(), key=lambda c: -c.shortlist_value)[:self.inspect]
            trip.queue = sorted(ranked, key=lambda c: (c.location, -c.shortlist_value))
            trip.stage = 'inspect' if trip.queue else 'buy'

        def on_item_page(self, trip, ob, valid):
            candidate = trip.opening
            action = super().on_item_page(trip, ob, valid)
            if candidate is not None and trip.opening is None and candidate.value is not None:  # just scored
                candidate.value += typed_check(candidate.title, str(option_groups(ob)), head, specs, words)
                trip.best = max((c for c in trip.found.values() if c.value is not None), key=lambda c: c.value)
            return action
    return TypedRequestShopper


STOP = {'a', 'an', 'the', 'and', 'or', 'for', 'with', 'in', 'of', 'to', 'on', 'my', 'me', 'i', 'that', 'is', 'it', 'its',
        'size', 'color', 'colour', 'pack', 'one', 'some', 'any', 'very', 'new', 'good', 'best', 'please'}


def estimated_match(product, price, chosen, query, budget):
    """Grades a purchase for a typed request with the store's own formula, (attributes + options + price) /
    parts x product type, where the parts come from the request's words. There is no answer key for a typed
    request, so this is an estimate; the breakdown shows each check."""
    plain = lambda text: re.sub(r"['’`]", '', str(text).lower())  # noqa: E731  ("men’s" is "men's" is "mens")
    query = plain(query)
    head = request_head(query)
    title = product.get('Title', '').lower()
    details = plain(' '.join([title, *map(str, product.get('BulletPoints') or []), str(product.get('Description', '')),
                              *map(str, product.get('Attributes') or [])]))
    role = head_role(title, head) if head else 'head'
    if role == 'head':
        r_type, type_text = 1.0, 'the kind asked for'
    elif role == 'modifier' or (head and re.search(rf'\b{re.escape(head)}', details)):
        r_type, type_text = 0.5, 'a related product'
    else:
        r_type, type_text = 0.1, 'a different kind of product'
    option_values = ' | '.join(plain(v) for values in (product.get('options') or {}).values() for v in values)
    chosen_values = ' | '.join(plain(v) for v in chosen.values())
    words = re.findall(r"[a-z][a-z'-]*", re.sub(r'\b\d+(?:\.\d+)?\s*-?\s*[a-z]+\b', ' ', query.lower()))
    head_forms = {head, f'{head}s', f'{head}es'} if head else set()
    wanted = [w for w in dict.fromkeys(words) if w not in STOP and w not in head_forms and w not in UNITS and len(w) > 1]
    wanted += [m.group(1) for m in re.finditer(r'\bsize\s+(\w+)', query.lower())]  # "size 10"
    options, attributes = [], []
    for word in wanted:
        if re.search(rf'\b{re.escape(word)}\b', option_values):
            options.append([word, bool(re.search(rf'\b{re.escape(word)}\b', chosen_values))])
        else:
            attributes.append([word, bool(re.search(rf'\b{re.escape(word)}', details))])
    have = request_specs(details + ' ' + option_values)
    for amount, unit in sorted(request_specs(query)):
        label = f'{amount:g} {unit}'
        attributes.append([label, (amount, unit) in have])
    price_ok = None if budget is None or price is None else price <= budget
    parts = len(attributes) + len(options) + (price_ok is not None)
    hits = sum(ok for _, ok in attributes) + sum(ok for _, ok in options) + bool(price_ok)
    score = round(100 * r_type * (hits / parts if parts else 1.0), 1)
    return {'score': score, 'type': [r_type, type_text, head], 'attributes': attributes, 'options': options,
            'price': [price_ok, price, budget]}


def normalize_request(text):
    """Writes a typed request the way the dataset's requests are written, which is what the value model
    learned from: the budget at the end as ", and price lower than N dollars". Also returns a search query
    without the budget and the filler words, and the budget. Test requests never pass through here."""
    text = ' '.join(text.split()).strip(' .').lower()
    match = BUDGET.search(text) or PRICE_BEFORE.search(text) or PRICE_AFTER.search(text)
    budget = float(match.group(1).replace(',', '')) if match else None
    core = (text[:match.start()] + ' ' + text[match.end():]) if match else text
    core = re.sub(r'[\s,;.]*(?:,?\s*and)?[\s,;.]*$', '', ' '.join(core.split())).strip(' ,;.')
    query = FILLER.sub('', core).strip(' ,;.') or core
    # Titles write these specs as one word ("100W", "20V", "10000mAh"), and the search matches whole words.
    query = re.sub(r'\b(\d+(?:\.\d+)?)\s*-?\s*(watts?|volts?|mah|gb|tb)\b',
                   lambda m: m.group(1) + {'watt': 'w', 'watts': 'w', 'volt': 'v', 'volts': 'v'}.get(m.group(2), m.group(2)),
                   query)
    instruction = f'{core}{SEARCH_PRICE_TEXT} {budget:.2f} dollars' if budget else core
    return instruction, query, budget
STAGES = {'scan': 'Reading the search results', 'inspect': 'Opening the most promising products',
          'buy': 'Going back to buy the best one'}


class Demo:
    def __init__(self):
        self.lock = threading.Lock()
        self.events, self.status, self.env, self.goals = [], 'Loading the store: 1,181,430 products…', None, []
        self.run_id, self.delay, self.stop = 0, 1.0, False
        self.models, self.recorded = {}, {}
        self.thread = None
        self.screen, self.step_no, self.goal_idx, self.custom, self.custom_budget = None, 0, 0, False, None
        self.custom_query, self.custom_text = '', ''

    # ------------------------------------------------------------------ setup
    def load(self):
        try:
            from env import WebEnv
            args = SimpleNamespace(state_format='text_rich', click_item_name=1, num_prev_obs=0, num_prev_actions=0,
                                   ban_buy=0, go_to_item=0, go_to_search=0, harsh_reward=0, num=None, human_goals=1,
                                   get_image=0, step_limit=100, extra_search_path='./data/goal_query_predict.json',
                                   assist=0)
            t0 = time.time()
            self.env = WebEnv(args, split='test', id='demo_')
            self.goals = [g['instruction_text'] for g in self.env.env.server.goals[:500]]
            self.install_screen()
            for key, (*_, path) in AGENTS.items():
                if path.exists():
                    with open(path, encoding='utf-8') as f:
                        self.recorded[key] = {r['goal_idx']: r['score'] for r in json.load(f)['results']}
            self.status = 'ready'
            print(f'Store ready in {time.time() - t0:.0f} s. Open the viewer and press Start.', flush=True)
        except Exception:
            self.status = 'The store failed to load:\n' + traceback.format_exc()
            print(self.status, flush=True)

    # ----------------------------------------------------------------- screen
    def install_screen(self):
        """Shows each page the agent gets in the store's current design, for people watching. Display only:
        the environment still hands the agent its own pages, which it reads and acts on as before."""
        import web_agent_site.envs.web_agent_text_env as text_env
        from web_agent_site.app import app as store_app
        from web_agent_site.engine.engine import END_BUTTON, parse_action, render_human_page
        agent_render, server = text_env.map_action_to_html, self.env.env.server

        def render(action, **kwargs):
            html = agent_render(action, **kwargs)
            try:
                name, arg = parse_action(action)
                goal = server.user_sessions.get(kwargs.get('session_id'), {}).get('goal', {})
                context = dict(kwargs, human=True, viewer='agent', steps=self.step_no, show_attrs=False,
                               budget=self.custom_budget if self.custom else goal.get('price_upper'),
                               request_no=None if self.custom else self.goal_idx + 1,
                               catalogue_size=len(server.product_item_dict))
                if name == 'click' and arg == END_BUTTON:
                    context.update(self.done_screen(server, kwargs))
                with store_app.test_request_context():
                    self.screen = render_human_page(name, arg, context)
            except Exception:
                self.screen = None  # the viewer then shows the environment's own page
                traceback.print_exc()
            return html
        text_env.map_action_to_html = render

    def done_screen(self, server, kwargs):
        """What the store's result page needs, from the finished session."""
        session = server.user_sessions[kwargs['session_id']]
        goal, info, asin = session['goal'], session.get('verbose_info', {}), kwargs['asin']
        options = goal.get('goal_options', [])
        options = list(options.values() if isinstance(options, dict) else options)
        return dict(score=round(kwargs['reward'] * 100, 1), product=server.product_item_dict[asin],
                    price_paid=server.product_prices.get(asin) or 0.0, goal_attributes=goal.get('attributes', []),
                    goal_options=options, att_hits=round((info.get('r_att') or 0) * len(goal.get('attributes', []))),
                    opt_hits=round(info.get('r_option', 0) * len(options)), r_type=info.get('r_type', 0),
                    r_price=info.get('r_price'), agents=[], custom=self.custom, retry_url='#', demo_url='#',
                    match=self.custom_match() if self.custom else None)

    def custom_match(self):
        """The estimated match of a typed request's purchase, or None if nothing was bought."""
        server = self.env.env.server
        session = server.user_sessions.get(f'demo_{CUSTOM_GOAL}', {})
        asin = session.get('asin')
        if not asin or 'verbose_info' not in session:
            return None
        return estimated_match(server.product_item_dict[asin], server.product_prices.get(asin),
                               session.get('options', {}), self.custom_query, self.custom_budget)

    def custom_goal(self, text):
        """A goal for a typed request: the request's text and budget. There is no correct product, so the
        rest is borrowed from a test goal only to let the store compute a reward, which is never shown."""
        goal = dict(self.env.env.server.goals[0])
        budget = BUDGET.search(text)
        goal.update(instruction_text=text, goal_options=[], asin='',
                    price_upper=float(budget.group(1)) if budget else 1e9)
        return goal

    def model(self, key):
        if key not in self.models:
            import torch
            from agent import Agent
            from test import add_agent_defaults
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            args = add_agent_defaults(SimpleNamespace(get_image=0, amp=0, truncation='right'))
            agent = Agent(args)
            agent.load_weights(str(AGENTS[key][2]))
            self.models[key] = agent
        return self.models[key]

    # ------------------------------------------------------------- publishing
    def publish(self, run_id, event):
        with self.lock:
            if run_id == self.run_id:
                event['run'] = run_id
                self.events.append(event)

    def target(self, action):
        """What to highlight on the page: the search box, a product, a button or an option."""
        if action.startswith('search['):
            return {'kind': 'search', 'text': action[7:-1]}
        if action.startswith('click['):
            arg = action[6:-1]
            if arg.startswith('item - '):
                return {'kind': 'item', 'text': self.env.name2asin.get(arg[7:], arg[7:])}
            return {'kind': 'click', 'text': arg}
        return {'kind': 'none', 'text': ''}

    # ---------------------------------------------------------------- deciding
    def policy(self, key):
        kind = AGENTS[key][1]
        if kind == 'rule':
            from test import rule_policy

            def act(ob, info):
                action = rule_policy([ob], [info])[0]
                why = ('Searches with the whole request, then buys the first result without choosing options.')
                return action, {'kind': 'rule', 'why': why}
            return act
        if kind == 'value':
            from value_model import ValueShopper, ValueWeights
            # The settings of the reported evaluation (kaggle/pipeline.py, evaluate).
            # Typed requests add the product-type and spec checks; test requests use it exactly as evaluated.
            shopper_class = typed_request_shopper(ValueShopper, self.custom_query) if self.custom else ValueShopper
            shopper = shopper_class(ValueWeights.load(str(AGENTS[key][2])), inspect=14, pages=2, read_details=True,
                                    step_limit=100)

            def act(ob, info):
                action = shopper([ob], [info])[0]
                trip = shopper.trips[0]
                scored = sorted((c for c in trip.found.values() if c.value is not None), key=lambda c: -c.value)
                shortlist = sorted(trip.found.values(), key=lambda c: -c.shortlist_value)[:6]
                return action, {
                    'kind': 'value', 'stage': STAGES.get(trip.stage, trip.stage), 'seen': len(trip.found),
                    'left': len(trip.queue), 'opening': trip.opening.title if trip.opening else None,
                    'scored': [[c.title, round(c.value, 1)] for c in scored[:8]],
                    'shortlist': [[c.title, round(c.shortlist_value, 1)] for c in shortlist] if trip.stage != 'scan' else [],
                    'best': trip.best.title if trip.best else None,
                }
            return act

        import torch
        agent, search = self.model(key), AGENTS[key][3]

        def act(ob, info):
            valid = info['valid']
            if len(valid) == 1:
                return valid[0], {'kind': 'forced', 'why': 'Only one action is possible here.'}
            if valid[0].startswith('search[') and search == 'top1':
                return valid[0], {'kind': 'forced', 'why': 'Uses the first search query predicted for this request.'}
            states, ids = agent.build_states([ob], [info]), agent.encode_valids([valid])
            with torch.no_grad():
                log_probs, _, values = agent.network.rl_forward(states, ids, value=True, act=True)
            probs = log_probs.float().exp().tolist()
            best = max(range(len(probs)), key=probs.__getitem__)  # greedy, as in the reported runs
            ranked = sorted(zip(valid, probs), key=lambda x: -x[1])[:8]
            return valid[best], {'kind': 'policy', 'probs': [[a, round(p, 4)] for a, p in ranked],
                                 'n': len(valid), 'value': round(float(values.float().reshape(-1)[0]), 2)}
        return act

    # ----------------------------------------------------------------- running
    def run(self, run_id, key, goal_idx, request=None):
        try:
            label, env = AGENTS[key][0], self.env
            self.custom, self.step_no, self.screen = request is not None, 0, None
            if self.custom:
                self.custom_text = request  # as typed, which the viewer's history keeps
                request, query, self.custom_budget = normalize_request(request)
                self.custom_query = query
                # Its first search uses the cleaned query, as test requests use their predicted one.
                env.extra_search[request.split(SEARCH_PRICE_TEXT)[0]] = [query]
                self.goal_idx, goal_idx = -1, CUSTOM_GOAL
                env.env.server.user_sessions[f'demo_{CUSTOM_GOAL}'] = {'goal': self.custom_goal(request), 'done': False}
            else:
                self.goal_idx = goal_idx
            self.publish(run_id, {'type': 'start', 'agent': label, 'key': key, 'custom': self.custom,
                                  'typed': self.custom_text if self.custom else None,
                                  'goal': None if self.custom else goal_idx,
                                  'instruction': request if self.custom else self.goals[goal_idx],
                                  'recorded': None if self.custom else self.recorded.get(key, {}).get(goal_idx)})
            act = self.policy(key)
            ob, info = env.reset(goal_idx)
            step = 0
            while not self.stop:
                info.update(env_id=0, goal_idx=goal_idx)
                action, think = act(ob, info)
                step += 1
                self.publish(run_id, {'type': 'step', 'n': step, 'action': action, 'target': self.target(action),
                                      'url': env.env.state['url'], 'html': self.screen or env.env.state['html'],
                                      'think': think})
                self.wait()
                self.step_no = step
                ob, _, done, info = env.step(action)
                if done:
                    verbose = info.get('verbose', {})
                    self.publish(run_id, {
                        'type': 'done', 'score': round(float(info['score']) * 10, 1), 'steps': info.get('steps', step),
                        'bought': bool(info.get('bought')), 'html': self.screen or env.env.state['html'],
                        'custom': self.custom, 'match': self.custom_match() if self.custom else None,
                        'parts': {k: round(float(verbose.get(k, 0.0)), 3) for k in ('r_type', 'r_att', 'r_option', 'r_price')}})
                    return
        except Exception:
            self.publish(run_id, {'type': 'error', 'message': traceback.format_exc()})
            print(traceback.format_exc(), flush=True)

    def wait(self):
        start = time.time()  # the pace is read again while waiting, so a change applies at once
        while time.time() < start + self.delay and not self.stop:
            time.sleep(0.03)

    def start(self, key, goal_idx, request=None):
        if self.thread and self.thread.is_alive():
            self.stop = True
            self.thread.join()
        with self.lock:
            self.run_id += 1
            self.events = []
        self.stop = False
        self.thread = threading.Thread(target=self.run, args=(self.run_id, key, goal_idx, request), daemon=True)
        self.thread.start()


DEMO_STATE = Demo()


STATIC_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css',
                '.json': 'application/json', '.md': 'text/plain; charset=utf-8', '.png': 'image/png',
                '.svg': 'image/svg+xml', '.pdf': 'application/pdf'}


# One address for everything, as on a hosted Space: the store runs inside at CHECKOUT_STORE_INTERNAL, and every
# path that is not the viewer's goes to it. Locally the store has its own address, CHECKOUT_STORE_URL.
STORE_INTERNAL = os.environ.get('CHECKOUT_STORE_INTERNAL', '').rstrip('/')
STORE_PUBLIC = os.environ.get('CHECKOUT_STORE_URL', 'http://127.0.0.1:3000').rstrip('/')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def proxy(self, method, raw=None):
        """Passes a request to the store and its answer back, redirects included."""
        import urllib.error
        import urllib.request

        class Keep(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        headers = {'Content-Type': self.headers['Content-Type']} if self.headers.get('Content-Type') else {}
        request = urllib.request.Request(STORE_INTERNAL + self.path, data=raw if method == 'POST' else None,
                                         method=method, headers=headers)
        try:
            answer = urllib.request.build_opener(Keep).open(request, timeout=900)
            code, head, data = answer.status, answer.headers, answer.read()
        except urllib.error.HTTPError as error:
            code, head, data = error.code, error.headers, error.read()
        except OSError:
            return self.send('The store is still starting. Try again in a minute.', 'text/plain; charset=utf-8', 503)
        self.send_response(code)
        for name in ('Content-Type', 'Location'):
            if head.get(name):
                self.send_header(name, head[name].replace(STORE_INTERNAL, ''))
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send(self, body, kind='application/json', code=200):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/':
            return self.send((DEMO / 'viewer.html').read_bytes(), 'text/html; charset=utf-8')
        if url.path.startswith('/static/'):
            path = (ROOT / 'web_agent_site' / url.path.lstrip('/')).resolve()
            if path.is_file() and (ROOT / 'web_agent_site' / 'static') in path.parents:
                return self.send(path.read_bytes(), STATIC_TYPES.get(path.suffix, 'application/octet-stream'))
            return self.send('{}', code=404)
        if url.path == '/api/setup':
            d = DEMO_STATE
            return self.send(json.dumps({
                'status': d.status, 'goals': d.goals if d.status == 'ready' else [],
                'agents': [{'key': k, 'label': v[0]} for k, v in AGENTS.items()],
                'recorded': {k: {str(g): s for g, s in r.items()} for k, r in d.recorded.items()},
                'store': '' if STORE_INTERNAL else STORE_PUBLIC}))
        if url.path == '/api/events':
            since = int(parse_qs(url.query).get('since', ['0'])[0])
            run = int(parse_qs(url.query).get('run', ['0'])[0])
            with DEMO_STATE.lock:
                current = DEMO_STATE.run_id
                events = DEMO_STATE.events[since:] if run == current else DEMO_STATE.events
            return self.send(json.dumps({'run': current, 'events': events, 'reset': run != current,
                                         'delay': DEMO_STATE.delay}))
        if STORE_INTERNAL:
            return self.proxy('GET')
        return self.send('{}', code=404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw, url = self.rfile.read(length), urlparse(self.path)
        if STORE_INTERNAL and not url.path.startswith('/api/'):
            return self.proxy('POST', raw)
        body = json.loads(raw or b'{}')
        if url.path == '/api/start':
            if DEMO_STATE.status != 'ready':
                return self.send(json.dumps({'error': 'The store is still loading.'}), code=409)
            key, request = body.get('agent', 'value_shopper'), ' '.join(str(body.get('request') or '').split())
            if request:  # a request typed by a person
                if key != 'value_shopper' or not 3 <= len(request) <= 300:
                    return self.send(json.dumps({'error': 'Type a request of 3 to 300 characters.'}), code=400)
                DEMO_STATE.delay = float(body.get('delay', DEMO_STATE.delay))
                DEMO_STATE.start(key, None, request)
                return self.send(json.dumps({'run': DEMO_STATE.run_id}))
            goal = int(body.get('goal', 0))
            if key not in AGENTS or not 0 <= goal < len(DEMO_STATE.goals):
                return self.send(json.dumps({'error': 'Unknown agent or request.'}), code=400)
            DEMO_STATE.delay = float(body.get('delay', DEMO_STATE.delay))
            DEMO_STATE.start(key, goal)
            return self.send(json.dumps({'run': DEMO_STATE.run_id}))
        if url.path == '/api/speed':
            DEMO_STATE.delay = max(0.0, min(5.0, float(body.get('delay', 1.0))))
            return self.send('{}')
        if url.path == '/api/stop':
            DEMO_STATE.stop = True
            return self.send('{}')
        return self.send('{}', code=404)


def main():
    parser = argparse.ArgumentParser(description='WebShop live agent demo')
    parser.add_argument('--port', type=int, default=3100)
    parser.add_argument('--host', default='127.0.0.1', help='address to listen on (0.0.0.0 inside a container)')
    parser.add_argument('--no_browser', action='store_true')
    args = parser.parse_args()
    for path in (PREPARED / 'webshop.sqlite', DATA / 'products.sqlite'):
        if not path.exists():
            sys.exit(f'{path} is missing. Run run.ps1 (Windows) or run.sh (macOS) first.')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=DEMO_STATE.load, daemon=True).start()
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Live demo at {url}  (loading the store takes a minute or two; press Ctrl+C to stop)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
