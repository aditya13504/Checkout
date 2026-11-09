import sys
import json
import random
from os.path import join, dirname, abspath
from collections import defaultdict

# connect to WebShop env
MODEL_PATH = dirname(abspath(__file__))
SITE_PATH = join(MODEL_PATH, '../')
sys.path.insert(0, SITE_PATH)

from shopping_rules import masked_valid, next_option_click, instruction_of
from web_agent_site.envs import WebAgentTextEnv

SEARCH_PRICE_TEXT = ", and price lower than"
MAX_VALID_ACTIONS = 20  # larger action sets are reduced to 16, as in imitation training
EMPTY_VERBOSE = {'r_att': 0.0, 'r_option': 0.0, 'r_price': 0.0, 'r_type': 0.0,
                 'w_att': 0.0, 'w_option': 0.0, 'w_price': 0.0}


def goal_range(n_goals, split):
    """The goals of a data split.

    Goals are shuffled with a fixed seed, so these are the paper's splits when the
    catalogue contains all 12,087 human goals. A small catalogue (a quick test) gets
    proportional splits instead, since the paper's would be empty.
    """
    if n_goals >= 2000:
        test_end, eval_end = 500, 1500
    else:
        test_end = max(1, n_goals // 10)
        eval_end = max(test_end + 1, n_goals // 5)
    ranges = {
        'test': range(0, test_end),
        'eval': range(test_end, eval_end),
        'train': range(eval_end, n_goals),
    }
    return ranges[split]


def shared_name_maps(server):
    """asin <-> title lookups, stored once on the server rather than once per env."""
    if not hasattr(server, 'asin2name'):
        products = server.product_item_dict
        titles = products.titles() if hasattr(products, 'titles') else ((k, v['Title']) for k, v in products.items())
        server.asin2name = {asin.lower(): title.lower() for asin, title in titles}
        server.name2asin = {v: k for k, v in server.asin2name.items()}
    return server.asin2name, server.name2asin


class WebEnv:
    ''' A wrapper of textEnv for models. Returns valid actions at each step of the game. '''

    def __init__(self, args, split='train', server=None, id=None):
        self.env = WebAgentTextEnv(observation_mode=args.state_format, server=server,
                                   filter_goals=None, limit_goals=-1,
                                   num_products=args.num, human_goals=args.human_goals,
                                   get_image=args.get_image,
                                   num_prev_obs=args.num_prev_obs, num_prev_actions=args.num_prev_actions,
                                   session_prefix=id)
        self.goal_idxs = goal_range(len(self.env.server.goals), split)
        self.steps = 0
        self.step_limit = args.step_limit
        self.stats = defaultdict(int)  # kept across episodes
        self.session = None
        self.click_item_name = args.click_item_name
        self.asin2name, self.name2asin = shared_name_maps(self.env.server)
        self.harsh_reward = args.harsh_reward
        self.go_to_item = args.go_to_item
        self.go_to_search = args.go_to_search
        self.ban_buy = args.ban_buy
        self.get_image = args.get_image
        self.assist = bool(getattr(args, 'assist', 0))

        if args.extra_search_path != "":
            server_cache = self.env.server
            if not hasattr(server_cache, 'extra_search'):
                with open(args.extra_search_path, encoding='utf-8') as f:
                    server_cache.extra_search = {k.strip("."): v for k, v in json.load(f).items()}
            self.extra_search = server_cache.extra_search
        else:
            self.extra_search = None

    def get_search_texts(self, atts, query, inst):
        if self.extra_search is not None:
            inst_ = inst.split(SEARCH_PRICE_TEXT)[0] if SEARCH_PRICE_TEXT in inst else inst
            return self.extra_search.get(inst_, []) + [inst.lower()]
        return [query] + [f'{att} {query}' for att in atts] + [inst.lower()]

    def select_named_options(self, ob):
        """Assist mode: clicks the options the instruction names, as part of arriving at a page."""
        if not self.assist:
            return ob
        instruction, _ = instruction_of(self.env.instruction_text)
        made = set()
        while not self.step_limit or self.steps + 1 < self.step_limit:
            action = next_option_click(ob, instruction, skip=made)
            if action is None:
                break
            made.add(action)
            ob, _, _, _ = self.env.step(action)
            self.steps += 1
        return ob

    def present_valid(self, ob, valid):
        """Assist mode: hides products that the page shows to be over budget."""
        return masked_valid(ob, valid, self.env.instruction_text) if self.assist else valid

    def get_valid_actions(self):
        valid_info = self.env.get_available_actions()
        if valid_info['has_search_bar']:  # only search action available
            goal = self.session['goal']
            texts = self.get_search_texts(goal['attributes'], goal['query'], goal['instruction_text'])
            return [f'search[{text}]' for text in texts]

        valids = []
        for text in valid_info['clickables']:
            if text == 'search':
                continue
            # ban buy when options not completed
            if text == 'buy now' and self.ban_buy:
                product = self.env.server.product_item_dict[self.session["asin"]]
                if len(self.session['options']) != len(product['options']):
                    continue
            if self.click_item_name and text in self.asin2name:
                text = 'item - ' + self.asin2name[text]
            valids.append(f'click[{text}]')
        # action space reduction: keep the navigation buttons and sample the rest
        if len(valids) > MAX_VALID_ACTIONS:
            valids = valids[:6] + random.sample(valids[6:], 10)
        return valids or ['finish']

    def step(self, action):
        if self.click_item_name and action.startswith('click[item - ') and action[13:-1] in self.name2asin:
            action = f'click[{self.name2asin[action[13:-1]]}]'

        ob, reward, done, info = self.env.step(action)
        info = info or {}

        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        if not done:
            ob = self.select_named_options(ob)
        if done:
            verbose = dict(self.session.get('verbose_info', EMPTY_VERBOSE))
            verbose['r_harsh'] = (reward == 1)
            verbose['r_exact'] = (reward == 1) and (self.session['goal']['asin'] == self.session['asin'])
            info['verbose'] = verbose
            info['category'] = self.session['goal']['category']
            if self.harsh_reward:
                reward = float(verbose['r_harsh'])
            for k, v in self.session['actions'].items():
                self.stats[f'action_{k}'] += v
            info['actions'] = dict(self.session['actions'])
            info['steps'] = self.steps
            info['bought'] = 'verbose_info' in self.session

        info.update({'valid': self.present_valid(ob, self.get_valid_actions()), 'goal': self.env.instruction_text,
                     'score': reward * 10})

        if self.get_image:
            info['image_feat'] = self.env.get_image()

        return ob, reward * 10, done, info

    def reset(self, idx=None):
        if idx is None:
            idx = random.choice(self.goal_idxs)
        ob, info = self.env.reset(idx)
        self.session = self.env.server.user_sessions[self.env.session]
        self.session.pop('verbose_info', None)  # sessions are reused per goal
        info = info or {}
        info.update({'valid': self.get_valid_actions(), 'goal': self.env.instruction_text, 'score': 0})
        if self.get_image:
            info['image_feat'] = self.env.get_image()
        self.steps = 0
        if self.go_to_search or self.go_to_item:
            name = self.session['goal']['name'].lower()
            ob, _, _, info = self.step(f'search[{name}]')
            self.stats['action_go_to_search'] += 1
            if self.go_to_item:
                asin = self.session['goal']['asin'].lower()
                if asin in self.env.get_available_actions()['clickables']:
                    ob, _, _, info = self.step(f'click[{asin}]')
                    self.stats['action_go_to_item'] += 1
        return ob, info

    def close(self):
        self.env.close()
