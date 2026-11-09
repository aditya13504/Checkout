"""
Domain knowledge that reads only what a shopper can see on the page: the instruction, the
prices shown and the option buttons. It never looks at the hidden goal or the reward.

- `masked_valid` and `next_option_click` are used by WebEnv's `assist` mode, for any agent:
  products over budget are hidden, and options named in the instruction are clicked.
- `CarefulShopper` needs no learning. It opens several results, compares them with the
  instruction, then selects options and buys the best one. It is a strong baseline.
"""
import re

from thefuzz import fuzz

NAVIGATION = {'back to search', '< prev', 'next >', 'description', 'features', 'reviews', 'attributes', 'buy now'}
OPTION_MATCH = 88  # how closely an option's text must match part of the instruction, out of 100
BUY_BUTTON = '[button] Buy Now [button_]'
STOP_WORDS = set("""a about all also am an and any are as at be but buy by can could find for from get give good have help
i im in is it its ive like looking lower me my need needs of on one or order please price product purchase seeking
search searching shop some than that the them there these this to too want wanting with would you dollars""".split())
TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
SPELLINGS = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
             'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12', 'fifteen': '15', 'sixteen': '16', 'eighteen': '18',
             'twenty-four': '24', 'gray': 'grey', 'xl': 'x-large', 'xxl': 'xx-large', 'xxxl': '3x-large'}


def tokens(text):
    return re.findall(r'[a-z0-9.]*[a-z0-9]', text.lower().replace("'", ''))


def instruction_of(goal):
    """Returns the instruction's text without the budget, and the budget (None if it has none)."""
    goal = goal.lower().replace('instruction:', '').strip()
    match = re.search(r',? and price lower than ([\d,.]*\d)', goal)
    if not match:
        return goal, None
    return goal[:match.start()].strip(), float(match.group(1).replace(',', ''))


def price_range(text):
    """The first '$12.50' or '$10.00 to $20.00' in a text, as (low, high)."""
    match = re.search(r'\$([\d,]+\.?\d*)(?:\s+to\s+\$([\d,]+\.?\d*))?', text)
    if not match:
        return None
    low = float(match.group(1).replace(',', ''))
    return low, float(match.group(2).replace(',', '')) if match.group(2) else low


def within_budget(prices, budget):
    """The chance that a product is within budget; a price range is settled by a hidden draw."""
    if budget is None or prices is None:
        return 1.0
    low, high = prices
    if high <= budget:
        return 1.0
    if low >= budget:
        return 0.0
    return (budget - low) / (high - low)


def result_prices(ob):
    """The price of each product on a results page, in page order."""
    blocks = re.split(r'\[(?:clicked )?button\] [^\[\]]+ \[(?:clicked )?button_\]', ob)[1:]
    return [price_range(block) for block in blocks if '$' in block]


def page_body(ob):
    """A details page without the instruction and buttons that head every page."""
    return ob.split('[button] < Prev [button_]')[-1]


def item_actions(valid):
    return [a for a in valid if a.startswith('click[item - ')]


def is_item_page(valid):
    return 'click[buy now]' in valid


def is_search_page(valid):
    return bool(valid) and valid[0].startswith('search[')


def clicked_options(ob):
    return set(re.findall(r'\[clicked button\] ([^\[\]]+) \[clicked button_\]', ob.lower()))


def option_groups(ob):
    """A product's option buttons, read from the page; each group's name is the line above its buttons.

    The page is used, not the action list, because a long action list is cut down at random.
    """
    groups, current = [], None
    for line in ob.lower().split('\n'):
        match = re.fullmatch(r'\[(?:clicked )?button\] ([^\[\]]+) \[(?:clicked )?button_\]', line.strip())
        if not match or match.group(1) in NAVIGATION:
            current = None
            continue
        if current is None:
            current = []
            groups.append(current)
        current.append(match.group(1))
    return groups


def spelled_out(instruction):
    """The instruction with the spellings that option buttons rarely use replaced by the ones they do."""
    text = re.sub(r'\b(' + '|'.join(SPELLINGS) + r')\b', lambda m: SPELLINGS[m.group(1)], instruction.lower())
    return re.sub(r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ](\d)\b',
                  lambda m: str(int(TENS[m.group(1)]) + int(m.group(2))), text)


def option_match(option, instruction):
    """How well an option's text appears in the instruction, 0 to 100."""
    words, wanted = tokens(instruction), tokens(option)
    if not wanted:
        return 0
    if ''.join(wanted) in ''.join(words) and (set(wanted) <= set(words) or len(''.join(wanted)) >= 3):
        if set(wanted) <= set(words) or re.search(r'\b' + re.escape(''.join(wanted)) + r'\b', ' '.join(words)):
            return 100
    size = len(wanted)
    best = 0
    for n in {max(size - 1, 1), size, size + 1}:
        for i in range(max(len(words) - n + 1, 0)):
            best = max(best, fuzz.ratio(' '.join(wanted), ' '.join(words[i:i + n])))
    return best


def best_options(groups, instruction):
    """The closest option of each group to the instruction.

    One is chosen even when the match is poor: an unselected group can't earn anything and a wrong
    selection loses nothing, so a guess can only help.
    """
    plain = spelled_out(instruction)
    return [max((max(option_match(option, instruction), option_match(option, plain)), len(option), option)
                for option in group)[2] for group in groups]


def next_option_click(ob, instruction, skip=()):
    """The next option to click on a product page, or None when they are all selected.

    `skip` holds clicks already made on this visit, so an option the page fails to show as
    selected is not clicked again and again.
    """
    if BUY_BUTTON not in ob:  # only a product page has options; on a results page the buttons are products
        return None
    clicked = clicked_options(ob)
    groups = option_groups(ob)
    for group, option in zip(groups, best_options(groups, instruction)):
        action = f'click[{option}]'
        if option not in clicked and action not in skip:
            return action
    return None


def masked_valid(ob, valid, goal):
    """Hides the products that the page shows to be over budget, if any others remain."""
    actions = item_actions(valid)
    if not actions:
        return valid
    _, budget = instruction_of(goal)
    prices = result_prices(ob)
    if budget is None or len(prices) != len(actions):
        return valid
    over = {a for a, p in zip(actions, prices) if within_budget(p, budget) == 0.0}
    if len(over) == len(actions):
        return valid
    return [a for a in valid if a not in over]


def text_overlap(instruction, text):
    """The share of the instruction's content words that appear in a text."""
    wanted = {w for w in tokens(instruction) if w not in STOP_WORDS}
    if not wanted:
        return 0.0
    found = set(tokens(text))
    compact = ''.join(tokens(text))
    return sum(w in found or (len(w) > 3 and w in compact) for w in wanted) / len(wanted)


class CarefulShopper:
    """Searches, opens the most promising results, then buys the one that best fits the instruction.

    One decision is made per step from the page in view, and each env keeps its own notes, so
    this works with `run_episodes`. Requires `env_id` and `goal_idx` in each info.
    """

    def __init__(self, queries=('top1',), inspect=5, read_details=True, pages=1):
        self.queries, self.inspect, self.read_details, self.pages = queries, inspect, read_details, pages
        self.notes = {}

    def __call__(self, obs_list, infos):
        return [self.decide(ob, info) for ob, info in zip(obs_list, infos)]

    def fresh_notes(self, info, valid):
        searches = [valid[0] if q == 'top1' else valid[-1] for q in self.queries]
        return {'goal_idx': info['goal_idx'], 'searches': list(dict.fromkeys(searches)), 'search_at': 0, 'to_open': None,
                'seen': {}, 'best': None, 'opening': None, 'read': set(), 'text': '', 'buying': False, 'page': 1,
                'clicks': set()}

    def decide(self, ob, info):
        valid = info['valid']
        notes = self.notes.get(info['env_id'])
        if notes is None or notes['goal_idx'] != info['goal_idx']:
            notes = self.notes[info['env_id']] = self.fresh_notes(info, valid)
        instruction, budget = instruction_of(info['goal'])
        if is_search_page(valid):
            return self.on_search_page(notes, valid)
        if is_item_page(valid):
            return self.on_item_page(notes, ob, valid, instruction, budget)
        if 'click[< prev]' in valid and not item_actions(valid):
            # A description, features or reviews page: note what it says and go back.
            notes['text'] += ' ' + page_body(ob)
            return 'click[< prev]'
        return self.on_results_page(notes, ob, valid, instruction, budget)

    def on_search_page(self, notes, valid):
        if notes['buying']:
            return notes['best']['search']
        search = notes['searches'][min(notes['search_at'], len(notes['searches']) - 1)]
        notes['current_search'], notes['to_open'], notes['page'] = search, None, 1
        return search

    def on_results_page(self, notes, ob, valid, instruction, budget):
        if notes['buying']:
            if notes['best']['action'] in valid:
                return notes['best']['action']
            notes['buying'] = False  # the listing changed, so fall back to what is on this page
        actions = item_actions(valid)
        if notes['to_open'] is None:
            prices = result_prices(ob)
            prices = prices if len(prices) == len(actions) else [None] * len(actions)
            # Relevance first; only products certain to be over budget are left until last.
            ranked = sorted(zip(actions, prices), key=lambda c: (
                within_budget(c[1], budget) == 0.0, -text_overlap(instruction, c[0][13:-1]), -within_budget(c[1], budget)))
            notes['to_open'] = [a for a, _ in ranked if a not in notes['seen']][:self.inspect]
        if notes['to_open']:
            notes['opening'] = notes['to_open'].pop(0)
            notes['read'], notes['text'] = set(), ''
            return notes['opening']
        if notes['search_at'] + 1 < len(notes['searches']):
            notes['search_at'] += 1
            return 'click[back to search]'
        if notes['best'] is None:  # nothing to look at: try the instruction as the query, then give up
            return valid[0]
        notes['buying'] = True
        return notes['best']['action'] if notes['best']['action'] in valid else 'click[back to search]'

    def on_item_page(self, notes, ob, valid, instruction, budget):
        if notes['buying'] or notes['opening'] is None:
            action = next_option_click(ob, instruction, skip=notes['clicks']) or 'click[buy now]'
            notes['clicks'].add(action)
            return action
        if self.read_details:
            for page in ('features', 'description'):
                action = f'click[{page}]'
                if page not in notes['read'] and action in valid:
                    notes['read'].add(page)
                    return action
        notes['seen'][notes['opening']] = True
        fit = self.fit(ob, valid, instruction, budget, notes['text'], notes['opening'])
        if notes['best'] is None or fit > notes['best']['fit']:
            notes['best'] = {'fit': fit, 'action': notes['opening'], 'search': notes['current_search']}
        notes['opening'] = None
        return 'click[< prev]'

    @staticmethod
    def fit(ob, valid, instruction, budget, details, action):
        """How well a product seems to fit the instruction, mirroring the parts of the reward."""
        title = action[13:-1]
        groups = option_groups(ob)
        named = sum(max(option_match(option, instruction) for option in group) >= OPTION_MATCH for group in groups)
        return (text_overlap(instruction, title) * 2 + text_overlap(instruction, title + ' ' + details)
                + within_budget(price_range(ob), budget) * 2 + (named / len(groups) if groups else 0.0) * 2 + named * 0.25)
