"""Tests of the page-reading rules, on page text in the environment's own format."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baseline_models'))

from env import WebEnv  # noqa: E402
from shopping_rules import (CarefulShopper, best_options, instruction_of, masked_valid, next_option_click,  # noqa: E402
                            option_groups, option_match, page_body, price_range, result_prices, spelled_out, text_overlap,
                            within_budget)

GOAL = 'Instruction: i need a red cotton shirt in large size, and price lower than 30.00 dollars'
RESULTS = """Instruction:
i need a red cotton shirt in large size, and price lower than 30.00 dollars
[button] Back to Search [button_]
Page 1 (Total results: 50)
[button] Next > [button_]

[button] B001 [button_]
Blue Wool Coat
$99.00

[button] B002 [button_]
Red Cotton Shirt
$10.00 to $40.00

[button] B003 [button_]
Green Shirt
$5.50
"""
RESULT_VALID = ['click[back to search]', 'click[next >]', 'click[item - blue wool coat]',
                'click[item - red cotton shirt]', 'click[item - green shirt]']
ITEM = """Instruction:
i need a red cotton shirt in large size, and price lower than 30.00 dollars
[button] Back to Search [button_]
[button] < Prev [button_]
color
[clicked button] blue [clicked button_]
[button] red [button_]
size
[button] x-large [button_]
[button] large [button_]
Red Cotton Shirt
Price: $10.00 to $40.00
Rating: N.A.
[button] Description [button_]
[button] Features [button_]
[button] Reviews [button_]
[button] Buy Now [button_]
"""
ITEM_VALID = ['click[back to search]', 'click[< prev]', 'click[blue]', 'click[red]', 'click[x-large]', 'click[large]',
              'click[description]', 'click[features]', 'click[reviews]', 'click[buy now]']


def test_budget_and_prices_come_from_the_text():
    assert instruction_of(GOAL) == ('i need a red cotton shirt in large size', 30.0)
    assert instruction_of('Instruction: i need a shirt')[1] is None
    assert price_range('Price: $10.00 to $1,040.50') == (10.0, 1040.5)
    assert result_prices(RESULTS) == [(99.0, 99.0), (10.0, 40.0), (5.5, 5.5)]
    assert within_budget((5.5, 5.5), 30) == 1.0 and within_budget((99.0, 99.0), 30) == 0.0
    assert within_budget((10.0, 40.0), 30) == 2 / 3


def test_mask_hides_only_products_certain_to_be_over_budget():
    masked = masked_valid(RESULTS, RESULT_VALID, GOAL)
    assert 'click[item - blue wool coat]' not in masked
    assert 'click[item - red cotton shirt]' in masked and 'click[next >]' in masked


def test_mask_keeps_everything_when_nothing_is_affordable():
    assert masked_valid(RESULTS, RESULT_VALID, 'i need a shirt, and price lower than 1.00 dollars') == RESULT_VALID


def test_option_groups_and_the_longest_right_match_wins():
    assert option_groups(ITEM) == [['blue', 'red'], ['x-large', 'large']]
    assert option_match('large', 'a shirt in large size') == 100
    assert option_match('x-large', 'a shirt in large size') < 100
    assert option_match('pack of 12', 'buy a pack of 12 bars') == 100
    assert option_match('s', 'a shirt in large size') < 88


def test_named_options_are_clicked_once_then_nothing_is_left():
    instruction, _ = instruction_of(GOAL)
    assert next_option_click(ITEM, instruction) == 'click[red]'
    selected = ITEM.replace('[button] red [button_]', '[clicked button] red [clicked button_]').replace(
        '[button] large [button_]', '[clicked button] large [clicked button_]')
    assert next_option_click(selected, instruction) is None


def test_every_option_group_gets_a_guess_because_a_wrong_one_costs_nothing():
    assert best_options([['blue', 'red'], ['small', 'medium']], 'a red shirt') == ['red', 'medium']


def test_common_spellings_find_their_option():
    assert best_options([['s', 'xx-large', 'large']], 'a tunic in size xxl') == ['xx-large']
    assert best_options([['grey', 'blue']], 'a gray chair') == ['grey']
    assert best_options([['3 pack', '1 pack']], 'just one pack of picks') == ['1 pack']
    assert spelled_out('a fifty-two inch width') == 'a 52 inch width'


def test_options_hidden_from_a_cut_down_action_list_are_still_found():
    assert next_option_click(ITEM, instruction_of(GOAL)[0]) == 'click[red]'  # reads the page, needs no action list


def test_no_option_is_clicked_on_a_results_page_whose_buttons_are_products():
    assert next_option_click(RESULTS, instruction_of(GOAL)[0]) is None


def test_assist_mode_leaves_a_results_page_alone():
    env = assisted_env()
    assert env.select_named_options(RESULTS) == RESULTS and not env.env.clicks


def test_a_click_the_page_does_not_show_is_not_repeated():
    instruction = 'a red pillow cover in 20"x20"'
    page = ITEM.replace('[button] large [button_]', '[button] 20"x20" [button_]')
    assert next_option_click(page, instruction) == 'click[red]'
    assert next_option_click(page, instruction, skip={'click[red]'}) == 'click[20"x20"]'
    assert next_option_click(page, instruction, skip={'click[red]', 'click[20"x20"]'}) is None


def test_assist_mode_stops_when_the_page_never_shows_a_click_as_made():
    env = assisted_env()
    env.env.step = lambda action: (env.env.clicks.append(action) or (ITEM, 0.0, False, None))  # the page never changes
    env.select_named_options(ITEM)
    assert env.env.clicks == ['click[red]', 'click[large]'] and env.steps == 2


def test_details_text_leaves_out_the_instruction_at_the_top_of_every_page():
    page = 'Instruction:\na red shirt\n[button] Back to Search [button_]\n[button] < Prev [button_]\nmade of cotton'
    assert page_body(page).strip() == 'made of cotton'


class FakeSite:
    """Stands in for the text environment: a product page whose options can be clicked."""
    instruction_text = GOAL

    def __init__(self):
        self.page, self.clicks = ITEM, []

    def step(self, action):
        self.clicks.append(action)
        option = action[6:-1]
        self.page = self.page.replace(f'[button] {option} [button_]', f'[clicked button] {option} [clicked button_]')
        return self.page, 0.0, False, None

    def get_available_actions(self):
        return {'has_search_bar': False, 'clickables': [a[6:-1] for a in ITEM_VALID]}


def assisted_env(assist=True, step_limit=100):
    env = WebEnv.__new__(WebEnv)
    env.env, env.assist, env.steps, env.step_limit = FakeSite(), assist, 0, step_limit
    env.ban_buy, env.click_item_name, env.asin2name = 0, 0, {}
    return env


def test_assist_mode_selects_named_options_inside_the_environment():
    env = assisted_env()
    ob = env.select_named_options(ITEM)
    assert env.env.clicks == ['click[red]', 'click[large]'] and env.steps == 2
    assert '[clicked button] red [clicked button_]' in ob and '[clicked button] large [clicked button_]' in ob
    assert env.select_named_options(ob) == ob and env.steps == 2


def test_assist_mode_respects_the_step_limit_and_can_be_off():
    env = assisted_env(step_limit=2)
    env.select_named_options(ITEM)
    assert env.steps == 1
    off = assisted_env(assist=False)
    assert off.select_named_options(ITEM) == ITEM and not off.env.clicks


def test_assist_mode_hides_unaffordable_products_from_the_action_set():
    assert 'click[item - blue wool coat]' not in assisted_env().present_valid(RESULTS, RESULT_VALID)
    assert 'click[item - blue wool coat]' in assisted_env(assist=False).present_valid(RESULTS, RESULT_VALID)


def test_text_overlap_ignores_filler_words():
    assert text_overlap('i need a red shirt', 'Red Shirt') == 1.0
    assert text_overlap('i need a red shirt', 'Blue Coat') == 0.0


def test_careful_shopper_reads_a_details_page_and_goes_back():
    shopper = CarefulShopper(inspect=1, read_details=True)
    info = dict(env_id=0, goal_idx=7, goal=GOAL)
    shopper(['search page'], [{**info, 'valid': ['search[red cotton shirt]']}])
    shopper([RESULTS], [{**info, 'valid': RESULT_VALID}])
    assert shopper([ITEM], [{**info, 'valid': ITEM_VALID}]) == ['click[features]']
    details = {**info, 'valid': ['click[back to search]', 'click[< prev]']}
    assert shopper(['made of soft cotton'], [details]) == ['click[< prev]']
    assert shopper([ITEM], [{**info, 'valid': ITEM_VALID}]) == ['click[description]']
    assert shopper(['a red shirt'], [details]) == ['click[< prev]']
    assert shopper([ITEM], [{**info, 'valid': ITEM_VALID}]) == ['click[< prev]']  # read both, so go back to compare


def test_careful_shopper_compares_products_then_buys_the_best():
    shopper = CarefulShopper(inspect=2, read_details=False)
    info = dict(env_id=0, goal_idx=7, goal=GOAL)
    search = ['search[red cotton shirt]', 'search[i need a red cotton shirt]']
    assert shopper(['search page'], [{**info, 'valid': search}]) == ['search[red cotton shirt]']
    first = shopper([RESULTS], [{**info, 'valid': RESULT_VALID}])[0]
    assert first == 'click[item - red cotton shirt]'  # the coat is over budget and fits worst

    assert shopper([ITEM], [{**info, 'valid': ITEM_VALID}]) == ['click[< prev]']
    second = shopper([RESULTS], [{**info, 'valid': RESULT_VALID}])[0]
    assert second == 'click[item - green shirt]'
    plain = ITEM.split('color')[0] + 'Green Shirt\nPrice: $5.50\nRating: N.A.\n[button] Buy Now [button_]\n'
    assert shopper([plain], [{**info, 'valid': ['click[< prev]', 'click[buy now]']}]) == ['click[< prev]']

    assert shopper([RESULTS], [{**info, 'valid': RESULT_VALID}]) == ['click[item - red cotton shirt]']
    assert shopper([ITEM], [{**info, 'valid': ITEM_VALID}]) == ['click[red]']
    selected = ITEM.replace('[button] red [button_]', '[clicked button] red [clicked button_]').replace(
        '[button] large [button_]', '[clicked button] large [clicked button_]')
    assert shopper([selected], [{**info, 'valid': ITEM_VALID}]) == ['click[buy now]']
