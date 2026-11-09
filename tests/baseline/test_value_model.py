"""Tests of the value model and the shopper that uses it, on page text in the environment's own format."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baseline_models'))

from train_value_model import attribute_phrases, fit_linear  # noqa: E402
from value_model import (FULL_FEATURES, TITLE_FEATURES, ValueShopper, ValueWeights, Wanted, full_features,  # noqa: E402
                         goal_context, phrase_cover, rarity)

GOAL = 'Instruction: i need a red cotton shirt, and price lower than 30.00 dollars'
SEARCHES = ['search[red shirt]', 'search[i need a red cotton shirt, and price lower than 30.00 dollars]']
HEAD = 'Instruction:\ni need a red cotton shirt, and price lower than 30.00 dollars\n[button] Back to Search [button_]\n'


def results(*products):
    blocks = ''.join(f'[button] B{i} [button_]\n{title}\n${price}\n\n' for i, (title, price) in enumerate(products))
    return HEAD + 'Page 1 (Total results: 2)\n[button] Next > [button_]\n\n' + blocks


def result_valid(*products):
    return ['click[back to search]', 'click[next >]'] + [f'click[item - {title.lower()}]' for title, _ in products]


ITEM_VALID = ['click[back to search]', 'click[< prev]', 'click[red]', 'click[blue]', 'click[description]', 'click[features]',
              'click[reviews]', 'click[buy now]']


def item_page(title, clicked=False):
    red = '[clicked button] red [clicked button_]' if clicked else '[button] red [button_]'
    return (HEAD + '[button] < Prev [button_]\ncolor\n' + red + '\n[button] blue [button_]\n' + title
            + '\nPrice: $10.00\nRating: N.A.\n[button] Description [button_]\n[button] Features [button_]\n'
            + '[button] Reviews [button_]\n[button] Buy Now [button_]\n')


def identity_weights():
    """A model that values covering the instruction's words in the title and details."""
    def model(names):
        weights = np.zeros(len(names))
        weights[names.index('text_cover' if 'text_cover' in names else 'title_cover')] = 1.0
        return {'mean': np.zeros(len(names)).tolist(), 'scale': np.ones(len(names)).tolist(), 'weights': weights.tolist()}
    return ValueWeights({'title': model(TITLE_FEATURES), 'full': model(FULL_FEATURES)})


def test_words_common_among_the_titles_count_for_less():
    weights, _ = rarity(['red shirt', 'blue shirt', 'green shirt'])
    assert weights['shirt'] < weights['red']


def test_full_features_have_the_names_given_and_see_each_part_of_the_page():
    wanted = Wanted('i need a red cotton shirt', 30.0)
    context = goal_context(wanted, ['red shirt'])
    plain = full_features(wanted, 'red shirt', (10.0, 10.0), {0: 0}, context, '', [])
    detailed = full_features(wanted, 'red shirt', (10.0, 10.0), {0: 0}, context, 'soft cotton', [])
    named = full_features(wanted, 'red shirt', (10.0, 10.0), {0: 0}, context, '', [['red', 'blue']])
    assert len(plain) == len(FULL_FEATURES)
    text_cover = FULL_FEATURES.index('text_cover')
    assert plain[text_cover] < 1.0 and detailed[text_cover] == 1.0
    assert named[FULL_FEATURES.index('n_named')] == 1.0 and plain[FULL_FEATURES.index('n_named')] == 0.0


def test_attribute_phrases_are_looked_for_word_for_word():
    wanted = Wanted('i need a long lasting red shirt', 30.0, ['long lasting', 'gluten free'])
    assert wanted.phrases == ['long lasting']
    context = goal_context(wanted, ['red shirt'])
    named, in_title, in_text = (FULL_FEATURES.index(n) for n in ('phrases_named', 'phrases_in_title', 'phrases_in_text'))
    found = full_features(wanted, 'red shirt', None, {0: 0}, context, 'A LONG-LASTING dye', [])
    missing = full_features(wanted, 'red shirt', None, {0: 0}, context, 'lasting a long time', [])
    assert found[named] == 1.0 and found[in_title] == 0.0 and found[in_text] == 1.0
    assert missing[in_text] == 0.0


def test_attribute_phrases_ignore_filler_words_and_word_endings():
    wanted = Wanted('i want something easy to use with long sleeves', None, ['easy use', 'long sleeve', 'easy clean'])
    assert wanted.phrases == ['easy use', 'long sleeve']
    assert phrase_cover(['long sleeve', 'easy use'], 'A long-sleeved top that is easy to use') == 1.0
    assert phrase_cover(['easy use'], 'not easy, but simple to use') == 0.0


def test_attribute_phrases_come_from_several_goals_and_have_more_than_one_word():
    goals = [{'attributes': ['long lasting', 'soft']}] * 3 + [{'attributes': ['very rare phrase']}]
    assert attribute_phrases(goals) == ['long lasting']


def test_fit_learns_which_feature_pays_within_a_goal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    goals = np.repeat(np.arange(40), 10)
    y = 5 * X[:, 1] + goals * 100.0  # the size of a goal's rewards tells nothing about its products
    weights = np.array(fit_linear(X, y, goals, alpha=1.0)['weights'])
    assert abs(weights[1]) > 4 and abs(weights[0]) < 0.5 and abs(weights[2]) < 0.5


def drive(shopper, page, valid):
    info = dict(env_id=0, goal_idx=3, goal=GOAL, valid=valid)
    return shopper([page], [info])[0]


def test_shopper_reads_both_searches_inspects_then_buys_the_most_valuable():
    shopper = ValueShopper(identity_weights(), inspect=2, pages=1)
    coat, shirt = ('Blue Wool Coat', '20.00'), ('Red Cotton Shirt', '25.00')
    first = results(coat)
    second = results(shirt)

    assert drive(shopper, 'search', SEARCHES) == SEARCHES[0]
    assert drive(shopper, first, result_valid(coat)) == 'click[back to search]'
    assert drive(shopper, 'search', SEARCHES) == SEARCHES[1]
    # Both results pages are read. Inspection then starts with the first search's product, so it goes back.
    assert drive(shopper, second, result_valid(shirt)) == 'click[back to search]'
    assert drive(shopper, 'search', SEARCHES) == SEARCHES[0]
    assert drive(shopper, first, result_valid(coat)) == 'click[item - blue wool coat]'

    assert drive(shopper, item_page('Blue Wool Coat'), ITEM_VALID) == 'click[features]'
    details = ['click[back to search]', 'click[< prev]']
    assert drive(shopper, HEAD + '[button] < Prev [button_]\nwarm wool', details) == 'click[< prev]'
    assert drive(shopper, item_page('Blue Wool Coat'), ITEM_VALID) == 'click[description]'
    assert drive(shopper, HEAD + '[button] < Prev [button_]\na coat', details) == 'click[< prev]'
    assert drive(shopper, item_page('Blue Wool Coat'), ITEM_VALID) == 'click[< prev]'

    assert drive(shopper, first, result_valid(coat)) == 'click[back to search]'
    assert drive(shopper, 'search', SEARCHES) == SEARCHES[1]
    assert drive(shopper, second, result_valid(shirt)) == 'click[item - red cotton shirt]'
    drive(shopper, item_page('Red Cotton Shirt'), ITEM_VALID)
    drive(shopper, HEAD + '[button] < Prev [button_]\nsoft', details)
    drive(shopper, item_page('Red Cotton Shirt'), ITEM_VALID)
    drive(shopper, HEAD + '[button] < Prev [button_]\na shirt', details)
    assert drive(shopper, item_page('Red Cotton Shirt'), ITEM_VALID) == 'click[< prev]'

    # The shirt is valued higher and is on this page: select its option, then buy.
    assert drive(shopper, second, result_valid(shirt)) == 'click[item - red cotton shirt]'
    assert drive(shopper, item_page('Red Cotton Shirt'), ITEM_VALID) == 'click[red]'
    assert drive(shopper, item_page('Red Cotton Shirt', clicked=True), ITEM_VALID) == 'click[buy now]'


def test_details_are_not_polluted_by_the_instruction_at_the_top_of_the_page():
    shopper = ValueShopper(identity_weights(), inspect=1, pages=1)
    coat = ('Blue Wool Coat', '20.00')
    drive(shopper, 'search', SEARCHES[:1])
    drive(shopper, results(coat), result_valid(coat))
    drive(shopper, item_page('Blue Wool Coat'), ITEM_VALID)
    drive(shopper, HEAD + '[button] < Prev [button_]\nwarm wool', ['click[back to search]', 'click[< prev]'])
    trip = shopper.trips[0]
    assert 'wool' in trip.details and 'cotton' not in trip.details


def test_inspection_stops_in_time_to_buy_within_the_step_limit():
    shopper = ValueShopper(identity_weights(), inspect=50, pages=1, step_limit=30)
    products = [(f'Red Shirt {i}', '10.00') for i in range(10)]
    page, valid = results(*products), result_valid(*products)
    item = item_page('Red Shirt')
    details = ['click[back to search]', 'click[< prev]']
    bought, on, steps = False, 'search', 0
    while steps < 30 and not bought:
        steps += 1
        if on == 'search':
            action = drive(shopper, 'search', SEARCHES[:1])
        elif on == 'results':
            action = drive(shopper, page, valid)
        elif on == 'item':
            action = drive(shopper, item, ITEM_VALID)
        else:
            action = drive(shopper, HEAD + '[button] < Prev [button_]\ntext', details)
        if action == 'click[buy now]':
            bought = True
        elif action.startswith('search['):
            on = 'results'
        elif action.startswith('click[item - ') or (on == 'details' and action == 'click[< prev]'):
            on = 'item'
        elif action in ('click[features]', 'click[description]'):
            on = 'details'
        elif action == 'click[< prev]':
            on = 'results'
        elif action == 'click[back to search]':
            on = 'search'
    assert bought and steps < 30
