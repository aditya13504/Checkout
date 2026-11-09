"""
Integration tests of the text environment on a small generated catalogue. They use the
real engine, templates, reward and SQLite search.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'search_engine'))

from prepare_catalog import prepare  # noqa: E402
import web_agent_site.engine.engine as engine  # noqa: E402
from web_agent_site.engine.engine import get_top_n_product_from_keywords, parse_action  # noqa: E402
from web_agent_site.engine.search import SqliteSearcher, build_match_query  # noqa: E402
from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv  # noqa: E402


def make_catalogue(directory, n=60):
    items, attrs, human = [], {}, {}
    for i in range(n):
        asin = f'B{i:09d}'
        options = {'color': [{'value': 'Navy Blue'}, {'value': 'Red'}], 'size': [{'value': 'Small'}]}
        if i == 0:
            options = {'size': [{'value': 'Large'}, {'value': '6 [wide]'}, {'value': '12"'}],
                       'flavor': [{'value': 'Café'}]}
        items.append(dict(
            asin=asin, name=f'Running Shoes Model {i}', full_description='comfortable shoes with rubber sole',
            small_description=['rubber sole', 'breathable'], pricing='$15.00' if i % 2 else '$10.00$40.00',
            customization_options=options, images=[f'http://x/{i}.jpg'], category='fashion',
            query='running shoes', product_category='Clothing › Shoes › Running'))
        attrs[asin] = {'attributes': ['rubber sole']}
        human[asin] = [dict(instruction=f'i want running shoes number {i}.', instruction_attributes=['rubber sole'],
                            instruction_options=['red'])]
    (directory / 'raw_items.json').write_text(json.dumps(items), encoding='utf-8')
    (directory / 'raw_attrs.json').write_text(json.dumps(attrs), encoding='utf-8')
    (directory / 'human.json').write_text(json.dumps(human), encoding='utf-8')


@pytest.fixture(scope='module')
def catalogue(tmp_path_factory):
    directory = tmp_path_factory.mktemp('catalogue')
    make_catalogue(directory)
    prepare(directory / 'raw_items.json', directory / 'raw_attrs.json', directory / 'human.json', directory / 'out')
    return directory


@pytest.fixture()
def server(catalogue, monkeypatch):
    out = catalogue / 'out'
    monkeypatch.setattr(engine, 'DEFAULT_ATTR_PATH', str(out / 'attrs.json'))
    monkeypatch.setattr(engine, 'HUMAN_ATTR_PATH', str(catalogue / 'human.json'))
    monkeypatch.setattr(engine, 'SQLITE_INDEX_PATH', str(out / 'webshop.sqlite'))
    return SimServer('http://x', str(out / 'items.json'), human_goals=1)


@pytest.fixture()
def env(server):
    return WebAgentTextEnv(observation_mode='text_rich', server=server)


def open_product(env, name, asin):
    env.step(f'search[{name}]')
    env.step(f'click[{asin}]')
    return env.server.user_sessions[env.session]


@pytest.mark.parametrize('action,expected', [
    ('search[red shoes]', ('search', 'red shoes')),
    ('click[6 [wide]]', ('click', '6 [wide]')),
    ('search[shoes [mens]]', ('search', 'shoes [mens]')),
    ('click[buy now]x', ('click[buy now]x', None)),
    ('start', ('start', None)),
])
def test_parse_action(action, expected):
    assert parse_action(action) == expected


def test_search_ranks_matches_and_ignores_stop_words(catalogue):
    assert build_match_query('the Red and red shoes!') == '"red" OR "shoes"'
    searcher = SqliteSearcher(str(catalogue / 'out' / 'webshop.sqlite'))
    assert searcher.search('model 7', k=1)[0].docid == 'B000000007'
    assert searcher.search('the and of', k=5) == []
    assert searcher.search('running', k=50)[0].docid.startswith('B')
    assert len(searcher.search('running', k=50)) == 50


def test_prepared_catalogue_keeps_what_the_environment_needs(server):
    product = server.product_item_dict['B000000000']
    assert product['options'] == {'size': ['large', '6 [wide]', '12"'], 'flavor': ['caf\u00e9']}
    assert product['Price'] == '$10.0 to $40.0'
    assert product['MainImage'] == 'http://x/0.jpg'


def test_catalogue_size_limit_keeps_every_goal_product(tmp_path):
    make_catalogue(tmp_path, n=60)
    human = json.loads((tmp_path / 'human.json').read_text(encoding='utf-8'))
    kept_for_goals = dict(list(human.items())[-5:])    # goal products that come last in the file
    (tmp_path / 'human.json').write_text(json.dumps(kept_for_goals), encoding='utf-8')
    summary = prepare(tmp_path / 'raw_items.json', tmp_path / 'raw_attrs.json', tmp_path / 'human.json',
                      tmp_path / 'out', max_products=10)
    assert summary['products'] == 10 and summary['human_goal_products'] == 5
    asins = [p['asin'] for p in json.loads((tmp_path / 'out' / 'items.json').read_text(encoding='utf-8'))]
    assert asins == sorted(asins)                      # the catalogue order is preserved


def test_stemming_matches_lucene_behaviour(catalogue):
    searcher = SqliteSearcher(str(catalogue / 'out' / 'webshop.sqlite'))
    assert len(searcher.search('shoe', k=5)) == 5


@pytest.mark.parametrize('action', ['search[<a> rubber sole]', 'search[<c>]', 'search[<c> fashion]',
                                    'search[<q> running shoes]', 'search[<r>]', 'search[<a>]'])
def test_special_search_tokens_do_not_crash(env, action):
    _, reward, done, _ = env.step(action)
    assert (reward, done) == (0.0, False)


def test_pagination_stops_at_the_last_page(env):
    env.step('search[running shoes]')
    for expected_page in range(2, 6):
        assert 'next >' in env.get_available_actions()['clickables']
        env.step('click[next >]')
        assert env.server.user_sessions[env.session]['page'] == expected_page
    assert 'next >' not in env.get_available_actions()['clickables']


def test_options_with_brackets_can_be_selected(env):
    session = open_product(env, 'model 0', 'b000000000')
    env.step('click[6 [wide]]')
    assert session['options'] == {'size': '6 [wide]'}


def test_non_ascii_selected_option_is_shown_as_clicked(env):
    open_product(env, 'model 0', 'b000000000')
    observation, *_ = env.step('click[café]')
    assert 'You have clicked café' in observation


def test_goals_and_prices_are_reproducible(catalogue, server, monkeypatch):
    first = [(g['asin'], g['price_upper'], g['instruction_text']) for g in server.goals]
    prices = dict(server.product_prices)
    out = catalogue / 'out'
    again = SimServer('http://x', str(out / 'items.json'), human_goals=1)
    assert [(g['asin'], g['price_upper'], g['instruction_text']) for g in again.goals] == first
    assert again.product_prices == prices


def test_buying_the_goal_product_with_its_options_is_rewarded(env, server):
    goal = server.goals[3]
    env.reset(session=3)
    assert env.server.user_sessions[env.session]['goal'] is goal
    number = int(goal['asin'][1:])
    env.step(f'search[model {number}]')
    env.step(f"click[{goal['asin'].lower()}]")
    env.step('click[red]')
    _, reward, done, _ = env.step('click[buy now]')
    assert done and reward == 1.0


def test_buying_without_options_loses_the_option_share(env, server):
    goal = server.goals[3]
    env.reset(session=3)
    env.step(f"search[model {int(goal['asin'][1:])}]")
    env.step(f"click[{goal['asin'].lower()}]")
    _, reward, done, _ = env.step('click[buy now]')
    assert done and reward == pytest.approx(2 / 3)


def test_invalid_actions_change_nothing(env):
    before = env.observation
    assert env.step('click[does not exist]')[1:3] == (0, False)
    assert env.step('nonsense')[1:3] == (0, False)
    assert env.observation == before


def test_site_pages_render_through_flask(catalogue, monkeypatch):
    import web_agent_site.app as site
    out = catalogue / 'out'
    monkeypatch.setattr(engine, 'DEFAULT_ATTR_PATH', str(out / 'attrs.json'))
    monkeypatch.setattr(engine, 'HUMAN_ATTR_PATH', str(catalogue / 'human.json'))
    monkeypatch.setattr(engine, 'SQLITE_INDEX_PATH', str(out / 'webshop.sqlite'))
    monkeypatch.setattr(site, 'DEFAULT_FILE_PATH', str(out / 'items.json'))
    monkeypatch.setattr(site, 'search_engine', None)
    client = site.app.test_client()
    assert client.get('/s1').status_code == 200
    assert client.get('/fixed').status_code == 200     # used to be a server error
    assert client.get('/fixed_99999').status_code == 200
    follow = client.post('/s1', data={'search_query': 'running shoes'}).headers['Location']
    assert client.get(follow).status_code == 200

    # People get their own pages; options from the URL only ever appear as escaped text there.
    page = client.get("/item_page/s1/B000000000/['a']/1/{'size':'`+alert(1)+`<b>'}").data.decode()
    assert '<script' not in page and '`+alert(1)+`&lt;b&gt;' in page

    # The agents' product page keeps its options in a script, as a plain, quoted JSON string.
    with site.app.test_request_context():
        agent_page = engine.map_action_to_html(
            'click', session_id='s1', product_info=site.product_item_dict['B000000000'], keywords=['a'], page=1,
            asin='B000000000', options={'size': '`+alert(1)+`'}, instruction_text='x', show_attrs=False)
    assert not re.search(r'JSON\.parse\(`', agent_page)  # no JavaScript template literal to break out of
    assert '"`+alert(1)+`"' in agent_page


def test_a_person_can_shop_and_see_the_score(catalogue, monkeypatch):
    import web_agent_site.app as site
    out = catalogue / 'out'
    monkeypatch.setattr(engine, 'DEFAULT_ATTR_PATH', str(out / 'attrs.json'))
    monkeypatch.setattr(engine, 'HUMAN_ATTR_PATH', str(catalogue / 'human.json'))
    monkeypatch.setattr(engine, 'SQLITE_INDEX_PATH', str(out / 'webshop.sqlite'))
    monkeypatch.setattr(site, 'DEFAULT_FILE_PATH', str(out / 'items.json'))
    monkeypatch.setattr(site, 'search_engine', None)
    client = site.app.test_client()

    assert client.get('/new').headers['Location'].split('/')[-1].startswith('fixed_')
    start = client.get('/fixed_3_try1').data.decode()
    assert 'Your task' in start and 'Request 4 of 500' in start and 'Step <b>0</b>' in start

    results = client.get(client.post('/fixed_3_try1', data={'search_query': 'running shoes'}).headers['Location'])
    assert 'Running Shoes Model' in results.data.decode()
    item = client.get("/item_page/fixed_3_try1/B000000001/['running','shoes']/1/{}").data.decode()
    assert 'Buy now' in item and 'Not chosen' in item
    tab = client.get("/item_sub_page/fixed_3_try1/B000000001/['running','shoes']/1/Features/{'color':'Red'}").data.decode()
    assert 'breathable' in tab and 'class="on" aria-current="page">Features' in tab

    done = client.post("/done/fixed_3_try1/B000000001/{'color':'Red'}").data.decode()
    assert 'How the store scored it' in done and 'Bought at step 4' in done
    assert '1 of 1 matched' in done                      # the requested option, red, was chosen

    # A second visit to the same goal starts afresh.
    assert 'Step <b>0</b>' in client.get('/fixed_3_try2').data.decode()
