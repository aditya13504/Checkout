import argparse, json, logging, os, random, re, secrets
from pathlib import Path
from ast import literal_eval

from flask import (
    Flask,
    request,
    redirect,
    url_for
)

from rich import print

from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    convert_web_app_string_to_var,
    get_top_n_product_from_keywords,
    get_product_per_page,
    map_action_to_html,
    END_BUTTON
)
from web_agent_site.engine.goal import get_reward, get_goals
from web_agent_site.utils import (
    generate_mturk_code,
    setup_logger,
    close_logger,
    DEFAULT_FILE_PATH,
    DEBUG_PROD_SIZE,
)

app = Flask(__name__)

search_engine = None
all_products = None
product_item_dict = None
product_prices = None
attribute_to_asins = None
goals = None
weights = None

user_sessions = dict()
user_log_dir = None
SHOW_ATTRS_TAB = False

# A session pinned to goal n, for people: fixed_<n>, or fixed_<n>_<anything> for a fresh visit to the same goal.
FIXED_SESSION = re.compile(r'fixed_(\d+)(?:_[0-9A-Za-z]+)?')
NUM_TEST_GOALS = 500
DEMO_URL = os.environ.get('WEBSHOP_DEMO_URL', 'http://127.0.0.1:3100').rstrip('/')  # '/' when on the same address
# The agents' recorded test scores, shown next to a person's score on the same goal.
RESULTS_DIR = Path(os.environ.get('WEBSHOP_RESULTS_DIR',
                                  Path(__file__).resolve().parents[1] / 'kaggle_outputs' / 'outputs_full' / 'results'))
COMPARED_AGENTS = [
    ('rule_softmax', 'Rule baseline', 'Fixed rule'),
    ('il_seed0_greedy', 'IL', 'Copies people'),
    ('rl_scratch_greedy', 'RL from BERT', 'A2C only'),
    ('il_rl_greedy', 'IL+RL', 'Imitation, then A2C'),
    ('value_shopper', 'Value shopper', 'Learned value model'),
]
_recorded = None


def recorded_scores(goal_idx):
    """[{name, sub, score, steps}] of the compared agents on this test goal, if their results are present."""
    global _recorded
    if _recorded is None:
        _recorded = {}
        for key, *_ in COMPARED_AGENTS:
            path = RESULTS_DIR / f'{key}.json'
            if path.exists():
                with open(path, encoding='utf-8') as f:
                    _recorded[key] = {r['goal_idx']: (r['score'], r['steps']) for r in json.load(f)['results']}
    return [dict(key=key, name=name, sub=sub, score=_recorded[key][goal_idx][0], steps=_recorded[key][goal_idx][1])
            for key, name, sub in COMPARED_AGENTS if goal_idx in _recorded.get(key, {})]


def take_step(session_id):
    """Counts a person's actions the way the agents' are counted: one per search, click, option or page."""
    user_sessions[session_id]['steps'] = user_sessions[session_id].get('steps', 0) + 1


def page_context(session_id):
    """What every page for people shows: the request, its number and the steps taken so far."""
    session = user_sessions[session_id]
    goal_idx = session.get('goal_idx')
    return dict(human=True, steps=session.get('steps', 0), budget=session['goal'].get('price_upper'),
                request_no=goal_idx + 1 if goal_idx is not None and goal_idx < NUM_TEST_GOALS else None,
                catalogue_size=len(product_item_dict) if product_item_dict else None, show_attrs=SHOW_ATTRS_TAB)


@app.route('/')
def home():
    return redirect(url_for('new_session'))


@app.route('/new')
def new_session():
    """A fresh visit to a random test goal, so the agents' scores on it can be shown at the end."""
    goal_idx = secrets.randbelow(NUM_TEST_GOALS)
    return redirect(url_for('index', session_id=f'fixed_{goal_idx}_{secrets.token_hex(3)}'))

@app.route('/<session_id>', methods=['GET', 'POST'])
def index(session_id):
    global user_log_dir
    global all_products, product_item_dict, \
           product_prices, attribute_to_asins, \
           search_engine, \
           goals, weights, user_sessions

    if search_engine is None:
        random.seed(233)  # prices and goal price limits are sampled while loading
        all_products, product_item_dict, product_prices, attribute_to_asins = \
            load_products(
                filepath=DEFAULT_FILE_PATH,
                num_products=DEBUG_PROD_SIZE
            )
        search_engine = init_search_engine(num_products=DEBUG_PROD_SIZE)
        goals = get_goals(all_products, product_prices)
        random.seed(233)
        random.shuffle(goals)
        weights = [goal['weight'] for goal in goals]

    fixed_goal = FIXED_SESSION.fullmatch(session_id)
    if session_id not in user_sessions and fixed_goal and int(fixed_goal.group(1)) < len(goals):
        goal_dix = int(fixed_goal.group(1))
        goal = goals[goal_dix]
        instruction_text = goal['instruction_text']
        user_sessions[session_id] = {'goal': goal, 'done': False, 'goal_idx': goal_dix, 'steps': 0}
        if user_log_dir is not None:
            setup_logger(session_id, user_log_dir)
    elif session_id not in user_sessions:
        goal = random.choices(goals, weights)[0]
        instruction_text = goal['instruction_text']
        user_sessions[session_id] = {'goal': goal, 'done': False, 'goal_idx': None, 'steps': 0}
        if user_log_dir is not None:
            setup_logger(session_id, user_log_dir)
    else:
        instruction_text = user_sessions[session_id]['goal']['instruction_text']
        if not (request.method == 'POST' and 'search_query' in request.form):
            take_step(session_id)  # back to the search page

    if request.method == 'POST' and 'search_query' in request.form:
        keywords = request.form['search_query'].lower().split(' ')
        return redirect(url_for(
            'search_results',
            session_id=session_id,
            keywords=keywords,
            page=1,
        ))
    if user_log_dir is not None:
        logger = logging.getLogger(session_id)
        logger.info(json.dumps(dict(
            page='index',
            url=request.url,
            goal=user_sessions[session_id]['goal'],
        )))
    return map_action_to_html(
        'start',
        session_id=session_id,
        instruction_text=instruction_text,
        **page_context(session_id),
    )


@app.route(
    '/search_results/<session_id>/<keywords>/<page>',
    methods=['GET', 'POST']
)
def search_results(session_id, keywords, page):
    instruction_text = user_sessions[session_id]['goal']['instruction_text']
    page = convert_web_app_string_to_var('page', page)
    keywords = convert_web_app_string_to_var('keywords', keywords)
    top_n_products = get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins,
    )
    products = get_product_per_page(top_n_products, page)
    take_step(session_id)
    html = map_action_to_html(
        'search',
        session_id=session_id,
        products=products,
        keywords=keywords,
        page=page,
        total=len(top_n_products),
        instruction_text=instruction_text,
        **page_context(session_id),
    )
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(
        page='search_results',
        url=request.url,
        goal=user_sessions[session_id]['goal'],
        content=dict(
            keywords=keywords,
            search_result_asins=[p['asin'] for p in products],
            page=page,
        )
    )))
    return html


@app.route(
    '/item_page/<session_id>/<asin>/<keywords>/<page>/<options>',
    methods=['GET', 'POST']
)
def item_page(session_id, asin, keywords, page, options):
    options = literal_eval(options)
    product_info = product_item_dict[asin]

    goal_instruction = user_sessions[session_id]['goal']['instruction_text']
    product_info['goal_instruction'] = goal_instruction

    take_step(session_id)
    html = map_action_to_html(
        'click',
        session_id=session_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=goal_instruction,
        **page_context(session_id),
    )
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(
        page='item_page',
        url=request.url,
        goal=user_sessions[session_id]['goal'],
        content=dict(
            keywords=keywords,
            page=page,
            asin=asin,
            options=options,
        )
    )))
    return html


@app.route(
    '/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>',
    methods=['GET', 'POST']
)
def item_sub_page(session_id, asin, keywords, page, sub_page, options):
    options = literal_eval(options)
    product_info = product_item_dict[asin]

    goal_instruction = user_sessions[session_id]['goal']['instruction_text']
    product_info['goal_instruction'] = goal_instruction

    take_step(session_id)
    html = map_action_to_html(
        f'click[{sub_page}]',
        session_id=session_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=goal_instruction,
        **page_context(session_id),
    )
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(
        page='item_sub_page',
        url=request.url,
        goal=user_sessions[session_id]['goal'],
        content=dict(
            keywords=keywords,
            page=page,
            asin=asin,
            options=options,
        )
    )))
    return html


@app.route('/done/<session_id>/<asin>/<options>', methods=['GET', 'POST'])
def done(session_id, asin, options):
    options = literal_eval(options)
    goal = user_sessions[session_id]['goal']
    purchased_product = product_item_dict[asin]
    price = product_prices[asin]

    reward, reward_info = get_reward(
        purchased_product,
        goal,
        price=price,
        options=options,
        verbose=True
    )
    if not user_sessions[session_id]['done']:
        take_step(session_id)
    user_sessions[session_id]['done'] = True
    user_sessions[session_id]['reward'] = reward

    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(
        page='done',
        url=request.url,
        goal=goal,
        content=dict(
            asin=asin,
            options=options,
            price=price,
        ),
        reward=reward,
        reward_info=reward_info,
    )))
    close_logger(session_id)
    
    return map_action_to_html(
        f'click[{END_BUTTON}]',
        session_id=session_id,
        reward=reward,
        asin=asin,
        options=options,
        reward_info=reward_info,
        query=purchased_product['query'],
        category=purchased_product['category'],
        product_category=purchased_product['product_category'],
        goal_attrs=user_sessions[session_id]['goal']['attributes'],
        purchased_attrs=purchased_product['Attributes'],
        goal=goal,
        mturk_code=generate_mturk_code(session_id),
        **done_context(session_id, goal, purchased_product, price, options, reward, reward_info),
    )


def done_context(session_id, goal, product, price, options, reward, info):
    """The breakdown of a person's score, and the agents' scores on the same goal."""
    goal_idx = user_sessions[session_id].get('goal_idx')
    test_goal = goal_idx is not None and goal_idx < NUM_TEST_GOALS
    goal_options = list(goal['goal_options'].values() if isinstance(goal['goal_options'], dict) else goal['goal_options'])
    n_att, n_opt = len(goal['attributes']), len(goal_options)
    fresh = f'fixed_{goal_idx}_{secrets.token_hex(3)}' if goal_idx is not None else None
    return dict(
        **page_context(session_id),
        instruction_text=goal['instruction_text'],
        score=round(reward * 100, 1), product=product, price_paid=price,
        goal_attributes=goal['attributes'], goal_options=goal_options,
        att_hits=round((info.get('r_att') or 0) * n_att), opt_hits=round(info.get('r_option', 0) * n_opt),
        r_type=info.get('r_type', 0), r_price=info.get('r_price'),
        agents=recorded_scores(goal_idx) if test_goal else [],
        retry_url=url_for('index', session_id=fresh) if fresh else url_for('new_session'),
        demo_url=f'{DEMO_URL}/?goal={goal_idx + 1}' if test_goal else DEMO_URL,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebShop flask app backend configuration")
    parser.add_argument("--log", action='store_true', help="Log actions on WebShop in trajectory file")
    parser.add_argument("--attrs", action='store_true', help="Show attributes tab in item page")
    parser.add_argument("--host", default='127.0.0.1', help="Address to serve on (use 0.0.0.0 to expose it)")

    args = parser.parse_args()
    if args.log:
        user_log_dir = Path('user_session_logs/mturk')
        user_log_dir.mkdir(parents=True, exist_ok=True)
    SHOW_ATTRS_TAB = args.attrs

    app.run(host=args.host, port=3000)
