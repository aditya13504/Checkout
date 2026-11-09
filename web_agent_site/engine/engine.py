"""
"""
import os
import re
import json
import random
from collections import defaultdict
from ast import literal_eval
from decimal import Decimal

from tqdm import tqdm
from flask import current_app
from rich import print

from web_agent_site.engine.product_store import load_products_from_store
from web_agent_site.engine.search import SqliteSearcher
from web_agent_site.utils import (
    BASE_DIR,
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH,
    PRODUCT_STORE_PATH,
    SQLITE_INDEX_PATH,
)

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
_compiled_templates = {}

SEARCH_RETURN_N = 50
PRODUCT_WINDOW = 10
TOP_K_ATTR = 10

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

# `name[argument]`. The argument may itself contain brackets, e.g. click[6 [wide]].
ACTION_PATTERN = re.compile(r'^([^\[\]]+)\[(.+)\]$')

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if kwargs.pop('human', False):
        return render_human_page(action_name, action_arg, kwargs)
    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template(
            path,
            session_id=kwargs['session_id'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template(
            path,
            session_id=kwargs['session_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template(
            path,
            session_id=kwargs['session_id'],
            reward=kwargs['reward'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            reward_info=kwargs.get('reward_info'),
            goal_attrs=kwargs.get('goal_attrs'),
            purchased_attrs=kwargs.get('purchased_attrs'),
            goal=kwargs.get('goal'),
            mturk_code=kwargs.get('mturk_code'),
            query=kwargs.get('query'),
            category=kwargs.get('category'),
            product_category=kwargs.get('product_category'),
        )
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template(
            path,
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text')
        )
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template(
            path,
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs']
        )
    else:
        raise ValueError('Action name not recognized.')
    return html


# Pages for people using the site in a browser. Agents never see these: they get the templates above,
# whose visible text, buttons and links make up their observations and actions, so those stay as they are.
HUMAN_TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates_human')
HUMAN_PAGES = {'start': 'search.html', 'search': 'results.html', 'done': 'done.html', 'item': 'item.html',
               'sub_page': 'sub_page.html'}
_human_env = None


def render_human_page(action_name, action_arg, context):
    global _human_env
    if _human_env is None:
        from flask import url_for
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        _human_env = Environment(loader=FileSystemLoader(HUMAN_TEMPLATE_DIR), autoescape=select_autoescape(['html']),
                                 trim_blocks=True, lstrip_blocks=True)
        _human_env.globals['url_for'] = url_for
    if action_name == 'click' and action_arg == END_BUTTON:
        page = 'done'
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        page, context = 'sub_page', {**context, 'sub_page': action_arg}
    elif action_name == 'click':
        page = 'item'
    elif action_name in ('start', 'search'):
        page = action_name
    else:
        raise ValueError('Action name not recognized.')
    return _human_env.get_template(HUMAN_PAGES[page]).render(**context)


def read_html_template(path):
    with open(path, encoding='utf-8') as f:
        template = f.read()
    return template


def render_template(path, **context):
    """Renders a template file, compiling it only the first time it is used."""
    app = current_app._get_current_object()
    key = (id(app), path)
    template = _compiled_templates.get(key)
    if template is None:
        template = app.jinja_env.from_string(read_html_template(path))
        _compiled_templates[key] = template
    return template.render(**context)


def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    m = ACTION_PATTERN.match(action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        keywords = string
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var


def get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins=None,
    ):
    if keywords[0] == '<r>':
        k = min(SEARCH_RETURN_N, len(all_products))
        top_n_products = random.sample(all_products, k=k)
    elif keywords[0] == '<a>' and attribute_to_asins is not None:
        attribute = ' '.join(keywords[1:]).strip()
        asins = attribute_to_asins.get(attribute, set())
        top_n_products = [p for p in all_products if p['asin'] in asins]
    elif keywords[0] == '<c>' and len(keywords) > 1:
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p['category'] == category]
    elif keywords[0] == '<q>':
        query = ' '.join(keywords[1:]).strip()
        top_n_products = [p for p in all_products if p['query'] == query]
    else:
        keywords = ' '.join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        top_n_asins = [hit.docid for hit in hits]
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    return top_n_products


def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


def init_search_engine(num_products=None):
    """Uses the SQLite index if it has been built, and otherwise a Lucene index."""
    if os.path.exists(SQLITE_INDEX_PATH):
        return SqliteSearcher(SQLITE_INDEX_PATH)

    if num_products == 100:
        indexes = 'indexes_100'
    elif num_products == 1000:
        indexes = 'indexes_1k'
    elif num_products == 100000:
        indexes = 'indexes_100k'
    elif num_products is None:
        indexes = 'indexes'
    else:
        raise NotImplementedError(f'num_products being {num_products} is not supported yet.')
    from pyserini.search.lucene import LuceneSearcher  # needs Java, so only imported here
    search_engine = LuceneSearcher(os.path.join(BASE_DIR, f'../search_engine/{indexes}'))
    return search_engine


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products


def finish_product(p, attributes, human_attributes, human_goals, all_reviews=None, all_ratings=None):
    """Adds the fields the site and the reward use to a product as read from the catalogue."""
    asin = p['asin']
    all_reviews = all_reviews or {}
    all_ratings = all_ratings or {}
    p['category'] = p['category']
    p['query'] = p['query']
    p['product_category'] = p['product_category']

    p['Title'] = p['name']
    p['Description'] = p['full_description']
    p['Reviews'] = all_reviews.get(asin, [])
    p['Rating'] = all_ratings.get(asin, 'N.A.')
    for r in p['Reviews']:
        if 'score' not in r:
            r['score'] = r.pop('stars')
        if 'review' not in r:
            r['body'] = ''
        else:
            r['body'] = r.pop('review')
    p['BulletPoints'] = p['small_description'] \
        if isinstance(p['small_description'], list) else [p['small_description']]

    pricing = p.get('pricing')
    if pricing is None or not pricing:
        pricing = [100.0]
        price_tag = '$100.0'
    else:
        pricing = [
            float(Decimal(re.sub(r'[^\d.]', '', price)))
            for price in pricing.split('$')[1:]
        ]
        if len(pricing) == 1:
            price_tag = f"${pricing[0]}"
        else:
            price_tag = f"${pricing[0]} to ${pricing[1]}"
            pricing = pricing[:2]
    p['pricing'] = pricing
    p['Price'] = price_tag

    options = dict()
    customization_options = p['customization_options']
    option_to_image = dict()
    if customization_options:
        for option_name, option_contents in customization_options.items():
            if option_contents is None:
                continue
            option_name = option_name.lower()

            option_values = []
            for option_content in option_contents:
                option_value = option_content['value'].strip().replace('/', ' | ').lower()
                option_image = option_content.get('image', None)

                option_values.append(option_value)
                option_to_image[option_value] = option_image
            options[option_name] = option_values
    p['options'] = options
    p['option_to_image'] = option_to_image

    # without color, size, price, availability
    # if asin in attributes and 'attributes' in attributes[asin]:
    #     p['Attributes'] = attributes[asin]['attributes']
    # else:
    #     p['Attributes'] = ['DUMMY_ATTR']
    # p['instruction_text'] = \
    #     attributes[asin].get('instruction', None)
    # p['instruction_attributes'] = \
    #     attributes[asin].get('instruction_attributes', None)

    # without color, size, price, availability
    if asin in attributes and 'attributes' in attributes[asin]:
        p['Attributes'] = attributes[asin]['attributes']
    else:
        p['Attributes'] = ['DUMMY_ATTR']
        
    if human_goals:
        if asin in human_attributes:
            p['instructions'] = human_attributes[asin]
    else:
        p['instruction_text'] = \
            attributes.get(asin, {}).get('instruction', None)

        p['instruction_attributes'] = \
            attributes.get(asin, {}).get('instruction_attributes', None)

    p['MainImage'] = (p.get('images') or [''])[0]
    p['query'] = p['query'].lower().strip()
    return p


def load_products(filepath, num_products=None, human_goals=True):
    if PRODUCT_STORE_PATH and os.path.exists(PRODUCT_STORE_PATH):
        return load_products_from_store(PRODUCT_STORE_PATH, generate_product_prices)
    # TODO: move to preprocessing step -> enforce single source of truth
    with open(filepath, encoding='utf-8') as f:
        products = json.load(f)
    print('Products loaded.')
    products = clean_product_keys(products)
    
    # with open(DEFAULT_REVIEW_PATH) as f:
    #     reviews = json.load(f)
    all_reviews = dict()
    all_ratings = dict()
    # for r in reviews:
    #     all_reviews[r['asin']] = r['reviews']
    #     all_ratings[r['asin']] = r['average_rating']

    human_attributes = {}
    if human_goals:
        with open(HUMAN_ATTR_PATH, encoding='utf-8') as f:
            human_attributes = json.load(f)
    with open(DEFAULT_ATTR_PATH, encoding='utf-8') as f:
        attributes = json.load(f)
    print('Attributes loaded.')

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        # using item_shuffle.json, we assume products already shuffled
        products = products[:num_products]
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan' or len(asin) > 10:
            continue

        if asin in asins:
            continue
        else:
            asins.add(asin)

        finish_product(p, attributes, human_attributes, human_goals, all_reviews, all_ratings)
        all_products.append(products[i])

    for p in all_products:
        for a in p['Attributes']:
            attribute_to_asins[a].add(p['asin'])

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices, attribute_to_asins
