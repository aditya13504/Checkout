"""The on-disk catalogue must give the same products, prices and goals as loading it all into memory."""
import json
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from web_agent_site.engine.engine import finish_product, generate_product_prices, load_products  # noqa: E402
from web_agent_site.engine.goal import get_human_goals  # noqa: E402
from web_agent_site.engine.product_store import ProductStore, build_store, load_products_from_store  # noqa: E402


def raw_product(i, asin=None):
    return {'asin': asin or f'B{i:03d}', 'name': f'Product {i}', 'full_description': f'about {i}', 'small_description': [f'point {i}'],
            'pricing': '$5.00 - $9.00' if i % 2 else '$7.00', 'customization_options': {'Color': [{'value': 'Red/Blue'}]} if i % 3 == 0 else None,
            'category': 'c', 'query': ' Thing ', 'product_category': 'a › b', 'images': ['x.jpg']}


def write_catalog(tmp_path):
    products = [raw_product(i) for i in range(30)] + [raw_product(5)]  # the last is a duplicate
    human = {f'B{i:03d}': [{'instruction': f'i want {i}', 'instruction_attributes': ['soft'], 'instruction_options': ['red']}]
             for i in range(0, 30, 4)}
    paths = {name: tmp_path / f'{name}.json' for name in ('items', 'attrs', 'human')}
    paths['items'].write_text(json.dumps(products))
    paths['attrs'].write_text(json.dumps({'B002': {'attributes': ['warm']}}))
    paths['human'].write_text(json.dumps(human))
    store = tmp_path / 'products.sqlite'
    build_store(paths['items'], paths['attrs'], paths['human'], store)
    return paths, store, products, human


def test_store_matches_loading_into_memory(tmp_path, monkeypatch):
    paths, store, _, _ = write_catalog(tmp_path)
    monkeypatch.setattr('web_agent_site.engine.engine.DEFAULT_ATTR_PATH', str(paths['attrs']))
    monkeypatch.setattr('web_agent_site.engine.engine.HUMAN_ATTR_PATH', str(paths['human']))

    random.seed(233)
    memory_products, memory_dict, memory_prices, _ = load_products(str(paths['items']), human_goals=True)
    memory_goals = get_human_goals(memory_products, memory_prices)

    random.seed(233)
    goal_products, stored, prices, _ = load_products_from_store(str(store), generate_product_prices)
    goals = get_human_goals(goal_products, prices)

    assert prices == memory_prices
    assert goals == memory_goals
    assert len(stored) == len(memory_dict) == 30
    for asin, product in memory_dict.items():
        assert stored[asin] == json.loads(json.dumps(product))
    assert dict(stored.titles()) == {asin: p['Title'] for asin, p in memory_dict.items()}


def test_unknown_products_are_missing_and_the_cache_stays_small(tmp_path, monkeypatch):
    _, store, _, _ = write_catalog(tmp_path)
    monkeypatch.setattr('web_agent_site.engine.product_store.CACHE_SIZE', 3)
    products = ProductStore(str(store))
    assert 'NOPE' not in products
    for i in range(10):
        assert products[f'B{i:03d}']['asin'] == f'B{i:03d}'
    assert len(products.cache) == 3
    assert sqlite3.connect(store).execute('SELECT COUNT(*) FROM products').fetchone()[0] == 30


def test_finish_product_gives_the_fields_the_site_needs():
    product = finish_product(raw_product(3), {}, {}, True)
    assert product['Title'] == 'Product 3' and product['Price'] == '$5.0 to $9.0' and product['pricing'] == [5.0, 9.0]
    assert product['options'] == {'color': ['red | blue']} and product['query'] == 'thing'
