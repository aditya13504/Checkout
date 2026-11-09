"""
Serves the catalogue from disk, for machines without the RAM to hold all 1.18 million products.

An agent only reads the products that search returns, so each is loaded when asked for. The prices
and goals come out the same as with `load_products`, because the random draws are made for the same
products in the same order.

    python -m web_agent_site.engine.product_store --items data/prepared_test/items.json \
        --attrs data/prepared_test/attrs.json --human data/items_human_ins.json --out data/products.sqlite

Set WEBSHOP_PRODUCT_STORE to the output file to use it.
"""
import argparse
import json
import os
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping

CACHE_SIZE = 20000


class ProductStore(Mapping):
    """asin -> product, read from SQLite and kept in a small cache."""

    def __init__(self, path):
        self.conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True, check_same_thread=False)
        self.lock = threading.Lock()
        self.cache = OrderedDict()

    def __getitem__(self, asin):
        with self.lock:
            if asin in self.cache:
                self.cache.move_to_end(asin)
                return self.cache[asin]
            row = self.conn.execute('SELECT product FROM products WHERE asin = ?', (asin,)).fetchone()
            if row is None:
                raise KeyError(asin)
            product = json.loads(row[0])
            self.cache[asin] = product
            if len(self.cache) > CACHE_SIZE:
                self.cache.popitem(last=False)
            return product

    def __iter__(self):
        return (row[0] for row in self.conn.execute('SELECT asin FROM products ORDER BY seq'))

    def __len__(self):
        return self.conn.execute('SELECT COUNT(*) FROM products').fetchone()[0]

    def titles(self):
        """(asin, title) of every product without loading the products."""
        return self.conn.execute('SELECT asin, title FROM products ORDER BY seq')

    def prices_input(self):
        """Just enough of every product, in catalogue order, to draw its hidden price."""
        for asin, pricing in self.conn.execute('SELECT asin, pricing FROM products ORDER BY seq'):
            yield {'asin': asin, 'pricing': json.loads(pricing)}

    def human_goal_products(self):
        rows = self.conn.execute('SELECT product FROM products WHERE has_instructions = 1 ORDER BY seq')
        return [json.loads(row[0]) for row in rows]


def load_products_from_store(path, generate_product_prices):
    """What `load_products` returns; `all_products` holds only the products that have goals."""
    store = ProductStore(path)
    prices = generate_product_prices(store.prices_input())
    return store.human_goal_products(), store, prices, {}


def build_store(items_path, attrs_path, human_path, out_path):
    import ijson

    from web_agent_site.engine.engine import finish_product

    with open(attrs_path, encoding='utf-8') as f:
        attributes = json.load(f)
    with open(human_path, encoding='utf-8') as f:
        human_attributes = json.load(f)
    if os.path.exists(out_path):
        os.remove(out_path)
    conn = sqlite3.connect(out_path)
    conn.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE products (seq INTEGER PRIMARY KEY, asin TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
                               pricing TEXT NOT NULL, has_instructions INTEGER NOT NULL, product TEXT NOT NULL);
    """)
    seen, rows = set(), []
    with open(items_path, 'rb') as f:
        for product in ijson.items(f, 'item', use_float=True):
            asin = product['asin']
            if asin == 'nan' or len(asin) > 10 or asin in seen:
                continue
            seen.add(asin)
            finish_product(product, attributes, human_attributes, True)
            rows.append((len(seen), asin, product['Title'], json.dumps(product['pricing']), int('instructions' in product),
                         json.dumps(product)))
            if len(rows) >= 5000:
                conn.executemany('INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)', rows)
                conn.commit()
                rows = []
                print(f'{len(seen)} products', flush=True)
    conn.executemany('INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)', rows)
    conn.execute('CREATE INDEX goal_products ON products (has_instructions)')
    conn.commit()
    conn.close()
    print(f'{len(seen)} products stored in {out_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--items', required=True)
    parser.add_argument('--attrs', required=True)
    parser.add_argument('--human', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    build_store(args.items, args.attrs, args.human, args.out)
