"""
Turns the raw WebShop catalogue into what the environment needs:

  items.json       only the product fields that are used
  attrs.json       the attributes of those products
  webshop.sqlite   the BM25 search index

The raw file is 5.5 GB, so it is streamed instead of being loaded into memory.
With --max_products, the catalogue keeps every product that has a human instruction
and fills the rest with the first other products. Their relative order is unchanged,
so the goal list, and therefore the train/eval/test split, is the same as with the
full catalogue.

    python search_engine/prepare_catalog.py --raw_items data/items_shuffle.json \
        --raw_attrs data/items_ins_v2.json --human data/items_human_ins.json \
        --out_dir data/prepared --max_products 200000
"""
import argparse
import json
import os
import sqlite3
import time

import ijson

KEPT_FIELDS = ('asin', 'name', 'full_description', 'small_description', 'pricing',
               'customization_options', 'category', 'query', 'product_category')


def slim_product(product):
    slim = {k: product.get(k) for k in KEPT_FIELDS}
    slim['images'] = (product.get('images') or [''])[:1]
    # Each option carries an image URL for the website, which agents never read.
    options = slim['customization_options']
    if options:
        slim['customization_options'] = {
            name: None if contents is None else [{'value': c['value']} for c in contents]
            for name, contents in options.items()
        }
    return slim


def document_text(product):
    """The searchable text: the same fields as the Lucene documents of the original."""
    option_texts = []
    for name, contents in (product.get('customization_options') or {}).items():
        if contents is None:
            continue
        values = [c['value'].strip().replace('/', ' | ').lower() for c in contents]
        option_texts.append(f"{name.lower()}: {', '.join(values)}")
    bullets = product.get('small_description')
    if isinstance(bullets, list):
        first_bullet = bullets[0] if bullets else ''
    else:
        first_bullet = bullets or ''
    return ' '.join([
        product.get('name') or '',
        product.get('full_description') or '',
        first_bullet,
        ', and '.join(option_texts),
    ]).lower()


def open_index(path):
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE VIRTUAL TABLE docs USING fts5(contents, content='', tokenize='porter unicode61');
        CREATE TABLE doc_map (rowid INTEGER PRIMARY KEY, asin TEXT NOT NULL);
    """)
    return conn


def stream_products(path):
    with open(path, 'rb') as f:
        yield from ijson.items(f, 'item', use_float=True)


def prepare(raw_items, raw_attrs, human_path, out_dir, max_products=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(human_path, encoding='utf-8') as f:
        human_asins = set(json.load(f))
    with open(raw_attrs, encoding='utf-8') as f:
        attributes = json.load(f)

    conn = open_index(os.path.join(out_dir, 'webshop.sqlite'))
    seen, kept_asins, kept, kept_human, rows = set(), [], 0, 0, []
    start = time.time()
    with open(os.path.join(out_dir, 'items.json'), 'w', encoding='utf-8') as out:
        out.write('[')
        for count, product in enumerate(stream_products(raw_items), 1):
            asin = product['asin']
            if asin == 'nan' or len(asin) > 10 or asin in seen:
                continue
            seen.add(asin)
            is_human = asin in human_asins
            if max_products is not None and not is_human:
                # Leave room for the human-goal products that are still to come.
                room = max_products - (len(human_asins) - kept_human) - kept
                if room <= 0:
                    continue

            out.write((',' if kept else '') + json.dumps(slim_product(product)))
            kept += 1
            kept_asins.append(asin)
            kept_human += is_human
            rows.append((kept, asin, document_text(product)))
            if len(rows) >= 5000:
                write_rows(conn, rows)
                rows = []
                print(f'{count} read, {kept} kept, {(time.time() - start) / 60:.1f} min', flush=True)
        out.write(']')
    write_rows(conn, rows)
    conn.execute("INSERT INTO docs(docs) VALUES('optimize')")
    conn.commit()
    conn.close()

    with open(os.path.join(out_dir, 'attrs.json'), 'w', encoding='utf-8') as f:
        json.dump({asin: attributes[asin] for asin in kept_asins if asin in attributes}, f)

    summary = {'products': kept, 'human_goal_products': kept_human,
               'human_goal_products_expected': len(human_asins), 'minutes': (time.time() - start) / 60}
    with open(os.path.join(out_dir, 'catalog_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(summary, flush=True)
    return summary


def write_rows(conn, rows):
    conn.executemany('INSERT INTO doc_map(rowid, asin) VALUES (?, ?)', [(r[0], r[1]) for r in rows])
    conn.executemany('INSERT INTO docs(rowid, contents) VALUES (?, ?)', [(r[0], r[2]) for r in rows])
    conn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw_items', required=True)
    parser.add_argument('--raw_attrs', required=True)
    parser.add_argument('--human', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_products', type=int, default=None, help='catalogue size (default: all products)')
    args = parser.parse_args()
    prepare(args.raw_items, args.raw_attrs, args.human, args.out_dir, args.max_products)
