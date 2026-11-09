#!/usr/bin/env bash
# Starts Checkout on one address, port 7860: the store runs inside on port 3000, and the demo server answers
# on 7860 and passes the store's pages through, so the viewer, the store and its links share one address.
set -e
cd "$(dirname "$0")/../.."
export WEBSHOP_ITEMS_PATH="$PWD/data/prepared_test/items.json" WEBSHOP_ATTR_PATH="$PWD/data/prepared_test/attrs.json"
export WEBSHOP_SQLITE_INDEX="$PWD/data/prepared_test/webshop.sqlite" WEBSHOP_HUMAN_PATH="$PWD/data/items_human_ins.json"
export WEBSHOP_PRODUCT_STORE="$PWD/data/products.sqlite"
export WEBSHOP_DEMO_URL='/'                             # the store links back to the viewer on the same address
export CHECKOUT_STORE_INTERNAL='http://127.0.0.1:3000'  # where the demo server finds the store

python -m web_agent_site.app --attrs &
# The store loads its products on its first page request, so make that request now, once it answers.
( until curl -fs -o /dev/null http://127.0.0.1:3000/static/store.css; do sleep 2; done
  curl -fs -o /dev/null -m 900 http://127.0.0.1:3000/new ) &
exec python demo/live_demo.py --no_browser --host 0.0.0.0 --port 7860
