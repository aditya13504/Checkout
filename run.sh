#!/usr/bin/env bash
# Checkout: gets, sets up and runs the whole project on macOS (and Linux), end to end.
#
#   curl -fsSL https://raw.githubusercontent.com/aditya13504/Checkout/main/run.sh | bash
#
# Inside a clone, `bash run.sh` does the same. Finished steps are skipped, so later runs start in a
# few minutes. Set CHECKOUT_SKIP_TESTS=1 to skip the tests.
# The first run downloads about 7 GB and needs about 20 GB of free disk; 8 GB stays in data/.
set -euo pipefail
REPO=https://github.com/aditya13504/Checkout
MODELS_URL=$REPO/releases/download/models-v1
CATALOGUE_URL=https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main

step() { printf '\n\033[33m=== %s ===\033[0m\n' "$1"; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fetch() {  # url file sha256
  [ -f "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$2.part" "$1"
  if [ "$(sha256 "$2.part")" != "$3" ]; then rm -f "$2.part"; echo "$1 arrived damaged; run the script again." >&2; exit 1; fi
  mv "$2.part" "$2"
}
open_url() { if [ "$(uname)" = Darwin ]; then open "$1"; elif command -v xdg-open >/dev/null; then xdg-open "$1" >/dev/null 2>&1 || true; fi; }

# ---------------------------------------------------------------- the code
step 'Getting the code'
if [ -f kaggle/pipeline.py ]; then ROOT=$PWD; else ROOT=$PWD/Checkout; fi
if [ -f "$ROOT/kaggle/pipeline.py" ]; then
  if [ -d "$ROOT/.git" ] && git --version >/dev/null 2>&1; then
    git -C "$ROOT" pull --ff-only || echo 'Could not update the code (local changes?); using it as it is.'
  fi
elif git --version >/dev/null 2>&1; then
  git clone "$REPO.git" "$ROOT"
else
  curl -fL "$REPO/archive/refs/heads/main.tar.gz" | tar -xz -C "$(dirname "$ROOT")"
  mv "$(dirname "$ROOT")/Checkout-main" "$ROOT"
fi
cd "$ROOT"
DATA=$ROOT/data
PY=$DATA/venv/bin/python
mkdir -p "$DATA"
FREE_GB=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4 / 1048576)}')
if [ ! -f "$DATA/products.sqlite" ] && [ "$FREE_GB" -lt 20 ]; then
  echo "The first run needs about 20 GB of free disk; this disk has $FREE_GB GB." >&2; exit 1
fi

# ---------------------------------------------------------------- Python
step 'Python 3.10 environment'
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -f "$DATA/venv/.installed" ]; then
  [ -x "$PY" ] || uv venv --python 3.10 "$DATA/venv"
  grep -v '^torch' requirements.txt > "$DATA/requirements_cpu.txt"
  uv pip install --python "$PY" -r "$DATA/requirements_cpu.txt" pypandoc_binary
  uv pip install --python "$PY" torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python "$PY" 'en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl'
  touch "$DATA/venv/.installed"
fi

export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false HF_HOME=$DATA/hf
PREPARED=$DATA/prepared_test
export WEBSHOP_ITEMS_PATH=$PREPARED/items.json WEBSHOP_ATTR_PATH=$PREPARED/attrs.json
export WEBSHOP_SQLITE_INDEX=$PREPARED/webshop.sqlite WEBSHOP_HUMAN_PATH=$DATA/items_human_ins.json
export WEBSHOP_PRODUCT_STORE=$DATA/products.sqlite

# ---------------------------------------------------------------- the store
step 'Product catalogue, search index and product store'
cp baseline_models/data/items_human_ins.json "$WEBSHOP_HUMAN_PATH"
if [ ! -f "$WEBSHOP_SQLITE_INDEX" ] || [ ! -f "$WEBSHOP_PRODUCT_STORE" ]; then
  echo 'Downloading 5.7 GB of products, then indexing them (about 20 minutes).'
  fetch "$CATALOGUE_URL/items_shuffle.json" "$DATA/items_shuffle.json" 2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8
  fetch "$CATALOGUE_URL/items_ins_v2.json" "$DATA/items_ins_v2.json" 1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e
  rm -f "$WEBSHOP_SQLITE_INDEX" "$WEBSHOP_PRODUCT_STORE"
  "$PY" search_engine/prepare_catalog.py --raw_items "$DATA/items_shuffle.json" --raw_attrs "$DATA/items_ins_v2.json" \
    --human "$WEBSHOP_HUMAN_PATH" --out_dir "$PREPARED"
  "$PY" -m web_agent_site.engine.product_store --items "$WEBSHOP_ITEMS_PATH" --attrs "$WEBSHOP_ATTR_PATH" \
    --human "$WEBSHOP_HUMAN_PATH" --out "$WEBSHOP_PRODUCT_STORE"
  # Only the index and the store are needed from here on.
  rm -f "$DATA/items_shuffle.json" "$DATA/items_ins_v2.json" "$WEBSHOP_ITEMS_PATH" "$WEBSHOP_ATTR_PATH"
fi

# ---------------------------------------------------------------- models
step 'BERT and the trained agents'
"$PY" -c "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
RUNS=$ROOT/kaggle_outputs/outputs_full/runs
fetch "$MODELS_URL/il_rl.pt" "$RUNS/il_rl/model_best.pt" abc18e43721eb30fa3cc8de4326894f873ed1ff037cde84487cf52d4b3a0471a
fetch "$MODELS_URL/rl_scratch.pt" "$RUNS/rl_scratch/model_best.pt" 5e23149bc671f991be376a429bae6de976e72152115a60be381017907f594705
fetch "$MODELS_URL/il_seed0.pth" "$RUNS/il_seed0/model_best.pth" 3d26ab737a5f77b8ec46fc99f7d980a40beed9e6e34e4382ea2e54c5c45f1648

# ---------------------------------------------------------------- checks and rebuilds
if [ "${CHECKOUT_SKIP_TESTS:-0}" != 1 ]; then
  step 'Tests'
  "$PY" -m pytest -q -p no:cacheprovider --disable-warnings tests
fi
step 'Report and results page, rebuilt from the run outputs'
"$PY" report/build_submission.py --out "$DATA/report"
"$PY" showcase/build_data.py

# ---------------------------------------------------------------- run
step 'Starting the store and the live demo'
if curl -fs -m 3 -o /dev/null http://127.0.0.1:3100/api/setup; then
  echo 'Checkout is already running; opening it.'
  open_url http://127.0.0.1:3100/; open_url http://127.0.0.1:3100/showcase/; open_url http://127.0.0.1:3000/new
  exit 0
fi
mkdir -p "$DATA/logs"
"$PY" -m web_agent_site.app --attrs > "$DATA/logs/store.log" 2>&1 &
STORE=$!
"$PY" demo/live_demo.py --no_browser > "$DATA/logs/demo.log" 2>&1 &
DEMO=$!
trap 'kill $STORE $DEMO 2>/dev/null || true' EXIT INT TERM
echo 'Loading 1,181,430 products into both servers (1 to 3 minutes)...'
WARM=''
for _ in $(seq 300); do
  sleep 3
  kill -0 $STORE 2>/dev/null && kill -0 $DEMO 2>/dev/null || { echo "A server stopped; see the logs in $DATA/logs." >&2; exit 1; }
  # The store loads its products on the first page request, so it gets exactly one.
  if [ -z "$WARM" ] && curl -fs -m 3 -o /dev/null http://127.0.0.1:3000/static/store.css; then
    curl -fsL -m 900 -o /dev/null http://127.0.0.1:3000/new & WARM=$!
  fi
  [ -n "$WARM" ] && curl -fs -m 5 http://127.0.0.1:3100/api/setup | grep -q '"status": "ready"' && break
done
[ -n "$WARM" ] && wait "$WARM" || true
open_url http://127.0.0.1:3100/
open_url http://127.0.0.1:3100/showcase/
open_url http://127.0.0.1:3000/new
printf '\n\033[32mCheckout is running:\033[0m\n'
echo '  Live agent demo    http://127.0.0.1:3100/'
echo '  Results page       http://127.0.0.1:3100/showcase/'
echo '  Store to try       http://127.0.0.1:3000/new'
echo '  Report             data/report/REPORT.md and REPORT.docx, rebuilt just now'
printf '\nPress Ctrl+C to stop.\n'
wait $DEMO
