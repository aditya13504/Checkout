#!/bin/bash
set -e

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-d small|all] [-n max_products]"
  echo -e "\t-d small|all - Download the 1,000 product catalogue (small) or the full one (all)"
  echo -e "\t-n N         - With -d all: keep N products, always including those with human goals"
  exit 1 # Exit script after printing help
}

max_products=""

# Get values of command line flags
while getopts d:n: flag
do
  case "${flag}" in
    d) data=${OPTARG};;
    n) max_products=${OPTARG};;
    *) helpFunction;;
  esac
done

if [ "$data" != "small" ] && [ "$data" != "all" ]; then
  echo "[ERROR]: -d must be small or all"
  helpFunction
fi

# Install Python Dependencies
pip install -r requirements.txt
pip install "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl"

# Download the catalogue. The original Google Drive links no longer work.
MIRROR=https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main
mkdir -p data
if [ "$data" == "small" ]; then
  items=items_shuffle_1000.json
  attrs=items_ins_v2_1000.json
else
  items=items_shuffle.json
  attrs=items_ins_v2.json
fi
for f in "$items" "$attrs"; do
  [ -f "data/$f" ] || curl -fL --retry 5 -C - -o "data/$f" "$MIRROR/$f"
done
cp baseline_models/data/items_human_ins.json data/items_human_ins.json

# Slim catalogue + BM25 search index
limit=""
if [ -n "$max_products" ]; then
  limit="--max_products $max_products"
fi
python search_engine/prepare_catalog.py --raw_items "data/$items" --raw_attrs "data/$attrs" \
  --human data/items_human_ins.json --out_dir data/prepared $limit

cat <<EOF

Setup finished. Before running anything, point the code at the prepared catalogue:

  export WEBSHOP_ITEMS_PATH=$PWD/data/prepared/items.json
  export WEBSHOP_ATTR_PATH=$PWD/data/prepared/attrs.json
  export WEBSHOP_SQLITE_INDEX=$PWD/data/prepared/webshop.sqlite
EOF
