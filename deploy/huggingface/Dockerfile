# Checkout on a Hugging Face Space (Docker SDK). The catalogue is built into the image, so the Space wakes
# in a minute or two instead of rebuilding it. push_space.ps1 copies this file to the Space's root.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user
WORKDIR /home/user/app

# Python packages: CPU PyTorch, and the spaCy model the store's reward uses.
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt \
 && pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl"

# The catalogue: downloaded, checked, turned into the search index and the product store, with the
# downloads removed in the same step, so the image keeps only what the site reads (about 6 GB).
# Only the files this step needs are copied first, so a code change does not rebuild it.
COPY search_engine/prepare_catalog.py search_engine/prepare_catalog.py
COPY web_agent_site/__init__.py web_agent_site/__init__.py
COPY web_agent_site/engine/__init__.py web_agent_site/engine/product_store.py web_agent_site/engine/
COPY baseline_models/data/items_human_ins.json data/items_human_ins.json
ARG MIRROR=https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main
RUN curl -fL --retry 5 -o data/items_shuffle.json "$MIRROR/items_shuffle.json" \
 && curl -fL --retry 5 -o data/items_ins_v2.json "$MIRROR/items_ins_v2.json" \
 && echo "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8  data/items_shuffle.json" | sha256sum -c - \
 && echo "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e  data/items_ins_v2.json" | sha256sum -c - \
 && python search_engine/prepare_catalog.py --raw_items data/items_shuffle.json --raw_attrs data/items_ins_v2.json \
      --human data/items_human_ins.json --out_dir data/prepared_test \
 && python -m web_agent_site.engine.product_store --items data/prepared_test/items.json \
      --attrs data/prepared_test/attrs.json --human data/items_human_ins.json --out data/products.sqlite \
 && rm data/items_shuffle.json data/items_ins_v2.json data/prepared_test/items.json data/prepared_test/attrs.json \
 && chmod -R a+rX data

# The code. The Space runs it as user 1000, who only needs to read.
COPY . .
USER user
EXPOSE 7860
CMD ["bash", "deploy/huggingface/start.sh"]
