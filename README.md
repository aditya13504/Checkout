# 🛒 Checkout

**Reinforcement learning agents that shop from a one-line request.** Each agent reads a request such as *"a single pack of brown hair dye under $30"*, searches a store of 1.18 million real products, picks options and buys. The store pays its reward only at checkout, and that sparse reward is the problem this project studies.

Agents are trained in [WebShop](https://github.com/princeton-nlp/WebShop) (Princeton NLP): imitation learning (IL) on human demonstrations, advantage actor-critic (A2C) from that start or from pretrained BERT, and a value shopper that learns what a purchase will score. On the 500 test requests, the rule baseline scores 46.7, IL+RL 61.9 with greedy actions, and the value shopper 80.4, against 82.1 for human experts. The full results are in [`submission/REPORT.md`](submission/REPORT.md).

The project also repairs defects in the original code that blocked training or changed results, and speeds up training. Appendix A of the report lists every change.

## Run the live demo (Windows)

1. Set up once, which needs Python 3.10 and about 20 GB of free disk:
   `powershell -ExecutionPolicy Bypass -File demo\setup_demo.ps1`
2. Double-click `demo\START_DEMO.bat`. It opens:
   - the agent viewer, at http://127.0.0.1:3100, where the trained agents shop live
   - the results page, at http://127.0.0.1:3100/showcase/
   - the store for people to try, at http://127.0.0.1:3000

## Reproduce the experiments

Upload [`kaggle/webshop_rl_kaggle.ipynb`](kaggle/webshop_rl_kaggle.ipynb) to Kaggle, set the accelerator to *GPU T4 x2* and turn Internet on. The notebook runs every stage and writes the report; its first cell explains the steps. After downloading its outputs into `kaggle_outputs/`, rebuild the report and the hand-in zip with:

```sh
data\venv\Scripts\python.exe report\build_submission.py
```

## Set up on Linux or macOS

```sh
git clone https://github.com/aditya13504/Checkout.git && cd Checkout
python3.10 -m venv .venv && source .venv/bin/activate
./setup.sh -d small        # or -d all for the full 1.18 million products
python -m pytest tests     # the test suite
python -m web_agent_site.app --attrs   # the store, at http://127.0.0.1:3000
```

`setup.sh` installs the dependencies, downloads the catalogue from a Hugging Face mirror, builds the slim catalogue and the SQLite search index in `data/prepared`, and prints the environment variables to set.

## Layout

| Path | Purpose |
|---|---|
| `web_agent_site/` | The WebShop store and the text environment the agents act in; `templates/` are the pages agents read, `templates_human/` the pages people see |
| `web_agent_site/engine/product_store.py` | Serves the catalogue from disk, for machines with little memory |
| `search_engine/prepare_catalog.py` | Streams the raw catalogue into a slim catalogue and a BM25 index |
| `baseline_models/` | The agents: IL (`train_choice_il.py`), batched A2C (`train_rl.py`, `agent.py`, `vec_env.py`), the value shopper (`value_model.py`, `train_value_model.py`), the careful shopper and assist mode (`shopping_rules.py`), and evaluation (`test.py`) |
| `kaggle/` | The pipeline of all experiment stages, and the notebook built from it |
| `report/` | Report template, written interpretation, the generator, and `build_submission.py` |
| `demo/` | Live demo: agent viewer, launcher and one-time setup |
| `showcase/` | Interactive results page, built from the outputs by `build_data.py` |
| `kaggle_outputs/` | The reported Kaggle run: results, logs and training metrics |
| `kaggle_outputs_run1/` | Results of the earlier run that the report compares with |
| `results_cpu/` | Validation and earlier test runs made on a laptop |
| `submission/` | The report, in Markdown and Word, with its figures |
| `tests/` | Tests of the environment, store, pipeline and agents |

## Credits

The environment, reward, network architecture and demonstration data come from WebShop by Shunyu Yao, Howard Chen, John Yang and Karthik Narasimhan, *[WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206)*, NeurIPS 2022. See [`LICENSE.md`](LICENSE.md).
