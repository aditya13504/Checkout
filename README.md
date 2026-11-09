# 🛒 Checkout

**Reinforcement learning agents that shop from a one-line request.** Each agent reads a request such as *"a single pack of brown hair dye under $30"*, searches a store of 1.18 million real products, picks options and buys. The store pays its reward only at checkout, and that sparse reward is the problem this project studies.

Agents are trained in [WebShop](https://github.com/princeton-nlp/WebShop) (Princeton NLP): imitation learning (IL) on human demonstrations, advantage actor-critic (A2C) from that start or from pretrained BERT, and a value shopper that learns what a purchase will score. On the 500 test requests, the rule baseline scores 46.7, IL+RL 61.9 with greedy actions, and the value shopper 80.4, against 82.1 for human experts. The full results are in [`submission/REPORT.md`](submission/REPORT.md).

The project repairs defects in the original code that blocked training or changed results, and speeds up training; Appendix A of the report lists every change.

**Run the live demo** (Windows): double-click `demo\START_DEMO.bat`, after setting up once with `powershell -ExecutionPolicy Bypass -File demo\setup_demo.ps1`. It opens the agent viewer at http://127.0.0.1:3100, the results page at http://127.0.0.1:3100/showcase/, and the store for people to try at http://127.0.0.1:3000.

**Reproduce the experiments:** upload [`kaggle/webshop_rl_kaggle.ipynb`](kaggle/webshop_rl_kaggle.ipynb) to Kaggle, set the accelerator to *GPU T4 x2* and turn Internet on. The notebook runs everything and writes the report. Its first cell explains the steps. Then `data\venv\Scripts\python.exe report\build_submission.py` rebuilds the report and the hand-in zip from the downloaded outputs.

**Layout added for this project:**

| Path | Purpose |
|---|---|
| `kaggle/pipeline.py` | All experiment stages, resumable |
| `kaggle/build_notebook.py` | Regenerates the notebook after a code change |
| `baseline_models/train_rl.py`, `agent.py`, `vec_env.py` | Batched A2C with mixed precision and parallel environments |
| `baseline_models/value_model.py`, `train_value_model.py`, `shopping_rules.py` | The value shopper, the careful shopper and assist mode |
| `search_engine/prepare_catalog.py` | Streams the raw catalogue into a slim catalogue and a BM25 index |
| `web_agent_site/engine/product_store.py` | Serves the catalogue from disk, for machines with little memory |
| `web_agent_site/templates_human/` | The store's pages for people; agents keep the original templates |
| `report/` | Report template, generator and `build_submission.py` |
| `demo/` | Live demo: agent viewer, launcher and one-time setup |
| `showcase/` | Interactive results page, built from the outputs by `build_data.py` |
| `kaggle_outputs/` | The reported Kaggle run; `kaggle_outputs_run1/` is the earlier run it is compared with |
| `results_cpu/` | Validation and earlier test runs made on a laptop |
| `submission/` | The report (Markdown and Word) and the hand-in zip |

The original README follows, updated where its instructions had stopped working.

---

# 🛒 WebShop

[![Python version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release/python-3813/)
[![License](https://img.shields.io/badge/License-Princeton-orange)](https://copyright.princeton.edu/policy)
[![PyPI version](https://badge.fury.io/py/webshop.svg)](https://badge.fury.io/py/webshop)
![Pytest workflow](https://github.com/princeton-nlp/webshop/actions/workflows/pytest.yml/badge.svg)

Implementation of the WebShop environment and search agents for the paper:

**[WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://webshop-pnlp.github.io/)**  
[Shunyu Yao*](https://ysymyth.github.io/), [Howard Chen*](https://howard50b.github.io/), [John Yang](https://john-b-yang.github.io/), [Karthik Narasimhan](https://www.cs.princeton.edu/~karthikn/)

<p float="left">
  <img src="assets/diagram.gif">
</p>

This repository contains code for reproducing results. If you find this work useful in your research, please cite:

```
@inproceedings{yao2022webshop,
  bibtex_show = {true},
  title = {WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents},
  author = {Yao, Shunyu and Chen, Howard and Yang, John and Narasimhan, Karthik},
  booktitle = {ArXiv},
  year = {preprint},
  html = {https://arxiv.org/abs/2207.01206},
  tag = {NLP}
}
```
## 📖 Table of Contents <!-- omit in toc -->
* [👋 Overview](#-overview)
* [🚀 Setup](#-setup)
* [🛠️ Usage](#-usage)
* [💫 Contributions](#-contributions)
* [🪪 License](#-license)
## 👋 Overview
WebShop is a simulated e-commerce website environment with 1.18 million real-world products and 12,087 crowd-sourced text instructions. In this environment, an agent needs to navigate multiple types of webpages and issue diverse actions to find, customize, and purchase a product given an instruction. WebShop provides several challenges including understanding compositional instructions, query (re-)formulation, dealing with noisy text in webpages, and performing strategic exploration.

**Hugging Face Demo**: Devise your own natural language query for a product and ask for an agent trained with WebShop to find it on Amazon or eBay, deployed as a 🤗 Hugging Face space [here](https://huggingface.co/spaces/webshop/amazon_shop)!

## 🚀 Setup
Our code is implemented in Python. To setup, do the following:
1. Install Python 3.10
2. Download the source code:
```sh
> git clone https://github.com/aditya13504/Checkout.git
```
3. Create a virtual environment and activate it
```sh
> python -m venv .venv
> source .venv/bin/activate
```
4. Run the setup script
```sh
> ./setup.sh -d small|all [-n max_products]
```
It installs the dependencies, downloads the data from a Hugging Face mirror (the original Google Drive links no longer work), and builds the slim catalogue and the search index in `data/prepared`. `-d small` gets 1,000 products and `-d all` gets 1.18 million; add `-n` to keep fewer, always including the products with human goals.

5. Set the environment variables that the script prints, so that the code uses the prepared catalogue.

The search index uses SQLite, so Java is not required.

## 🛠️ Usage
The WebShop environment can be rendered in two modes - `html` and `simple` - each of which offer a different observation space. The `simple` mode strips away the extraneous meta-data that the `html` mode includes to make model training and evaluation easier.
### Webpage Environment (`html` mode)
Launch the `WebShop` webpage:
```sh
> ./run_dev.sh
```
The site should then be viewable in the browser. Go to http://localhost:3000/ABC, where you should land on the search home page with a random instruction. It only listens on your own machine unless you pass `--host 0.0.0.0`.

Navigating the website will automatically generate a corresponding trajectory file in the `user_session_logs/mturk` folder. Each file corresponds to a single instruction/web session, and each step of the file corresponds to a single action (i.e. `search[...]`, `click[...]`).

The current WebShop build comes with two flags:
* `--log`: Include this flag to create a trajectory `.jsonl` log file of actions on WebShop
* `--attrs`: Include this flag to display an `Attributes` tab on the `item_page` of WebShop

### Text Environment (`simple` mode)
The `simple` mode of the WebShop environment is packaged and readily available as an OpenAI environment. The OpenAI gym definitions of the text environment can be found in the `web_agent_site/envs` folder.

To start using the gym and building agents that interact with the WebShop environment, include the following statements in your Python file:
```python
import gym
from web_agent_site.envs import WebAgentTextEnv

env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=...)
```
Now, you can write your own agent that interacts with the environment via the standard OpenAI gym [interface](https://www.gymlibrary.ml/content/api/).

Examples of a `RandomPolicy` agent interacting with the WebShop environment in both `html` and `simple` mode can be found in the `run_envs` folder. To run these examples locally, run the `run_web_agent_text_env.sh` or `run_web_agent_site_env.sh` script:
```sh
> ./run_web_agent_text_env.sh
Products loaded.
Keys Cleaned.
Attributes Loaded.
100%|██████████████████| 1000/1000
Loaded 6910 goals.
Amazon Shopping Game [SEP] Instruction: [SEP] Find me slim f...
Available actions: {'has_search_bar': True, 'clickables': ['search']}
Taking action "search[shoes]" -> Reward = 0.0
...
```
In order to run the `run_web_agent_site_env.sh` script, you must download a version of [ChromeDriver](https://chromedriver.chromium.org/downloads) compatible with your Chrome browser version. Once you have downloaded and unzipped the executable, rename it `chromedriver` and place it in the `web_agent_site/envs` folder.

### Baseline Models
To run baseline models (rule, IL, RL, IL+RL) from the paper, please refer to the `README.md` in the [baseline_models](baseline_models/README.md) folder.

### Sim-to-real Transfer
The original repository also transfers agents to Amazon and eBay. That code is outside this project's scope and is not included; see the [original repository](https://github.com/princeton-nlp/webshop/tree/master/transfer).

## 🪪 License
Check `LICENSE.md`
