# 🤖 WebShop Baseline Models

This folder contains the agents: imitation learning, reinforcement learning and the rule baseline, along with the scripts to train and test them on WebShop. Run the commands below from this folder.

## 🚀 Set Up
* The root `requirements.txt` covers everything here except `train_search_il.py`, which also needs `pip install -r requirements.txt` in this folder.
* Unzip the training data for choice IL in the `data` folder:
```bash
cd data
unzip il_trajs_finalized_images.zip
cd ..
```
* Pretrained models are on Hugging Face: [`webshop/il-choice-bert-image_0`](https://huggingface.co/webshop/il-choice-bert-image_0) (text-only choice model) and [`webshop/il_search_bart`](https://huggingface.co/webshop/il_search_bart) (search model).

## 🛠️ Usage
➤ Train the **search IL model** (BART Transformer). Weights go to `./ckpts/web_search`.
```bash
python train_search_il.py
```

➤ Train the **choice IL model** (BERT Transformer). Weights go to `./ckpts/web_click`; see `--help` for the arguments.
```bash
python train_choice_il.py
```

➤ Train the **choice RL** models. `--bert_path` is the IL weights to start from, `--max_hours` limits the run and `--resume 1` continues it; see `--help` for the rest.
```bash
python train_rl.py --bert_path ckpts/web_click/model_best.pth --output_dir logs/il_rl
```

## 🧪 Testing
```bash
python test.py --model_path ckpts/web_click/model_best.pth --out results/il.json
python test.py --hub_model webshop/il-choice-bert-image_0 --out results/il_hub.json
python test.py --policy rule --out results/rule.json
```

### 📙 Notes about Testing
1. Model results vary because the choice policy samples its actions. `--softmax 0` always takes the best action instead.

2. `--search` sets how search pages are handled: `top1` uses the first query predicted by the search model (from `data/goal_query_predict.json`), `instruction` uses the instruction itself and `model` lets the choice model pick among them.

## 🔀 Miscellaneous
Generate the search IL model's top-10 queries on all WebShop instructions:
```bash
# Will generate ./data/goal_query_predict.json
python generate_search.py
```
