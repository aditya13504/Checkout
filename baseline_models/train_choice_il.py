"""
Trains the choice model by imitation learning (behaviour cloning) on human trajectories.

Each sample is a page, its valid actions and the action the human took. Training uses
dynamic padding, 16-bit mixed precision and large batches, and reduces a batch if the
GPU runs out of memory.
"""
import argparse
import csv
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.bert import BertConfigForWebshop, BertModelForWebshop

# Keeps the start of the page, with the instruction and the top results; see agent.py.
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side='right')
tokenizer.add_tokens(['[button]', '[button_]', '[clicked button]', '[clicked button_]'], special_tokens=True)

PATH = "./data/il_trajs_finalized_images.jsonl"
HUMAN_GOAL_PATH = './data/human_goals.json'
STATE_MAX_LEN = 512
ACTION_MAX_LEN = 128


def process(s):
    s = s.lower().replace('"', '').replace("'", "").strip()
    s = s.replace('[sep]', '[SEP]')
    return s


def process_goal(state):
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    return state


def get_data(split, path=PATH, filter_search=True):
    """Reads the (state, valid actions, chosen action) samples of a split."""
    with open(path, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)
    with open(HUMAN_GOAL_PATH, 'r', encoding='utf-8') as f:
        human_goals = json.load(f)
    goal_index = {}
    for i, goal in enumerate(human_goals):
        goal_index.setdefault(goal, i)

    rng = random.Random(233)
    rng.shuffle(json_list)

    goal_ranges = {
        'train': range(1500, len(human_goals)),
        'eval': range(500, 1500),
        'test': range(0, 500),
    }
    goal_range = goal_ranges.get(split, range(len(human_goals)))

    samples, num_trajs, total, reduced = [], 0, 0, 0
    for json_str in json_list:
        result = json.loads(json_str)
        goal_idx = goal_index[process_goal(result['states'][0])]
        if goal_idx not in goal_range:
            continue
        num_trajs += 1
        images = result.get('images', [0] * len(result['states']))
        for state, valid_acts, idx, image in zip(result['states'], result['available_actions'],
                                                 result['action_idxs'], images):
            total += 1
            if filter_search and idx == -1:
                continue
            if len(valid_acts) > 20:  # do some action space reduction...
                reduced += 1
                new_idxs = list(range(6)) + rng.sample(range(6, len(valid_acts)), 10)
                if idx not in new_idxs:
                    new_idxs += [idx]
                new_idxs = sorted(new_idxs)
                valid_acts = [valid_acts[i] for i in new_idxs]
                idx = new_idxs.index(idx)
            samples.append({
                'state': process(state),
                'actions': [process(a) for a in valid_acts],
                'label': idx,
                'image': [0.] * 512 if image == 0 else image,
            })
    print(f'{split}: {num_trajs} trajectories, {total} transitions, '
          f'{len(samples)} samples, {reduced} with reduced action sets', flush=True)
    return samples


class ChoiceDataset(Dataset):
    def __init__(self, samples):
        self.states = tokenizer([s['state'] for s in samples], truncation=True,
                                max_length=STATE_MAX_LEN)['input_ids']
        flat_actions = tokenizer([a for s in samples for a in s['actions']], truncation=True,
                                 max_length=ACTION_MAX_LEN)['input_ids']
        self.actions, start = [], 0
        for s in samples:
            self.actions.append(flat_actions[start:start + len(s['actions'])])
            start += len(s['actions'])
        self.labels = [s['label'] for s in samples]
        self.images = [s['image'] for s in samples]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i], self.actions[i], self.labels[i], self.images[i]


def pad(sequences):
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    ids = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return ids, (ids > 0).long()


def data_collator(batch):
    """Pads to the longest sequence of the batch (not to the maximum length)."""
    state_ids, state_mask = pad([b[0] for b in batch])
    action_ids, action_mask = pad([a for b in batch for a in b[1]])
    return {
        'state_input_ids': state_ids,
        'state_attention_mask': state_mask,
        'action_input_ids': action_ids,
        'action_attention_mask': action_mask,
        'sizes': torch.tensor([len(b[1]) for b in batch]),
        'images': torch.tensor([b[3] for b in batch], dtype=torch.float),
        'labels': torch.tensor([b[2] for b in batch]),
    }


def split_batch(batch, parts):
    """Splits a batch by states, keeping each state's actions with it."""
    n = len(batch['sizes'])
    action_ends = batch['sizes'].cumsum(0).tolist()
    chunks = []
    for start in range(0, n, max(n // parts, 1)):
        end = min(start + max(n // parts, 1), n)
        first_action = action_ends[start - 1] if start else 0
        last_action = action_ends[end - 1]
        chunks.append({
            key: value[first_action:last_action] if key.startswith('action_') else value[start:end]
            for key, value in batch.items()
        })
    return chunks


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class BatchSplitter:
    """Remembers how finely batches must be split, so that one size is not retried on every batch."""
    parts = 1


def backward_batch(model, optimizer, batch, scaler, device, use_amp, splitter=None):
    """Computes the gradient of a batch, in more pieces if the GPU runs out of memory."""
    n = len(batch['sizes'])
    splitter = splitter or BatchSplitter()
    parts = splitter.parts
    while True:
        try:
            optimizer.zero_grad(set_to_none=True)
            total_loss, correct = 0.0, 0
            for chunk in split_batch(batch, parts):
                count = len(chunk['sizes'])
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(**to_device(chunk, device))
                    loss = outputs.loss * count / n
                scaler.scale(loss).backward()
                total_loss += float(outputs.loss) * count
                labels = chunk['labels'].to(device)
                correct += sum(int(logit.argmax(0) == label) for logit, label in zip(outputs.logits, labels))
            return total_loss, correct
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower() or parts >= n:
                raise
            parts *= 2
            splitter.parts = parts
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f'GPU out of memory: splitting batches into {parts}', flush=True)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**to_device(batch, device))
        labels = batch['labels'].to(device)
        correct += sum(int(logit.argmax(0) == label) for logit, label in zip(outputs.logits, labels))
        total_loss += float(outputs.loss) * len(labels)
        count += len(labels)
    return total_loss / max(count, 1), correct / max(count, 1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the choice model by imitation learning')
    parser.add_argument('--data_path', default=PATH)
    parser.add_argument('--output_dir', default='./ckpts/web_click')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--max_minutes', type=float, default=0, help='stop after this long (0 = no limit)')
    parser.add_argument('--max_train_samples', type=int, default=0, help='for quick tests')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image', type=int, default=0, help='use image features')
    parser.add_argument('--pretrain', type=int, default=1, help='start from pretrained BERT')
    parser.add_argument('--bert_layers', type=int, default=12, help='layers when --pretrain 0')
    parser.add_argument('--amp', type=int, default=1, help='16-bit mixed precision on GPU')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and bool(args.amp)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    config = BertConfigForWebshop(image=bool(args.image), pretrained_bert=bool(args.pretrain),
                                  bert_layers=args.bert_layers)
    model = BertModelForWebshop(config).to(device)

    train_samples = get_data('train', args.data_path)
    eval_samples = get_data('eval', args.data_path)
    if args.max_train_samples:
        train_samples = train_samples[:args.max_train_samples]
    pin = device.type == 'cuda'
    train_loader = DataLoader(ChoiceDataset(train_samples), shuffle=True, collate_fn=data_collator,
                              batch_size=args.per_device_train_batch_size, pin_memory=pin)
    eval_loader = DataLoader(ChoiceDataset(eval_samples), collate_fn=data_collator,
                             batch_size=args.per_device_eval_batch_size, pin_memory=pin)

    no_decay = ('bias', 'LayerNorm.weight')
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ], lr=args.learning_rate)
    total_steps = args.num_train_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = os.path.join(args.output_dir, 'il_metrics.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'elapsed_min', 'train_loss', 'train_acc', 'eval_loss', 'eval_acc'])

    best_acc, start, splitter = -1.0, time.time(), BatchSplitter()
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f'epoch {epoch}', mininterval=30):
            loss_sum, n_correct = backward_batch(model, optimizer, batch, scaler, device, use_amp, splitter)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss_sum
            correct += n_correct
            count += len(batch['sizes'])
            if args.max_minutes and (time.time() - start) / 60 >= args.max_minutes:
                break  # an epoch can be long, so the limit is also checked between batches

        eval_loss, eval_acc = evaluate(model, eval_loader, device, use_amp)
        minutes = (time.time() - start) / 60
        print(f'epoch {epoch}: {minutes:.1f} min, train loss {total_loss / count:.3f}, '
              f'train acc {correct / count:.3f}, eval loss {eval_loss:.3f}, eval acc {eval_acc:.3f}', flush=True)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, minutes, total_loss / count, correct / count, eval_loss, eval_acc])

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_best.pth'))
        if args.max_minutes and minutes >= args.max_minutes:
            print('Time limit reached', flush=True)
            break

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.pth'))
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        json.dump({'best_eval_accuracy': best_acc, 'minutes': (time.time() - start) / 60,
                   'train_samples': len(train_samples), 'eval_samples': len(eval_samples),
                   'args': vars(args)}, f, indent=2)


if __name__ == '__main__':
    main()
