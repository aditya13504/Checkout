import os
import random
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from models.bert import BertConfigForWebshop, BertModelForWebshop
from models.rnn import RCDQN

STATE_MAX_LEN = 512
ACTION_MAX_LEN = 128
BUTTON_TOKENS = ['[button]', '[button_]', '[clicked button]', '[clicked button_]']

# A GPU micro-batch is limited by its token count. The limit adapts during training:
# it grows while there is spare GPU memory and halves on an out-of-memory error.
INITIAL_TOKEN_BUDGET = 16000
MIN_TOKEN_BUDGET = 1024
TARGET_GPU_MEMORY_SHARE = 0.80

State = namedtuple('State', ('obs', 'goal', 'obs_str', 'goal_str', 'image_feat'))
Transition = namedtuple('Transition', ('states', 'act_idxs', 'rewards', 'valid_ids', 'dones'))


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def discounted_returns(transitions, last_values, gamma):
    """n-step returns R_t = r_t + gamma * R_{t+1}, cut at episode ends."""
    returns = [None] * len(transitions)
    running = last_values.detach().float().cpu()
    for t in reversed(range(len(transitions))):
        rewards = torch.tensor(transitions[t].rewards, dtype=torch.float)
        not_done = 1.0 - torch.tensor(transitions[t].dones, dtype=torch.float)
        running = rewards + gamma * running * not_done
        returns[t] = running
    return returns


def clean_text(text):
    text = text.lower().replace('"', '').replace("'", "").strip()
    return text.replace('[sep]', '[SEP]')


class Agent:
    def __init__(self, args):
        self.device = get_device()
        self.use_amp = self.device.type == 'cuda' and bool(args.amp)
        self.token_budget = args.token_budget or INITIAL_TOKEN_BUDGET

        # 'right' keeps the start of the page, with the instruction and the top results. 'left' (the
        # original behaviour) cuts the instruction from about half of all result pages.
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',
                                                       truncation_side=getattr(args, 'truncation', 'right'))
        self.tokenizer.add_tokens(BUTTON_TOKENS, special_tokens=True)
        self.action_cache = {}

        if args.network == 'rnn':
            self.network = RCDQN(len(self.tokenizer), args.embedding_dim, args.hidden_dim, args.get_image)
            self.network.rl_forward = self.network.forward
        elif args.network == 'bert':
            config = BertConfigForWebshop(image=bool(args.get_image), pretrained_bert=(args.bert_path != 'scratch'))
            self.network = BertModelForWebshop(config)
            if args.bert_path not in ('', 'scratch'):
                self.load_weights(args.bert_path)
        else:
            raise ValueError('Unknown network: {}'.format(args.network))
        # No dropout in RL: the policy that is updated must be the one that acted.
        self.network = self.network.to(self.device).eval()

        self.save_path = args.output_dir
        self.clip = args.clip
        self.w = {'loss_pg': args.w_pg, 'loss_td': args.w_td, 'loss_il': args.w_il, 'loss_en': args.w_en}
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.gamma = args.gamma

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp)

    def load_weights(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        missing, _ = self.network.load_state_dict(state, strict=False)
        missing = [k for k in missing if 'position_ids' not in k and not k.startswith('linear_3')]
        if missing:
            raise ValueError(f'{path} does not match the model; missing {missing[:5]}')

    # ---------------------------------------------------------------- encoding

    def encode_texts(self, texts, max_length):
        return self.tokenizer(texts, truncation=True, max_length=max_length,
                              add_special_tokens=True)['input_ids']

    def encode_actions(self, actions):
        """Encodes action strings; they repeat a lot, so they are cached."""
        unknown = [a for a in set(actions) if a not in self.action_cache]
        if unknown:
            if len(self.action_cache) > 500_000:
                self.action_cache.clear()
            encoded = self.encode_texts([clean_text(a) for a in unknown], ACTION_MAX_LEN)
            self.action_cache.update(zip(unknown, encoded))
        return [self.action_cache[a] for a in actions]

    def build_states(self, obs_list, infos):
        """Builds the state representations of a batch of observations."""
        obs_strs = [clean_text(ob) for ob in obs_list]
        goal_strs = [clean_text(info['goal']) for info in infos]
        obs_ids = self.encode_texts(obs_strs, STATE_MAX_LEN)
        goal_ids = self.encode_texts(goal_strs, STATE_MAX_LEN)
        return [
            State(o, g, s, gs, info.get('image_feat'))
            for o, g, s, gs, info in zip(obs_ids, goal_ids, obs_strs, goal_strs, infos)
        ]

    def build_state(self, ob, info):
        return self.build_states([ob], [info])[0]

    def encode_valids(self, valids):
        """Encodes a list of lists of action strings."""
        return [self.encode_actions(valid) for valid in valids]

    # ------------------------------------------------------------------ acting

    def act(self, states, valid_acts, method='softmax', eps=0.1):
        """Picks one action per state; returns (strings, indices, valid ids, values)."""
        valid_ids = self.encode_valids(valid_acts)
        with torch.no_grad(), self.autocast():
            log_probs, sizes, values = self.network.rl_forward(states, valid_ids, value=True, act=True)

        idxs = []
        for vals in log_probs.split(sizes):
            if method == 'softmax':
                idxs.append(torch.multinomial(vals.exp(), num_samples=1).item())
            elif method == 'greedy':
                idxs.append(vals.argmax(dim=0).item())
            elif method == 'eps':
                explore = random.random() < eps
                idxs.append(random.randrange(len(vals)) if explore else vals.argmax(dim=0).item())
            else:
                raise ValueError(f'Unknown action selection method: {method}')
        act_strs = [valid[i] for valid, i in zip(valid_acts, idxs)]
        return act_strs, idxs, valid_ids, values.float()

    def get_values(self, states, valid_acts):
        """State values, used to bootstrap the return at the end of a rollout."""
        valid_ids = self.encode_valids(valid_acts)
        with torch.no_grad(), self.autocast():
            _, _, values = self.network.rl_forward(states, valid_ids, value=True, act=True)
        return values.float()

    # ---------------------------------------------------------------- learning

    def make_micro_batches(self, samples):
        """Groups samples of similar length so that padding wastes little GPU work."""
        order = sorted(samples, key=lambda s: len(s[0].obs), reverse=True)
        batches, current = [], []
        max_state, max_act, n_acts = 0, 0, 0
        for sample in order:
            state, valid_ids = sample[0], sample[2]
            new_state = max(max_state, len(state.obs))
            new_act = max(max_act, max(len(ids) for ids in valid_ids))
            new_n_acts = n_acts + len(valid_ids)
            cost = (len(current) + 1) * new_state + new_n_acts * new_act
            if current and cost > self.token_budget:
                batches.append(current)
                current, new_state = [], len(state.obs)
                new_act, new_n_acts = max(len(ids) for ids in valid_ids), len(valid_ids)
            current.append(sample)
            max_state, max_act, n_acts = new_state, new_act, new_n_acts
        if current:
            batches.append(current)
        return batches

    def backward_all(self, samples):
        """Accumulates the gradient of the A2C loss over the micro-batches."""
        n_samples = len(samples)
        n_actions = sum(len(s[2]) for s in samples)
        totals = defaultdict(float)
        for batch in self.make_micro_batches(samples):
            states = [s[0] for s in batch]
            act_idxs = [s[1] for s in batch]
            valid_ids = [s[2] for s in batch]
            returns = torch.stack([s[3] for s in batch]).to(self.device)

            with self.autocast():
                log_probs, sizes, values = self.network.rl_forward(states, valid_ids, value=True)
                values = values.float()
                chosen = torch.stack([
                    vals[idx] for vals, idx in zip(log_probs.split(sizes), act_idxs)
                ])
                adv = returns - values

                losses = {
                    'loss_pg': -(chosen * adv.detach()).sum() / n_samples,
                    'loss_td': adv.pow(2).sum() / n_samples,
                    'loss_il': -log_probs.sum() / n_actions,
                    'loss_en': (log_probs * log_probs.exp()).sum() / n_actions,
                }
                loss = sum(self.w[k] * v for k, v in losses.items())

            self.scaler.scale(loss).backward()
            for k, v in losses.items():
                totals[k] += v.item()
            totals['loss'] += loss.item()
            totals['returns'] += returns.sum().item() / n_samples
            totals['advs'] += adv.detach().sum().item() / n_samples
        return totals

    def update(self, transitions, last_values):
        """One A2C update from a rollout of `transitions`."""
        returns = discounted_returns(transitions, last_values, self.gamma)
        samples = [
            (state, act_idx, valid, returns[t][i])
            for t, tr in enumerate(transitions)
            for i, (state, act_idx, valid) in enumerate(zip(tr.states, tr.act_idxs, tr.valid_ids))
        ]

        while True:
            try:
                self.optimizer.zero_grad(set_to_none=True)
                stats = self.backward_all(samples)
                break
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower() or self.token_budget <= MIN_TOKEN_BUDGET:
                    raise
                self.token_budget = max(self.token_budget // 2, MIN_TOKEN_BUDGET)
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f'GPU out of memory: token budget lowered to {self.token_budget}', flush=True)

        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        stats['grad_norm'] = float(grad_norm)
        stats['token_budget'] = self.token_budget
        self.adapt_token_budget(stats)
        return stats

    def adapt_token_budget(self, stats):
        """Raises the micro-batch size until the GPU memory is well used."""
        if self.device.type != 'cuda':
            return
        total = torch.cuda.get_device_properties(self.device).total_memory
        share = torch.cuda.max_memory_allocated(self.device) / total
        stats['gpu_mem_share'] = share
        torch.cuda.reset_peak_memory_stats(self.device)
        if share < TARGET_GPU_MEMORY_SHARE:
            self.token_budget = int(self.token_budget * 1.1)

    # ------------------------------------------------------------- checkpoints

    def save_checkpoint(self, name, **extra):
        path = os.path.join(self.save_path, name)
        tmp_path = path + '.tmp'
        torch.save({
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'token_budget': self.token_budget,
            **extra,
        }, tmp_path)
        os.replace(tmp_path, path)  # never leaves a half-written checkpoint behind
        return path

    def save_weights(self, name):
        path = os.path.join(self.save_path, name)
        torch.save(self.network.state_dict(), path)
        return path

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.network.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.token_budget = checkpoint['token_budget']
        return checkpoint
