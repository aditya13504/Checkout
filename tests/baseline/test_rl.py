"""
Tests of the RL training pieces. The network is a one-layer BERT so these run quickly on
a CPU; the update, batching and masking code is the same as for the full model.
"""
import sys
from argparse import Namespace
from collections import namedtuple
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baseline_models'))

from agent import Agent, Transition, discounted_returns  # noqa: E402
from env import goal_range  # noqa: E402
from models.bert import BertConfigForWebshop, BertModelForWebshop  # noqa: E402
from models.modules import BiAttention  # noqa: E402
from models.rnn import RCDQN  # noqa: E402
from train_rl import make_policy  # noqa: E402

State = namedtuple('State', ('obs', 'goal', 'image_feat'))


def agent_args(**kwargs):
    defaults = dict(network='bert', bert_path='scratch', get_image=0, embedding_dim=32, hidden_dim=32,
                    output_dir='.', clip=10, w_pg=1, w_td=1, w_il=0, w_en=1, learning_rate=1e-3, gamma=0.9,
                    amp=0, token_budget=0)
    defaults.update(kwargs)
    return Namespace(**defaults)


@pytest.fixture(scope='module')
def tiny_agent():
    original = BertConfigForWebshop.__init__

    def one_layer(self, pretrained_bert=False, image=False, bert_layers=1, **kwargs):
        original(self, pretrained_bert=False, image=image, bert_layers=1, **kwargs)

    BertConfigForWebshop.__init__ = one_layer
    try:
        yield Agent(agent_args())
    finally:
        BertConfigForWebshop.__init__ = original


def test_returns_are_cut_at_episode_ends():
    mk = lambda r, d: Transition(None, None, r, None, d)
    transitions = [mk([0.0, 1.0], [False, True]), mk([2.0, 0.0], [True, False])]
    returns = discounted_returns(transitions, torch.tensor([100.0, 10.0]), gamma=0.5)
    assert returns[1].tolist() == [2.0, 5.0]       # env 0 ended; env 1 bootstraps 0 + 0.5 * 10
    assert returns[0].tolist() == [1.0, 1.0]       # env 0: 0 + 0.5 * 2; env 1 ended at t=0


def test_tokenizer_has_each_button_token(tiny_agent):
    assert len(tiny_agent.tokenizer) == 30526
    ids = tiny_agent.encode_texts(['[button] search [button_]'], 16)[0]
    assert tiny_agent.tokenizer.convert_tokens_to_ids('[button]') in ids
    assert tiny_agent.tokenizer.convert_tokens_to_ids('[button_]') in ids


def test_token_budget_splits_micro_batches(tiny_agent):
    sample = lambda n: (State(list(range(1, n)), [1], None), 0, [[1, 2], [1, 3]], torch.tensor(0.0))
    samples = [sample(100) for _ in range(10)]
    tiny_agent.token_budget = 300
    batches = tiny_agent.make_micro_batches(samples)
    assert sum(len(b) for b in batches) == 10 and len(batches) > 1
    assert all(len(b) * 99 + sum(len(s[2]) for s in b) * 2 <= 300 for b in batches)
    tiny_agent.token_budget = 16000


def test_batched_scores_equal_one_by_one_scores(tiny_agent):
    agent = tiny_agent
    infos = [{'goal': 'g'}, {'goal': 'g'}]
    states = agent.build_states(['a short page', 'a much longer page with many more words on it'], infos)
    valids = [['click[one]', 'click[two]'], ['click[three]', 'click[four]', 'click[a long action name]']]
    ids = agent.encode_valids(valids)
    with torch.no_grad():
        together, sizes, values = agent.network.rl_forward(states, ids, value=True, act=True)
        alone = [agent.network.rl_forward([s], [i], value=True, act=True) for s, i in zip(states, ids)]
    assert sizes == [2, 3]
    assert torch.allclose(together, torch.cat([a[0] for a in alone]), atol=1e-4)
    assert torch.allclose(values, torch.cat([a[2] for a in alone]), atol=1e-4)


def test_rl_policy_has_no_dropout(tiny_agent):
    assert not tiny_agent.network.training


def test_attention_mask_hides_padding():
    attention = BiAttention(4, 0.0)
    context, memory = torch.rand(1, 3, 4), torch.rand(1, 5, 4)
    mask = torch.tensor([[1, 1, 0, 0, 0]])
    out = attention(context, memory, mask)
    changed = memory.clone()
    changed[0, 2:] = 99.0
    assert torch.allclose(out, attention(context, changed, mask), atol=1e-5)
    assert torch.isfinite(out).all()


def test_policy_learns_to_prefer_the_rewarded_action(tiny_agent):
    agent = tiny_agent
    valid = ['click[good]', 'click[bad]']
    states = agent.build_states(['pick one'] * 4, [{'goal': 'g'}] * 4)
    ids = agent.encode_valids([valid] * 4)
    good = lambda: agent.network.rl_forward(states[:1], ids[:1])[0][0].exp().item()
    assert good() < 0.9
    for _ in range(60):
        idxs = [0, 1, 0, 1]
        rewards = [10.0 if i == 0 else 0.0 for i in idxs]
        rollout = [Transition(states, idxs, rewards, ids, [True] * 4)]
        agent.update(rollout, torch.zeros(4))
    assert good() > 0.9


def test_resume_restores_the_training_state(tiny_agent, tmp_path):
    agent = tiny_agent
    agent.save_path = str(tmp_path)
    path = agent.save_checkpoint('ckpt.pt', step=7)
    weights = {k: v.clone() for k, v in agent.network.state_dict().items()}
    with torch.no_grad():
        next(agent.network.parameters()).add_(1.0)
    assert agent.load_checkpoint(path)['step'] == 7
    assert all(torch.equal(v, weights[k]) for k, v in agent.network.state_dict().items())


def test_pretrain_flag_builds_a_random_model_without_downloading():
    model = BertModelForWebshop(BertConfigForWebshop(pretrained_bert=False, bert_layers=1))
    assert len(model.bert.encoder.layer) == 1
    assert model.bert.get_input_embeddings().num_embeddings == 30526


def test_rnn_network_runs_and_learns_a_step():
    agent = Agent(agent_args(network='rnn'))
    states = agent.build_states(['page one', 'page number two'], [{'goal': 'find a'}, {'goal': 'find b c'}])
    valids = [['click[a]', 'click[b]'], ['click[c]', 'click[d]', 'click[e]']]
    act_strs, idxs, ids, values = agent.act(states, valids)
    assert isinstance(agent.network, RCDQN) and len(act_strs) == 2 and values.shape == (2,)
    stats = agent.update([Transition(states, idxs, [1.0, 0.0], ids, [False, True])], torch.zeros(2))
    assert stats['grad_norm'] > 0


def test_search_modes_pick_the_right_query():
    class NoModel:
        def act(self, *args, **kwargs):
            raise AssertionError('a fixed search mode must not ask the model')

    info = {'valid': ['search[predicted one]', 'search[predicted two]', 'search[the instruction]']}
    assert make_policy(NoModel(), 'softmax', search='top1')(['obs'], [info]) == ['search[predicted one]']
    assert make_policy(NoModel(), 'softmax', search='instruction')(['obs'], [info]) == ['search[the instruction]']


def test_goal_splits():
    assert goal_range(12087, 'test') == range(0, 500)
    assert goal_range(12087, 'eval') == range(500, 1500)
    assert goal_range(12087, 'train') == range(1500, 12087)
    n = 60
    parts = [goal_range(n, s) for s in ('test', 'eval', 'train')]
    assert all(len(p) > 0 for p in parts) and sum(len(p) for p in parts) == n
