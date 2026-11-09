import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from .modules import BiAttention, get_aggregated

VOCAB_SIZE = 30526  # bert-base-uncased plus the four button tokens
HIDDEN_SIZE = 768


class BertConfigForWebshop(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        pretrained_bert=True,
        image=False,
        bert_layers=12,  # only used when `pretrained_bert` is False
        **kwargs
    ):
        self.pretrained_bert = pretrained_bert
        self.image = image
        self.bert_layers = bert_layers
        super().__init__(**kwargs)


def pad_token_ids(sequences, device):
    """Pads token id lists into a batch; returns (ids, attention_mask)."""
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    ids = nn.utils.rnn.pad_sequence(tensors, batch_first=True).to(device, non_blocking=True)
    return ids, (ids > 0).long()


class BertModelForWebshop(PreTrainedModel):

    config_class = BertConfigForWebshop

    def __init__(self, config):
        super().__init__(config)
        if config.pretrained_bert:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            # BertConfig's defaults are the bert-base-uncased architecture.
            self.bert = BertModel(BertConfig(num_hidden_layers=config.bert_layers))
        self.bert.resize_token_embeddings(VOCAB_SIZE)
        self.attn = BiAttention(HIDDEN_SIZE, 0.0)
        self.linear_1 = nn.Linear(HIDDEN_SIZE * 4, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(HIDDEN_SIZE, 1)
        if config.image:
            self.image_linear = nn.Linear(512, HIDDEN_SIZE)
        else:
            self.image_linear = None

        # for state value prediction, used in RL
        self.linear_3 = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 1),
            )

    def score_actions(self, state_rep, state_mask, action_ids, action_mask, sizes):
        """One unnormalized score per action, given the encoded states."""
        action_rep = self.bert(action_ids, attention_mask=action_mask)[0]
        counts = torch.as_tensor(sizes, device=state_rep.device)
        state_rep = state_rep.repeat_interleave(counts, dim=0)
        state_mask = state_mask.repeat_interleave(counts, dim=0)
        state_action_rep = self.attn(action_rep, state_rep, state_mask)
        state_action_rep = self.relu(self.linear_1(state_action_rep))
        act_lens = action_mask.sum(1).tolist()
        act_values = get_aggregated(state_action_rep, act_lens, 'mean')
        return self.linear_2(act_values).squeeze(1)

    def add_image(self, state_rep, state_mask, images):
        """Adds the projected image features to the state as one extra token."""
        if images is None or self.image_linear is None:
            return state_rep, state_mask
        images = self.image_linear(images)
        state_rep = torch.cat([images.unsqueeze(1), state_rep], dim=1)
        state_mask = torch.cat([state_mask[:, :1], state_mask], dim=1)
        return state_rep, state_mask

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images=None, labels=None):
        """Imitation learning pass: one log-probability tensor per state."""
        sizes = sizes.tolist()
        state_rep = self.bert(state_input_ids, attention_mask=state_attention_mask)[0]
        state_rep, state_mask = self.add_image(state_rep, state_attention_mask, images)
        act_values = self.score_actions(state_rep, state_mask, action_input_ids, action_attention_mask, sizes)
        logits = [F.log_softmax(_, dim=0) for _ in act_values.split(sizes)]

        loss = None
        if labels is not None:
            loss = - sum([logit[label] for logit, label in zip(logits, labels)]) / len(logits)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def rl_forward(self, state_batch, act_batch, value=False, q=False, act=False):
        """RL pass over a whole batch of states with their valid actions.

        Returns the concatenated log-probabilities of all actions, the number of
        actions per state and, with `value=True`, one value per state. Encoding the
        states once, in one batch, shares the BERT pass between policy and value.
        """
        device = next(self.parameters()).device
        sizes = [len(valid_acts) for valid_acts in act_batch]
        with torch.set_grad_enabled(not act):
            state_ids, state_mask = pad_token_ids([state.obs for state in state_batch], device)
            flat_acts = [ids for valid_acts in act_batch for ids in valid_acts]
            act_ids, act_mask = pad_token_ids(flat_acts, device)

            state_rep = self.bert(state_ids, attention_mask=state_mask)[0]
            values = self.linear_3(state_rep[:, 0]).squeeze(1) if value else None

            images = None
            if self.image_linear is not None:
                images = torch.stack([
                    torch.zeros(512) if state.image_feat is None else state.image_feat
                    for state in state_batch
                ]).to(device)
            state_rep, state_mask = self.add_image(state_rep, state_mask, images)

            act_values = self.score_actions(state_rep, state_mask, act_ids, act_mask, sizes)
            log_probs = torch.cat([F.log_softmax(_.float(), dim=0) for _ in act_values.split(sizes)])

        if value:
            return log_probs, sizes, values
        return log_probs, sizes
