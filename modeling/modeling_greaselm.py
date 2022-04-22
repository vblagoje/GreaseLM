import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertLayer, RobertaConfig
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.modeling_roberta import RobertaEmbeddings, RobertaPooler
from transformers.modeling_utils import PreTrainedModel

from modeling import modeling_gnn


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# Todo: replace with transformers new gelu
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num + 2, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.fill_(0)
                self.emb.weight.data[:concept_num].copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


# TODO: upgrade to the latest version before HF integration
class GreaseLMPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class GreaseLMForMultipleChoice(GreaseLMPreTrainedModel):

    def __init__(self, config, pretrained_concept_emb_file=None):
        super().__init__(config)
        self.greaselm = GreaseLMModel(config, pretrained_concept_emb_file="./greaselm_model/tzw.ent.npy")
        self.pooler = MultiheadAttPoolLayer(config.n_attention_head,
                                            config.hidden_size,
                                            config.concept_dim) if config.k >= 0 else None

        concat_vec_dim = config.concept_dim * 2 + config.hidden_size
        self.fc = MLP(concat_vec_dim, config.fc_dim, 1, config.n_fc_layer, config.p_fc, layer_norm=True)

        self.dropout_fc = nn.Dropout(config.p_fc)
        self.layer_id = config.layer_id
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                special_tokens_mask,
                concept_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                special_nodes_mask,
                edge_index,
                edge_type,
                emb_data=None,
                detail=False,
                cache_output=False):
        """
         :param input_ids:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, seq_len)`):
                    Input ids for the language model.
         :param attention_mask:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, seq_len)`):
                    Attention mask for the language model.
         :param token_type_ids:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, seq_len)`):
                    Token type ids for the language model.
         :param special_tokens_mask:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, seq_len)`):
                    Output mask for the language model.
         :param concept_ids:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num)`):
                    Resolved conceptnet ids.
         :param node_type_ids:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num)`):
                    Conceptnet id types where 0 == question entity; 1 == answer choice entity;
                    2 == other node; 3 == context node
         :param node_scores:
               (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num, 1)`):
                    LM relevancy scores for each resolved conceptnet id.
         :param adj_lengths:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices)`):
                    Adjacency matrix lengths for each batch sample.
         :param special_nodes_mask:
               (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num)`):
                    Mask identifying special nodes in the graph (interaction node in the GreaseLM paper).
         :param edge_index:
                torch.tensor(2, E)) where E is the total number of edges in the particular graph.
         :param edge_type:
                torch.tensor(E, ) where E is the total number of edges in the particular graph.
         :param emb_data:
                torch.tensor(batch_size, number_of_choices, max_node_num, emb_dim)
         :param detail:
                (bool): Whether to return detailed output.
         :param cache_output:
                Whether to cache the output of the language model.
        """
        bs, nc = input_ids.shape[0:2]

        # Merged core
        outputs, gnn_output = self.greaselm(input_ids, attention_mask, token_type_ids, special_tokens_mask, concept_ids,
                                            edge_index, edge_type, node_type_ids, node_scores, adj_lengths,
                                            special_nodes_mask)
        # outputs: ([bs, seq_len, sent_dim], [bs, sent_dim], ([bs, seq_len, sent_dim] for _ in range(25)))
        # gnn_output: [bs, n_node, dim_node]

        # LM outputs
        all_hidden_states = outputs[-1] # ([bs, seq_len, sent_dim] for _ in range(25))
        hidden_states = all_hidden_states[self.layer_id] # [bs, seq_len, sent_dim]

        sent_vecs = self.greaselm.pooler(hidden_states) # [bs, sent_dim]

        # GNN outputs
        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out

        mask = mask | (node_type_ids == 3) # pool over all KG nodes (excluding the context node)
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        graph_vecs, pool_attn = self.pooler(sent_vecs, gnn_output, mask)
        # graph_vecs: [bs, node_dim]

        concat = torch.cat((graph_vecs, sent_vecs, Z_vecs), 1)
        logits = self.fc(self.dropout_fc(concat))

        logits = logits.view(bs, nc)
        return MultipleChoiceModelOutput(
            logits=logits,
            attentions=pool_attn,
        )


class GreaseLMModel(GreaseLMPreTrainedModel):

    def __init__(self, config, pretrained_concept_emb_file, freeze_ent_emb=True, add_pooling_layer=True, dropout=0.2):
        super().__init__(config)
        self.config = config

        pretrained_concept_emb = torch.tensor(np.load(pretrained_concept_emb_file), dtype=torch.float)
        concept_num, concept_in_dim = pretrained_concept_emb.size(0), pretrained_concept_emb.size(1)
        self.hidden_size = config.concept_dim
        self.emb_node_type = nn.Linear(config.n_ntype, config.concept_dim // 2)

        self.basis_f = 'sin'  # ['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, config.concept_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, config.concept_dim // 2)
            self.emb_score = nn.Linear(config.concept_dim // 2, config.concept_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(config.concept_dim // 2, config.concept_dim // 2)

        self.Vh = nn.Linear(config.concept_dim, config.concept_dim)
        self.Vx = nn.Linear(config.concept_dim, config.concept_dim)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_e = nn.Dropout(dropout)
        self.cpnet_vocab_size = config.n_concept
        self.concept_emb = CustomizedEmbedding(concept_num=config.n_concept, concept_out_dim=config.concept_dim,
                                               use_contextualized=False, concept_in_dim=config.concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb,
                                               freeze_ent_emb=freeze_ent_emb) if config.k >= 0 else None
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = GreaseLMEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        """
        edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        """
        def flatten(iterable):
            for item in iterable:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item

        edge_index_init = list(flatten(edge_index_init))
        edge_type_init = list(flatten(edge_type_init))
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E,]
        return edge_index, edge_type

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                special_tokens_mask,
                concept_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                special_nodes_mask,
                edge_index,
                edge_type,
                position_ids=None,
                head_mask=None,
                emb_data=None,
                cache_output=False,
                output_hidden_states=True):
        """
           :param input_ids:
                 (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_len)`):
                      Input ids for the language model.
           :param attention_mask:
                 (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_len)`):
                      Attention mask for the language model.
           :param token_type_ids:
                 (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_len)`):
                      Token type ids for the language model.
           :param special_tokens_mask:
                 (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, seq_len)`):
                      Output mask for the language model.
           :param concept_ids:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num)`):
                    Resolved conceptnet ids.
           :param node_type_ids:
                 (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_nodes)`):
                      0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
           :param node_scores:
                  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, number_of_choices, max_node_num, 1)`):
                    LM relevancy scores for each resolved conceptnet id
           :param adj_lengths:
               (:obj:`torch.LongTensor` of shape :obj:`(batch_size, number_of_choices)`):
                    Adjacency matrix lengths for each batch sample.
           :param special_nodes_mask:
                 (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, number_of_nodes)`):
                    Mask identifying special nodes in the graph (interaction node in the GreaseLM paper).
           :param edge_index:
                torch.tensor(2, E)) where E is the total number of edges in the particular graph.
           :param edge_type:
                torch.tensor(E, ) where E is the total number of edges in the particular graph.
           :param position_ids:
                (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_len)`, `optional`, defaults to :obj:`None`):
                    Indices of positions of each input sequence tokens in the position embeddings.
           :param head_mask:
                    list of shape [num_hidden_layers]
           :param emb_data:
                torch.tensor(batch_size, number_of_choices, max_node_num, emb_dim)
           :param cache_output:
                    Whether to cache the output of the language model.
           :param output_hidden_states: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                    If set to ``True``, the model will return all hidden-states.

        """
        def merge_first_two_dim(x):
            return x.view(-1, *(x.size()[2:]))

        input_ids, attention_mask, token_type_ids, special_tokens_mask, concept_ids, \
        node_type_ids, node_scores, adj_lengths, special_nodes_mask = [
            merge_first_two_dim(t) for t in [input_ids, attention_mask,
                                             token_type_ids,
                                             special_tokens_mask,
                                             concept_ids, node_type_ids,
                                             node_scores, adj_lengths,
                                             special_nodes_mask]]
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        node_scores = torch.zeros_like(node_scores)

        # LM inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 1D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError("Attnetion mask should be either 1D or 2D.")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        # GNN inputs
        concept_ids[concept_ids == 0] = self.cpnet_vocab_size + 2
        gnn_input = self.concept_emb(concept_ids - 1, emb_data).to(node_scores.device)
        gnn_input[:, 0] = 0
        # H - node features from the previous layer
        H = self.dropout_e(gnn_input)  # (batch_size, n_node, dim_node)

        # Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(
            1)).float()  # 0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :]  # [batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2)  # [batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  # [batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05)  # [batch_size, n_node]
        node_score = node_scores.unsqueeze(2)  # [batch_size, n_node, 1]

        _batch_size, _n_nodes = node_type_ids.size()

        #Embed type
        T = modeling_gnn.make_one_hot(node_type_ids.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type_ids.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type_ids.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        # Merged core
        encoder_outputs, _X = self.encoder(embedding_output,
                                           extended_attention_mask, special_tokens_mask, head_mask, _X, edge_index,
                                           edge_type, _node_type, _node_feature_extra, special_nodes_mask,
                                           output_hidden_states=output_hidden_states)

        # LM outputs
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        # GNN outputs
        X = _X.view(node_type_ids.size(0), node_type_ids.size(1), -1) #[batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return outputs, output


class GreaseLMEncoder(nn.Module):

    def __init__(self, config, dropout=0.2):
        super().__init__()
        self.config = config
        self.k = config.k
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(config.n_etype + 1 + config.n_ntype * 2, config.gnn_hidden_size),
            torch.nn.BatchNorm1d(config.gnn_hidden_size), torch.nn.ReLU(),
            torch.nn.Linear(config.gnn_hidden_size, config.gnn_hidden_size))

        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gnn_layers = nn.ModuleList(
            [modeling_gnn.GATConvE(config.gnn_hidden_size, config.n_ntype, config.n_etype, self.edge_encoder) for _ in
             range(config.k)])
        self.activation = GELU()
        self.dropout_rate = dropout

        self.sep_ie_layers = config.sep_ie_layers
        if self.sep_ie_layers:
            self.ie_layers = nn.ModuleList(
                [MLP(config.hidden_size + config.concept_dim, config.ie_dim,
                     config.hidden_size + config.concept_dim, config.ie_layer_num, config.p_fc) for _ in
                 range(config.k)])
        else:
            ie_layer_size = config.hidden_size + config.concept_dim
            self.ie_layer = MLP(ie_layer_size, config.ie_dim, ie_layer_size, config.ie_layer_num, config.p_fc)

        self.num_hidden_layers = config.num_hidden_layers
        self.info_exchange = config.info_exchange

    def forward(self, hidden_states, attention_mask, special_tokens_mask, head_mask, _X, edge_index, edge_type,
                _node_type, _node_feature_extra, special_nodes_mask, output_attentions=False,
                output_hidden_states=True):

        """
         :param hidden_states:
               (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, sent_dim)`):
         :param attention_mask:
               (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 1, 1, seq_len)`):
         :param special_tokens_mask:
               (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, seq_len)`):
                    Token type ids for the language model.
         :param head_mask: list of shape [num_hidden_layers]
         :param _X:
              (:obj:`torch.FloatTensor` of shape :obj:`(total_n_nodes, node_dim)`):
                `total_n_nodes` = batch_size * num_nodes
         :param edge_index:
               (:obj:`torch.LongTensor` of shape :obj:`(2, E)`):
         :param edge_type:
               (:obj:`torch.LongTensor` of shape :obj:`(E, )`):
         :param _node_type:
               (:obj:`torch.LongTensor` of shape :obj:`(total_n_nodes,)`):
         :param _node_feature_extra:
              (:obj:`torch.FloatTensor` of shape :obj:`(total_n_nodes, node_dim)`):
                `total_n_nodes` = batch_size * num_nodes
         :param special_nodes_mask:
               (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, max_node_num)`):
         :param output_attentions: (:obj:`bool`) Whether or not to return the attentions tensor.
         :param output_hidden_states: (:obj:`bool`) Whether or not to return the hidden states tensor.
        """
        bs = hidden_states.size(0)
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            # LM
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            if i >= self.num_hidden_layers - self.k:
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                _X = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training=self.training)

                # Exchange info between LM and GNN hidden states (Modality interaction)
                if self.info_exchange == True or (
                        self.info_exchange == "every-other-layer" and (i - self.num_hidden_layers + self.k) % 2 == 0):
                    X = _X.view(bs, -1, _X.size(1))  # [bs, max_num_nodes, node_dim]
                    context_node_lm_feats = hidden_states[:, 0, :]  # [bs, sent_dim]
                    context_node_gnn_feats = X[:, 0, :]  # [bs, node_dim]
                    context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)
                    if self.sep_ie_layers:
                        context_node_feats = self.ie_layers[gnn_layer_index](context_node_feats)
                    else:
                        context_node_feats = self.ie_layer(context_node_feats)
                    context_node_lm_feats, context_node_gnn_feats = torch.split(context_node_feats,
                                                                                [context_node_lm_feats.size(1),
                                                                                 context_node_gnn_feats.size(1)], dim=1)
                    hidden_states[:, 0, :] = context_node_lm_feats
                    X[:, 0, :] = context_node_gnn_feats
                    _X = X.view_as(_X)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, _X  # last-layer hidden state, (all hidden states), (all attentions)
