import unittest
import numpy as np
import torch
from transformers import RobertaModel

from modeling.modeling_greaselm import LMGNN, TextKGMessagePassing, RoBERTaGAT, GreaseLM


class GreaseLMTest(unittest.TestCase):

    def setUp(self):
        self.concept_dim = 200
        self.sent_dim = 768
        self.hidden_size = 200
        self.num_hidden_layers = 12

    def test_GreaseLM(self):
        device = torch.device("cuda:0")
        cp_emb = torch.tensor(np.load("data/cpnet/tzw.ent.npy"), dtype=torch.float)
        model = GreaseLM(pretrained_concept_emb=cp_emb).to(device)
        inputs = self.get_greaselm_inputs(device)
        logits, _ = model(**inputs)
        batch_size = 4
        num_choices = 5
        assert logits.size() == (batch_size, num_choices)

    def test_LMGNN(self):
        device = torch.device("cuda:0")
        cp_emb = torch.tensor(np.load("data/cpnet/tzw.ent.npy"), dtype=torch.float)
        model = LMGNN(pretrained_concept_emb=cp_emb).to(device)
        inputs = self.get_lmgnn_inputs()
        logits, _ = model(**inputs)
        batch_size = 20
        assert logits.size() == (batch_size, 1)

    def test_TextKGMessagePassing(self):
        device = torch.device("cuda:0")
        conf = dict(k=5, n_ntype=4, n_etype=38,
                    n_concept=799273, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                    fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                    pretrained_concept_emb=None, freeze_ent_emb=True,
                    init_range=0.02, ie_dim=200, info_exchange=True, ie_layer_num=1, sep_ie_layers=False,
                    layer_id=-1)
        model, loading_info = TextKGMessagePassing.from_pretrained("roberta-large",
                                                                   output_hidden_states=True,
                                                                   output_loading_info=True, args={}, k=conf["k"],
                                                                   n_ntype=conf["n_ntype"], n_etype=conf["n_etype"],
                                                                   dropout=conf["p_gnn"],
                                                                   concept_dim=conf["concept_dim"],
                                                                   ie_dim=conf["ie_dim"], p_fc=conf["p_fc"],
                                                                   info_exchange=conf["info_exchange"],
                                                                   ie_layer_num=conf["ie_layer_num"],
                                                                   sep_ie_layers=conf["sep_ie_layers"])
        _ = model.to(device)
        inputs = self.get_textkg_inputs()
        outputs, gnn_output = model(*inputs)
        bs = 20
        seq_len = 100
        assert outputs[0].size() == (bs, seq_len, 1024)
        n_node = 200
        assert gnn_output.size() == (bs, n_node, self.hidden_size)

    def test_RoBERTaGAT(self):
        device = torch.device("cuda:0")
        config, _ = RobertaModel.config_class.from_pretrained(
            "roberta-large",
            cache_dir=None, return_unused_kwargs=True,
            force_download=False,
            output_hidden_states=True
        )
        model = RoBERTaGAT(config, sep_ie_layers=True).to(device)
        inputs = self.get_gat_inputs(device)
        outputs, _X = model(*inputs)
        bs = 20
        seq_len = 100
        assert outputs[0].size() == (bs, seq_len, self.sent_dim)
        n_node = 200
        assert _X.size() == (bs * n_node, self.concept_dim)

    def get_gat_inputs(self, device="cuda:0"):
        bs = 20
        seq_len = 100
        hidden_states = torch.zeros([bs, seq_len, self.sent_dim]).to(device)
        attention_mask = torch.zeros([bs, 1, 1, seq_len]).to(device)
        head_mask = [None] * self.num_hidden_layers

        special_tokens_mask = torch.zeros([bs, seq_len]).to(device)

        n_node = 200
        _X = torch.zeros([bs * n_node, self.concept_dim]).to(device)
        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        _node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        _node_type[:, 0] = 3
        _node_type = _node_type.view(-1)
        _node_feature_extra = torch.zeros([bs * n_node, self.concept_dim]).to(device)
        _special_nodes_mask = torch.zeros([bs, n_node], dtype=torch.bool).to(device)
        return hidden_states, attention_mask, special_tokens_mask, head_mask, _X, \
               edge_index, edge_type, _node_type, _node_feature_extra, _special_nodes_mask

    @staticmethod
    def get_lmgnn_inputs(device="cuda:0"):
        bs = 20
        seq_len = 100
        input_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        token_type_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, seq_len]).to(device)
        output_mask = torch.ones([bs, seq_len]).to(device)

        n_node = 200
        concept_ids = torch.arange(end=n_node).repeat(bs, 1).to(device)
        adj_lengths = torch.zeros([bs], dtype=torch.long).fill_(10).to(device)

        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        adj = (edge_index, edge_type)

        node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        node_type[:, 0] = 3
        node_score = torch.zeros([bs, n_node, 1]).to(device)
        node_score[:, 1] = 180

        special_nodes_mask = torch.zeros([bs, n_node], dtype=torch.long).to(device)

        return dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    output_mask=output_mask, concept_ids=concept_ids, node_type_ids=node_type,
                    node_scores=node_score, adj_lengths=adj_lengths, special_nodes_mask=special_nodes_mask,
                    edge_index=edge_index, edge_type=edge_type)

    def get_textkg_inputs(self, device="cuda:0"):
        bs = 20
        seq_len = 100
        input_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        token_type_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, seq_len]).to(device)
        special_tokens_mask = torch.ones([bs, seq_len]).to(device)

        n_node = 200
        H = torch.zeros([bs, n_node, self.hidden_size]).to(device)
        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        A = (edge_index, edge_type)

        node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        node_type[:, 0] = 3
        node_score = torch.zeros([bs, n_node, 1]).to(device)
        node_score[:, 1] = 180

        special_nodes_mask = torch.zeros([bs, n_node], dtype=torch.long).to(device)

        return input_ids, token_type_ids, attention_mask, special_tokens_mask, H, A, node_type, node_score, special_nodes_mask

    def get_greaselm_inputs(self, device):
        bs = 4
        nc = 5
        seq_len = 100
        input_ids = torch.zeros([bs, nc, seq_len], dtype=torch.long).to(device)
        token_type_ids = torch.zeros([bs, nc, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, nc, seq_len]).to(device)
        output_mask = torch.zeros([bs, nc, seq_len]).to(device)

        n_node = 200
        concept_ids = torch.arange(end=n_node).repeat(bs, nc, 1).to(device)
        adj_lengths = torch.zeros([bs, nc], dtype=torch.long).fill_(10).to(device)

        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)

        edge_index = [[edge_index] * nc] * bs
        edge_type = [[edge_type] * nc] * bs

        node_type = torch.zeros([bs, nc, n_node], dtype=torch.long).to(device)
        node_type[:, :, 0] = 3
        node_score = torch.zeros([bs, nc, n_node, 1]).to(device)
        node_score[:, :, 1] = 180

        special_nodes_mask = torch.zeros([bs, nc, n_node], dtype=torch.long).to(device)
        return dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    special_tokens_mask=output_mask, concept_ids=concept_ids, node_type_ids=node_type,
                    node_scores=node_score, adj_lengths=adj_lengths, special_nodes_mask=special_nodes_mask,
                    edge_index=edge_index, edge_type=edge_type)
