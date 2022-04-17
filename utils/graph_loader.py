import itertools
import json
from typing import Dict, List, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, BatchEncoding)
from transformers import (BertTokenizer, XLNetTokenizer, RobertaTokenizer)

from preprocess_utils.convert_csqa import create_hypothesis, create_output_dict, get_fitb_from_question
from utils.conceptnet_client import ConceptNetClient
from utils_base import KGEncoding

try:
    from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    from transformers import AlbertTokenizer
except:
    pass

from preprocess_utils import conceptnet

MODEL_CLASS_TO_NAME = {
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
}
try:
    MODEL_CLASS_TO_NAME['albert'] = list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
except:
    pass

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for
                       model_name in model_name_list}


class MultiGPUSparseAdjDataBatchGenerator(object):
    """A data generator that batches the data and moves them to the corresponding devices."""

    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_tensors1[0] = batch_tensors1[0].to(self.device0)
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            edge_index_all, edge_type_all = self.adj_data
            # edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            # edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            yield tuple(
                [batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index,
                 edge_type])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


class GraphLoader(object):

    def __init__(self, url, batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 cxt_node_connects_all=False, kg="cpnet"):
        super().__init__()
        self.conceptnet_client = ConceptNetClient(url)
        self.batch_size = batch_size
        self.device0, self.device1 = device
        self.model_name = model_name
        self.max_node_num = max_node_num
        self.debug_sample_size = 32
        self.max_seq_length = max_seq_length
        self.cxt_node_connects_all = cxt_node_connects_all

        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.load_resources(kg)
        self.debug = False

    def resolve_csqa(self, common_sense_qa_example: Dict[str, str]):
        # Load data
        entailed_qa = self.convert_qajson_to_entailment(common_sense_qa_example)
        qids, labels, encoder_data, concepts_by_sents_list = self.load_input_tensors([entailed_qa],
                                                                                     self.max_seq_length)

        # Load adj data
        response = self.conceptnet_client.resolve_csqa(common_sense_qa_example=common_sense_qa_example)
        num_choices = encoder_data[0].size(1)
        assert num_choices > 0
        kg_encoding: Dict[str, Any] = self.load_sparse_adj_data_with_contextnode(response["result"],
                                                                                 self.max_node_num,
                                                                                 concepts_by_sents_list,
                                                                                 num_choices)

        lm_encoding = dict(input_ids=encoder_data[0], attention_mask=encoder_data[1],
                           token_type_ids=encoder_data[2], special_tokens_mask=encoder_data[3])
        encoding = KGEncoding(data={**lm_encoding, **kg_encoding})
        return qids, labels, encoding

    def convert_qajson_to_entailment(self, qa_json: Dict[str, str], ans_pos: bool = False):
        question_text = qa_json["question"]["stem"]
        choices = qa_json["question"]["choices"]
        for choice in choices:
            choice_text = choice["text"]
            pos = None
            if not ans_pos:
                statement = create_hypothesis(get_fitb_from_question(question_text), choice_text, ans_pos)
            else:
                statement, pos = create_hypothesis(get_fitb_from_question(question_text), choice_text, ans_pos)
            create_output_dict(qa_json, statement, choice["label"] == qa_json.get("answerKey", "A"), ans_pos, pos)

        return qa_json

    def load_resources(self, kg):
        # Load the tokenizer
        try:
            tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer,
                               'albert': AlbertTokenizer}.get(self.model_type)
        except:
            tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer}.get(
                self.model_type)
        tokenizer = tokenizer_class.from_pretrained(self.model_name)
        self.tokenizer = tokenizer

        if kg == "cpnet":
            # Load cpnet
            cpnet_vocab_path = "data/cpnet/concept.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = conceptnet.merged_relations
        elif kg == "ddb":
            cpnet_vocab_path = "data/ddb/vocab.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = [
                'belongstothecategoryof',
                'isacategory',
                'maycause',
                'isasubtypeof',
                'isariskfactorof',
                'isassociatedwith',
                'maycontraindicate',
                'interactswith',
                'belongstothedrugfamilyof',
                'child-parent',
                'isavectorfor',
                'mabeallelicwith',
                'seealso',
                'isaningradientof',
                'mabeindicatedby'
            ]
        else:
            raise ValueError("Invalid value for kg.")

    def load_input_tensors(self, input_jsonl_path, max_seq_length):
        """Construct input tensors for the LM component of the model."""
        input_tensors = load_bert_xlnet_roberta_input_tensors(input_jsonl_path, max_seq_length,
                                                              self.debug, self.tokenizer,
                                                              self.debug_sample_size)
        return input_tensors

    def load_sparse_adj_data_with_contextnode(self, adj_concept_pairs, max_node_num,
                                              concepts_by_sents_list, num_choices) -> Dict[str, Any]:
        """Construct input tensors for the GNN component of the model."""
        # Set special nodes and links
        context_node = 0
        n_special_nodes = 1
        cxt2qlinked_rel = 0
        cxt2alinked_rel = 1
        half_n_rel = len(self.id2relation) + 2
        if self.cxt_node_connects_all:
            cxt2other_rel = half_n_rel
            half_n_rel += 1

        n_samples = len(adj_concept_pairs)  # this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)  # default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
        special_nodes_mask = torch.zeros(n_samples, max_node_num, dtype=torch.bool)

        adj_lengths_ori = adj_lengths.clone()
        if not concepts_by_sents_list:
            concepts_by_sents_list = itertools.repeat(None)
        for idx, (_data, cpts_by_sents) in tqdm(enumerate(zip(adj_concept_pairs, concepts_by_sents_list)),
                                                total=n_samples, desc='loading adj matrices'):
            if self.debug and idx >= self.debug_sample_size * self.num_choice:
                break
            adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], \
                                               _data['cid2score']

            assert n_special_nodes <= max_node_num
            special_nodes_mask[idx, :n_special_nodes] = 1
            num_concept = min(len(concepts) + n_special_nodes,
                              max_node_num)  # this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            # Prepare nodes
            concepts = concepts[:num_concept - n_special_nodes]
            concept_ids[idx, n_special_nodes:num_concept] = torch.tensor(
                concepts + 1)  # To accomodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = context_node  # this is the "concept_id" for contextnode

            # Prepare node scores
            if cid2score is not None:
                if -1 not in cid2score:
                    cid2score[-1] = 0
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1  # Now context node is -1
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            # Prepare node types
            node_type_ids[idx, 0] = 3  # context node
            node_type_ids[idx, 1:n_special_nodes] = 4  # sent nodes
            node_type_ids[idx, n_special_nodes:num_concept][
                torch.tensor(qm, dtype=torch.bool)[:num_concept - n_special_nodes]] = 0
            node_type_ids[idx, n_special_nodes:num_concept][
                torch.tensor(am, dtype=torch.bool)[:num_concept - n_special_nodes]] = 1

            # Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            assert len(self.id2relation) == adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            # Prepare edges
            i += 2
            j += 1
            k += 1  # **** increment coordinate by 1, rel_id by 2 ****
            extra_i, extra_j, extra_k = [], [], []
            for _coord, q_tf in enumerate(qm):
                _new_coord = _coord + n_special_nodes
                if _new_coord > num_concept:
                    break
                if q_tf:
                    extra_i.append(cxt2qlinked_rel)  # rel from contextnode to question concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # question concept coordinate
                elif self.cxt_node_connects_all:
                    extra_i.append(cxt2other_rel)  # rel from contextnode to other concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # other concept coordinate
            for _coord, a_tf in enumerate(am):
                _new_coord = _coord + n_special_nodes
                if _new_coord > num_concept:
                    break
                if a_tf:
                    extra_i.append(cxt2alinked_rel)  # rel from contextnode to answer concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # answer concept coordinate
                elif self.cxt_node_connects_all:
                    extra_i.append(cxt2other_rel)  # rel from contextnode to other concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # other concept coordinate

            # half_n_rel += 2 #should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)
            ########################

            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j),
                                                                                         0)  # add inverse relations
            edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
            edge_type.append(i)  # each entry is [E, ]

        edge_index = list(map(list, zip(*(iter(edge_index),) * num_choices)))
        edge_type = list(map(list, zip(*(iter(edge_type),) * num_choices)))

        concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask = [
            x.view(-1, num_choices, *x.size()[1:]) for x in
            (concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask)]

        # concept_ids: (n_questions, num_choice, max_node_num)
        # node_type_ids: (n_questions, num_choice, max_node_num)
        # node_scores: (n_questions, num_choice, max_node_num, 1)
        # adj_lengths: (n_questions,　num_choice)
        # special_nodes_mask: (n_questions, num_choice, max_node_num)

        # edge_index: list of size (n_questions, n_choices), where each entry is tensor[2, E]
        # edge_type: list of size (n_questions, n_choices), where each entry is tensor[E, ]
        # We can't stack edge_index and edge_type lists of tensors as tensors are not of equal size
        return dict(concept_ids=concept_ids, node_type_ids=node_type_ids, node_scores=node_scores,
                    adj_lengths=adj_lengths, special_nodes_mask=special_nodes_mask,
                    edge_index=edge_index, edge_type=edge_type)


def load_bert_xlnet_roberta_input_tensors(entailed_qa_examples, max_seq_length, debug, tokenizer, debug_sample_size):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(qa_entailed_statements: List[Dict[str, Any]]):
        examples = []
        for json_dic in qa_entailed_statements:
            label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
            contexts = json_dic["question"]["stem"]
            if "para" in json_dic:
                contexts = json_dic["para"] + " " + contexts
            if "fact1" in json_dic:
                contexts = json_dic["fact1"] + " " + contexts
            examples.append(
                InputExample(
                    example_id=json_dic["id"],
                    contexts=[contexts] * len(json_dic["question"]["choices"]),
                    question="",
                    endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                    label=label
                ))

        return examples

    def simple_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        concepts_by_sents_list = []
        for ex_index, example in tqdm(enumerate(examples), total=len(examples), desc="Converting examples to features"):
            if debug and ex_index >= debug_sample_size:
                break
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                ans = example.question + " " + ending

                encoded_input = tokenizer(context, ans, padding="max_length", truncation=True,
                                          max_length=max_seq_length, return_token_type_ids=True,
                                          return_special_tokens_mask=True)
                input_ids = encoded_input["input_ids"]
                output_mask = encoded_input["special_tokens_mask"]
                input_mask = encoded_input["attention_mask"]
                segment_ids = encoded_input["token_type_ids"]

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append((input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(
                InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features, concepts_by_sents_list

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    examples = read_examples(entailed_qa_examples)
    features, concepts_by_sents_list = simple_convert_examples_to_features(examples,
                                                                           list(range(len(examples[0].endings))),
                                                                           max_seq_length, tokenizer)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return example_ids, all_label, data_tensors, concepts_by_sents_list
