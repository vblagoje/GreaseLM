import argparse
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from modeling import modeling_greaselm
from utils import graph_loader
from utils import parser_utils
from utils import utils

DECODER_DEFAULT_LR = {
    'mcsqa': 1e-3,
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

import numpy as np

logger = logging.getLogger(__name__)


def construct_model(args, kg):
    ########################################################
    #   Load pretrained concept embeddings
    ########################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)

    concept_num, concept_in_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    if args.random_ent_emb:
        cp_emb = None
        freeze_ent_emb = False
        concept_in_dim = args.gnn_dim
    else:
        freeze_ent_emb = args.freeze_ent_emb

    ##########################################################
    #   Build model
    ##########################################################

    if kg == "cpnet":
        n_ntype = 4
        n_etype = 38
    elif kg == "ddb":
        n_ntype = 4
        n_etype = 34
    else:
        raise ValueError("Invalid KG.")
    if args.cxt_node_connects_all:
        n_etype += 2
    model = modeling_greaselm.GreaseLM(args, args.encoder, k=args.k, n_ntype=n_ntype, n_etype=n_etype,
                                       n_concept=concept_num,
                                       concept_dim=args.gnn_dim,
                                       concept_in_dim=concept_in_dim,
                                       n_attention_head=args.att_head_num, fc_dim=args.fc_dim,
                                       n_fc_layer=args.fc_layer_num,
                                       p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                                       pretrained_concept_emb=cp_emb, freeze_ent_emb=freeze_ent_emb,
                                       init_range=args.init_range, ie_dim=args.ie_dim, info_exchange=args.info_exchange,
                                       ie_layer_num=args.ie_layer_num, sep_ie_layers=args.sep_ie_layers,
                                       layer_id=args.encoder_layer)
    return model


def calc_loss_and_acc(logits, labels, loss_type, loss_func):
    bs = labels.size(0)

    if loss_type == 'margin_rank':
        num_choice = logits.size(1)
        flat_logits = logits.view(-1)
        correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1,
                                                                                        num_choice - 1).contiguous().view(
            -1)  # of length batch_size*(num_choice-1)
        wrong_logits = flat_logits[correct_mask == 0]
        y = wrong_logits.new_ones((wrong_logits.size(0),))
        loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
    elif loss_type == 'cross_entropy':
        loss = loss_func(logits, labels)
    loss *= bs

    n_corrects = (logits.argmax(1) == labels).sum().item()

    return loss, n_corrects


def calc_eval_accuracy(eval_set, model, loss_type, loss_func):
    total_loss_acm = 0.0
    n_samples_acm = n_corrects_acm = 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set, desc="Dev/Test batch"):
            bs = labels.size(0)
            logits, _ = model(*input_data)
            loss, n_corrects = calc_loss_and_acc(logits, labels, loss_type, loss_func)
            total_loss_acm += loss.item()
            n_corrects_acm += n_corrects
            n_samples_acm += bs
    return total_loss_acm / n_samples_acm, n_corrects_acm / n_samples_acm


def evaluate(args, devices, kg):
    assert args.load_model_path is not None
    load_model_path = args.load_model_path
    print("loading from checkpoint: {}".format(load_model_path))
    checkpoint = torch.load(load_model_path, map_location='cpu')

    print(checkpoint["config"])

    args = utils.import_config(checkpoint["config"], args)
    dataset = graph_loader.GraphLoader(url="http://localhost:8080",
                                       batch_size=2,
                                       device=devices,
                                       model_name="roberta-large")

    csqa = {"answerKey": "A", "id": "1afa02df02c908a558b4036e80242fac",
            "question": {"question_concept": "revolving door",
                         "choices": [{"label": "A", "text": "bank"}, {"label": "B", "text": "library"},
                                     {"label": "C", "text": "department store"}, {"label": "D", "text": "mall"},
                                     {"label": "E", "text": "new york"}],
                         "stem": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"}}

    input_example = dataset.resolve_csqa(csqa)
    model = construct_model(args, kg)
    model.lmgnn.mp.resize_token_embeddings(len(dataset.tokenizer))

    model.load_state_dict(checkpoint["model"], strict=False)

    model.to(devices[1])
    model.lmgnn.concept_emb.to(devices[0])
    model.eval()

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    dev_total_loss, dev_acc = calc_eval_accuracy(input_example, model, "cross_entropy", loss_func)

    print('dev_acc {:7.4f}'.format(dev_acc))


def get_devices(use_cuda):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""
    if torch.cuda.device_count() >= 2 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        print("device0: {}, device1: {}".format(device0, device1))
    elif torch.cuda.device_count() == 1 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    return device0, device1


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARNING)

    devices = get_devices(args.cuda)
    evaluate(args, devices, kg="cpnet")


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    # General
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/greaselm/', help='model output directory')
    parser.add_argument('--save_model', default=True, type=utils.bool_flag,
                        help="Whether to save model checkpoints or not.")
    parser.add_argument('--load_model_path', default=None, help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='show this help message and exit')
    parser.add_argument("--run_name", required=True, type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str,
                        help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")

    # Data
    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/graph/train.graph.adj.pk',
                        help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/graph/dev.graph.adj.pk',
                        help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/graph/test.graph.adj.pk',
                        help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=200, type=int,
                        help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int,
                        help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")

    # Model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='The number of GreaseLM layers')
    parser.add_argument('--att_head_num', default=2, type=int,
                        help='number of attention heads of the final graph nodes\' pooling')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int,
                        help='number of FC hidden units (except for the MInt operators)')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of hidden layers of the final MLP')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True,
                        help='Whether to freeze the entity embedding layer.')
    parser.add_argument('--ie_dim', default=200, type=int, help='number of the hidden units of the MInt operator.')
    parser.add_argument('--info_exchange', default=True, choices=[True, False, "every-other-layer"],
                        type=utils.bool_str_flag,
                        help="Whether we have the MInt operator in every GreaseLM layer or every other GreaseLM layer or not at all.")
    parser.add_argument('--ie_layer_num', default=1, type=int, help='number of hidden layers in the MInt operator')
    parser.add_argument("--sep_ie_layers", default=False, type=utils.bool_flag,
                        help="Whether to share parameters across the MInt ops across differernt GreaseLM layers or not. Setting it to `False` means sharing.")
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True,
                        help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag,
                        help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")

    # Regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # Optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float,
                        help='Learning rate of parameters not in LM')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int,
                        help="Number of the first few epochs in which LM’s parameters are kept frozen.")
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')

    args = parser.parse_args()
    main(args)
