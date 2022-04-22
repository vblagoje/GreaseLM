import argparse
import logging
import random

import numpy as np
import torch

from modeling.modeling_greaselm import GreaseLMForMultipleChoice
from utils.graph_loader import GraphLoader

from utils import parser_utils
from utils import utils

logger = logging.getLogger(__name__)


def evaluate(devices):
    graph_loader = GraphLoader(url="http://localhost:8080",
                               batch_size=2,
                               device=devices,
                               model_name="roberta-large")

    csqa = {"answerKey": "A", "id": "1afa02df02c908a558b4036e80242fac",
            "question": {"question_concept": "revolving door",
                         "choices": [{"label": "A", "text": "bank"}, {"label": "B", "text": "library"},
                                     {"label": "C", "text": "department store"}, {"label": "D", "text": "mall"},
                                     {"label": "E", "text": "new york"}],
                         "stem": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"}}

    qids, labels, single_batch = graph_loader.resolve_csqa(csqa)
    model = GreaseLMForMultipleChoice.from_pretrained("./greaselm_model/")
    model.resize_token_embeddings(len(graph_loader.tokenizer))

    model.to(devices[0])
    model.eval()

    output: GreaseLMForMultipleChoice = model(**single_batch.to(devices[0]))
    print(output)


def get_devices(use_cuda):
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
    evaluate(devices)


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

    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int,
                        help="Number of the first few epochs in which LM’s parameters are kept frozen.")
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')

    args = parser.parse_args()
    main(args)
