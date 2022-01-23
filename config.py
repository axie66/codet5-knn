import os
import multiprocessing
import random
import numpy as np
import torch

def add_args(parser):
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'conala'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--data_num", default=-1, type=int)
    
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)

    # parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=-1, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=-1, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_bleu", action='store_true', 
                        help="Whether to evaluate bleu on dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")

    parser.add_argument('--wandb', action='store_true', 
                        help='log to wandb')

    parser.add_argument('--calc_stats', action='store_true')
    return parser


def add_knn_args(parser):
    parser.add_argument('--dstore-fp16', default=False, action='store_true',
                        help='if true, datastore items are saved in fp16 and int32')

    # KNN Hyperparameters
    parser.add_argument('--k', default=1024, type=int, 
                        help='number of nearest neighbors to retrieve')
    parser.add_argument('--probe', default=8, type=int,
                            help='for FAISS, the number of lists to query')
    parser.add_argument('--lmbda', default=0.0, type=float,
                        help='controls interpolation with knn, 0.0 = no knn')
    parser.add_argument('--knn_temp', default=1.0, type=float,
                        help='temperature for knn distribution')
    parser.add_argument('--knn-sim-func', default=None, type=str, # don't actually need this one
                        help='similarity function to use for knns')
    parser.add_argument('--use-faiss-only', action='store_true', default=False,
                        help='do not look up the keys/values from a separate array')
    parser.add_argument('--faiss_metric_type', type=str, default='l2',
                        help='distance metric for faiss')

    # Datastore related stuff
    parser.add_argument('--dstore-size', type=int,
                        help='number of items in the knn datastore')
    parser.add_argument('--dstore-filename', type=str, default=None,
                        help='File where the knn datastore is saved')
    parser.add_argument('--indexfile', type=str, default=None,
                        help='File containing the index built using faiss for knn')
    parser.add_argument('--knn_embed_dim', type=int, default=768,
                        help='dimension of keys in knn datastore')
    parser.add_argument('--no-load-keys', default=False, action='store_true',
                        help='do not load keys')
    parser.add_argument('--knn-q2gpu', action='store_true', default=False,
                        help='move the quantizer from faiss to gpu')
    
    # probably shouldn't use this flag (except for small datastores)
    parser.add_argument('--move-dstore-to-mem', default=False, action='store_true', 
                        help='move the keys and values for knn to memory')
    return parser


def add_conala_args(parser):
    parser.add_argument('--mono-min-prob', type=float, default=.1)
    return parser

def parse_args(parser):
    args = parser.parse_args()

    if args.task in ['summarize']:
        args.lang = args.sub_task
    elif args.task in ['refine', 'concode', 'clone']:
        args.lang = 'java'
    elif args.task == 'conala':
        args.lang = 'python'
    elif args.task == 'defect':
        args.lang = 'c'
    elif args.task == 'translate':
        args.lang = 'c_sharp' if args.sub_task == 'java-cs' else 'java'

    args.cpu_count = multiprocessing.cpu_count()

    args.train_filename = os.path.join(args.data_dir, 'train.json')
    args.dev_filename = os.path.join(args.data_dir, 'dev.json')
    args.test_filename = os.path.join(args.data_dir, 'test.json')

    for dir_path in (args.summary_dir, args.data_dir, args.res_dir, args.output_dir):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    return args

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
