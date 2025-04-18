import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')
    parser.add_argument('--load_model_path', type=str, default='',
                        help='model path for pretrain or test')   ####
    #parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=42)  ####

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after WordPiece tokenization. "
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default='./bert_data/hoc',
        help="Directory for data storage (default: ./data/hoc_av)."
    )

    parser.add_argument(
        "--vocab_file",
        type=str,
        default='./vocab_extip20.txt',
        help="Directory for vocabulary file."
    )

    parser.add_argument(
        "--num_aspects",
        type=int,
        default=10,
        help="Number of aspects (default: 10)."
    )

    parser.add_argument(
        "--aspect_value_list",
        type=int,
        nargs='*',
        default=[0, 1],
        help="List of aspect values (default: [-2, -1, 0, 1])."
    )

    parser.add_argument('--do_lower_case', action='store_true', help='Whether do lower case')
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')  ##
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)

    parser.add_argument(
        "--model",
        type=str,
        default="bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
        help="Name of model, listed in https://huggingface.co/bionlp"
    )
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model.replace('/','_') + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.load_model_path.replace(ext, '')
    val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
