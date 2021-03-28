import  argparse
from    math import inf


def get_parser():
    parser = argparse.ArgumentParser()
    get_model_config(parser)
    get_train_config(parser)
    get_checkpoint_config(parser)
    get_data_config(parser)
    return parser.parse_args()


def get_model_config(parser):
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--share_embed', action='store_true')
    parser.add_argument('--max_src_position', type=int, default=inf)
    parser.add_argument('--max_tgt_position', type=int, default=inf)
    parser.add_argument('--num_layer_decoder', type=int, default=6)
    parser.add_argument('--num_layer_encoder', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--position_method', type=str, default='encoding')
    parser.add_argument('--dropout_embed', type=float, default=0.1)
    parser.add_argument('--dropout_sublayer', type=float, default=0.1)
    parser.add_argument('--normalize_before', action='store_true')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_learning_rate', type=int, default=0)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)


def get_train_config(parser):
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cuda_num', type=str, default='0', nargs='+')
    parser.add_argument('--batch_print_info', type=int, default=500)
    parser.add_argument('--grad_accumulate', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=500)


def get_data_config(parser):
    parser.add_argument('--discard_invalid_data', action='store_true')
    parser.add_argument('--vocab', type=str, default='./')
    parser.add_argument('--src_vocab', type=str, default='./')
    parser.add_argument('--tgt_vocab', type=str, default='./')
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--valid_ref', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--source_file', type=str, default=None)


def get_checkpoint_config(parser):
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--restore_file', type=str, default=None)
    parser.add_argument('--checkpoint_num', type=int, default=inf)
    parser.add_argument('--checkpoint_step', type=int, default=1000)
    

def get_generate_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cuda_num', type=str, default='0', nargs='+')
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--raw_file', type=str, default=None)
    parser.add_argument('--ref_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=inf)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--src_vocab', type=str, default=None)
    parser.add_argument('--tgt_vocab', type=str, default=None)
    parser.add_argument('--vocab', type=str, default=None)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--decode_method', type=str, default='greedy')
    parser.add_argument('--beam', type=int, default=5)
    return parser.parse_args()
