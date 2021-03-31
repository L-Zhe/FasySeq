import  torch
from    utils.tools import load_vocab
from    utils.trainer import trainer
from    utils.args import get_parser
from    os import environ
import  importlib
import  random
import  warnings
import  socket
from    contextlib import closing

warnings.filterwarnings("ignore")


def get_vocab_info(args, share_embed):
    if share_embed:
        _, tgt_index2word = load_vocab(args.vocab)
        vocab_size = len(tgt_index2word)
        src_vocab_size = vocab_size
        tgt_vocab_size = vocab_size
    else:
        src_word2index, _ = load_vocab(args.src_vocab)
        _, tgt_index2word = load_vocab(args.tgt_vocab)
        src_vocab_size = len(src_word2index)
        tgt_vocab_size = len(tgt_index2word)
        vocab_size = 0
    setattr(args, 'vocab_size', vocab_size)
    setattr(args, 'src_vocab_size', src_vocab_size)
    setattr(args, 'tgt_vocab_size', tgt_vocab_size)


def set_cuda(cuda, cuda_num):
    if cuda:
        environ['CUDA_VISIBLE_DEVICES'] = ','.join(cuda_num)
    else:
        environ['CUDA_VISIBLE_DEVICES'] = ''


def get_free_port():
    """ Get free port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s: 
        s.bind(('', 0)) 
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        return str(s.getsockname()[1])


def train():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    args = get_parser()
    get_vocab_info(args=args, share_embed=args.share_embed)
    set_cuda(cuda=args.cuda, cuda_num=args.cuda_num)

    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = get_free_port()
    mp = importlib.import_module('torch.multiprocessing')
    seed = random.randint(0, 2048)
    setattr(args, 'world_size', len(args.cuda_num))
    mp.spawn(trainer, nprocs=args.world_size, args=(args, seed))


if __name__ == '__main__':
    train()
