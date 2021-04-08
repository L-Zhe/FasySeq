import  os
import  pickle
import  torch
import  socket
import  torch.distributed as dist
from    torch import FloatTensor
from    contextlib import closing


def save_vocab(word2index, index2word, lower, save_path):
    vocab = {'word2idx':  word2index,
             'idx2word':  index2word,
             'lower':     lower}
    
    file_path = os.path.join(*os.path.split(save_path)[:-1])
    if os.path.exists(file_path) == False:
        os.makedirs(file_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('===> Save Vocabulary Successfully.')

def load_vocab(save_path):
    with open(save_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab['word2idx'], vocab['idx2word'], vocab['lower']

def show_info(args):
    if args.rank == 0:
        print('+' * 22)
        print('EPOCH: \t\t%d' % args.epoch)
        if args.share_embed:
            print('Share Embed:\tTrue')
            print('Vocab Size: \t%d' % args.vocab_size)
            print('max_position: \t%d' % max(args.max_src_position, args.max_tgt_position))
        else:
            print('SRC Vocab Size: %d' % args.src_vocab_size)
            print('TGT Vocab Size: %d' % args.tgt_vocab_size)
            print('max_src_position: %d' % args.max_src_position)
            print('max_tgt_position: %d' % args.max_tgt_position)
        print('USE_CUDA:\t', args.cuda)
        print("Let's use", torch.cuda.device_count(), "GPUs!")


def save2file(data, file):
    path = os.path.join(*os.path.split(file)[:-1])
    
    if not os.path.exists(path):
        os.makedirs(path)
    flag = False
    with open(file, 'w') as f:
        for line in data:
            if flag:
                f.write('\n')
            flag = True
            f.write(line)


def get_free_port():
    """ Get free port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s: 
        s.bind(('', 0)) 
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        return str(s.getsockname()[1])


def sync_between_gpu(val):
    val = FloatTensor([val]).cuda(non_blocking=True)
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    return val.item()


def move2cuda(data):
    if data.dim() >= 3:
        data = data[0]
    return data.cuda(non_blocking=True)


def triu_mask(length):
    mask = torch.ones(length, length).triu(1)
    return mask.unsqueeze(0) == 1


def pad_mask(input, PAD):
    return (input == PAD).unsqueeze(1)