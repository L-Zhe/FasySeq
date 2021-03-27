import  argparse
from    utils import constants
from    utils.tools import load_vocab
import  os
from    math import inf, ceil
from    torch import LongTensor
import  random
import  pickle
from    tqdm import tqdm

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--src_vocab', type=str)
    parser.add_argument('--tgt_vocab', type=str, default=None)
    parser.add_argument('--save_file', type=str)

    return parser.parse_args()


def data_process(filelist, word2index):
    '''
    Change word to index. 
    '''
    data = []
    for file in filelist:
        with open(file, 'r', encoding='utf-8') as f:
            data.extend([line.strip('\n').lower().split() \
                         for line in f.readlines()])
    def prepare_sequence(seq):
        return list(map(lambda word: word2index[constants.UNK_WORD] 
                        if word2index.get(word) is None else word2index[word], seq))
    return [prepare_sequence(seq) for seq in tqdm(data)]


def process_invalid_date(source, target_input, target_output, args):
    max_src_position = args.max_src_position
    max_tgt_position = args.max_tgt_position
    discard_invalid_data = getattr(args, 'discard_invalid_data', False)
    if max_src_position != inf or max_tgt_position != inf:
        total_len = len(source)
        if discard_invalid_data:
            del_index = []
            for i in range(total_len):
                if len(source[i]) > max_src_position or \
                   (target_input and len(target_input[i]) > max_tgt_position):
                    del_index.insert(0, i)
            if args.rank == 0:
                print("===> Discard invalid data: %d" % len(del_index))
            for index in del_index:
                del source[index]
                if target_input and target_output:
                    del target_input[index]
                    del target_output[index]
        else:
            for i in range(total_len):
                source[i] = source[i][:max_src_position]
                if target_input and target_output:
                    target_input[i] = target_input[i][:max_tgt_position]
                    target_output[i] = target_output[i][:max_tgt_position]


def sort_data_by_len(source, target_input=None, target_output=None):
    if target_input and target_output:
        data = list(zip(source, target_input, target_output))
        data.sort(key=lambda line: max(len(line[0]), len(line[1])), reverse=True)
        return zip(*data)
    else:
        data = [(index, value) for index, value in sorted(list(enumerate(source)), key=lambda x: len(x[1]), reverse=True)]
        return zip(*data)


def _get_tokens(source, target_input, target_output, shuffle, args):
    total_len = len(source)
    if target_input and target_output:
        source, target_input, target_output = sort_data_by_len(source=source,
                                                               target_input=target_input,
                                                               target_output=target_output)
        rank = None
    else:
        rank, source = sort_data_by_len(source)
    index_pair = []
    st = 0
    total_len = len(source)
    while st < total_len:
        if target_input and target_output:
            max_length = max(len(source[st]), len(target_input[st]))
        else:
            max_length = len(source[st])
        ed = min(st + args.max_tokens // max_length, total_len)
        if ed == st:
            ed += 1
        index_pair.append((st, ed))
        st = ed
    data = []
    for (st, ed) in tqdm(index_pair):
        if target_input and target_output:
            data.append((LongTensor(pad_batch(source[st:ed], args.PAD_index)),
                         LongTensor(pad_batch(target_input[st:ed], args.PAD_index)),
                         LongTensor(pad_batch(target_output[st:ed], args.PAD_index))))
        else:
            data.append(LongTensor(pad_batch(source[st:ed], args.PAD_index)))
    if shuffle:
        random.shuffle(data)
    return data, rank


class data_loader:

    def __init__(self, source, target=None,
                 BOS_index=None, EOS_index=None):
        self.source = source
        self.target_input = None
        self.target_output = None
        self.max_src_len = max(len(seq) for seq in self.source)
        self.max_tgt_len = None
        if target:
            self.target_input = []
            self.target_output = []
            for line in target:
                self.target_input.append([BOS_index] + line)
                self.target_output.append(line + [EOS_index])
            self.max_tgt_len = max(len(seq) for seq in self.target_input)

    def restore_rank(self, data):
        rank_data = []
        rank = [(index, value) for index, value in sorted(list(enumerate(self.rank)), key=lambda x: x[1], reverse=False)]
        self.rank, _ = zip(*rank)
        for index in self.rank:
            rank_data.append(data[index])
        return rank_data

    def set_param(self, shuffle, args, seed=None):
        setattr(args, 'max_src_position', min(self.max_src_len, args.max_src_position))
        setattr(args, 'max_tgt_position', min(self.max_tgt_len, args.max_tgt_position))
        if args.rank == 0:
            print("max source sentence length: ", self.max_src_len, "max source position length: ", args.max_src_position)
            if self.max_tgt_len:
                print("max target sentence length: ", self.max_tgt_len, "max target position length: ", args.max_tgt_position)
            if args.position_method == 'Embedding' and \
               (args.max_src_position > self.max_src_len or args.max_tgt_position > self.max_tgt_len):
                print("You are using Positional Embedding and max source and target position are set greater than max sentence length, \
                       the vectors in Positional Embedding that exceed the max sentence length whill not be trained.")
        self.shuffle = shuffle
        if seed is not None:
            random.seed(seed)
        process_invalid_date(source=self.source,
                             target_input=self.target_input,
                             target_output=self.target_output,
                             args=args)
        data, self.rank = _get_tokens(self.source, self.target_input, self.target_output, shuffle, args)
        return data, 1


def pad_batch(batch, pad_index):
    max_len = max(len(seq) for seq in batch)
    return [list(seq) + [pad_index] * (max_len - len(seq)) for seq in batch]


def save_data_loader(dataloader, save_file):

    save_path = os.path.join(*os.path.split(save_file)[:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_file, 'wb') as f:
        pickle.dump(dataloader, f)


def preprocess():
    args = get_args()
    src_word2index, _ = load_vocab(args.src_vocab)

    source = data_process(filelist=[args.source],
                          word2index=src_word2index)

    if args.target is not None:
        tgt_word2index, _ = load_vocab(args.tgt_vocab)
        target = data_process(filelist=[args.target],
                              word2index=tgt_word2index)

    else:

        target = None

    dataloader = data_loader(source=source,
                             target=target,
                             BOS_index=constants.BOS_index, 
                             EOS_index=constants.EOS_index)

    save_data_loader(dataloader, args.save_file)


if __name__ == '__main__':

    preprocess()
