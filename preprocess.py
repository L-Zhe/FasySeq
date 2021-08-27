import  argparse
from    utils import constants
from    utils.tools import load_vocab
import  os
from    math import inf
from    torch import LongTensor
import  pickle
from    tqdm import tqdm
from    utils.tools import pad_mask


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--src_vocab', type=str)
    parser.add_argument('--tgt_vocab', type=str, default=None)
    parser.add_argument('--save_file', type=str)

    return parser.parse_args()


def data_process(filelist, word2index, lower):
    '''
    Change word to index. 
    '''
    data = []
    for file in filelist:
        with open(file, 'r', encoding='utf-8') as f:
            if lower:
                data.extend([line.strip('\n').lower().split() \
                             for line in f.readlines()])
            else:
                data.extend([line.strip('\n').split() \
                             for line in f.readlines()]) 

    def prepare_sequence(seq):
        return list(map(lambda word: word2index[constants.UNK_WORD] 
                        if word2index.get(word) is None else word2index[word], seq))

    return [prepare_sequence(seq) for seq in tqdm(data)]


def process_invalid_date(data, args):
    max_src_position = args.max_src_position
    max_tgt_position = args.max_tgt_position
    discard_invalid_data = getattr(args, 'discard_invalid_data', False)
    source = data['source']
    target = data.get('target')
    if max_src_position != inf or max_tgt_position != inf:
        total_len = len(source)
        if discard_invalid_data:
            del_index = []
            for i in range(total_len):
                if len(source[i]) > max_src_position or \
                   (target and len(target[i]) > max_tgt_position - 1):
                    del_index.insert(0, i)
            if args.rank == 0:
                print("===> Discard invalid data: %d" % len(del_index))
            for index in del_index:
                del source[index]
                if target:
                    del target[index]
        else:
            for i in range(total_len):
                source[i] = source[i][:max_src_position]
                if target:
                    target[i] = target[i][:max_tgt_position - 1]


def sort_data_by_len(source, target_input=None, target_output=None):
    if target_input and target_output:
        data = list(zip(source, target_input, target_output))
        data.sort(key=lambda line: max(len(line[0]), len(line[1])), reverse=True)
        return zip(*data)
    else:
        data = [(index, value) for index, value in sorted(list(enumerate(source)), key=lambda x: len(x[1]), reverse=True)]
        return zip(*data)


def get_tokens(data, args):
    total_len = len(data['source'])
    train_flag = args.mode == 'train'
    if train_flag:
        target_input = [[args.BOS_index] + seq for seq in data['target']]
        target_output = [seq + [args.EOS_index] for seq in data['target']]
        source, target_input, target_output = sort_data_by_len(source=data['source'],
                                                               target_input=target_input,
                                                               target_output=target_output)
    else:
        rank, source = sort_data_by_len(data['source'])
    del data
    index_pair = []
    st = 0
    total_len = len(source)
    while st < total_len:
        if train_flag:
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
        src = LongTensor(pad_batch(source[st:ed], args.PAD_index))
        src_mask = pad_mask(src, args.PAD_index)
        if train_flag:
            tgt_input = LongTensor(pad_batch(target_input[st:ed], args.PAD_index))
            tgt_output = LongTensor(pad_batch(target_output[st:ed], args.PAD_index))
            data.append((src, tgt_input, tgt_output, src_mask))
        else:
            data.append((rank[st:ed], src, src_mask))
    return data


def restore_rank(data):
    data.sort(key=lambda x: x[0])
    return list(zip(*data))[1]


def get_data(args, data=None):
    if data is None:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
    max_src_len = data['max_src_len']
    max_tgt_len = data.get('max_tgt_len')
    args.max_src_position = min(max_src_len, args.max_src_position)
    if max_tgt_len:
        args.max_tgt_position = min(max_tgt_len + 1, args.max_tgt_position)

    process_invalid_date(data=data,
                         args=args)
    data = get_tokens(data, args)
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
    src_word2index, _, src_lower = load_vocab(args.src_vocab)

    source = data_process(filelist=[args.source],
                          word2index=src_word2index,
                          lower=src_lower)

    max_src_len = max(len(seq) for seq in source)
    data = {'source': source,
            'max_src_len': max_src_len}

    if args.target is not None:
        tgt_word2index, _, tgt_lower = load_vocab(args.tgt_vocab)
        target = data_process(filelist=[args.target],
                              word2index=tgt_word2index,
                              lower=tgt_lower)
        max_tgt_len = max(len(seq) for seq in target)
        data['target'] = target
        data['max_tgt_len'] = max_tgt_len

    save_data_loader(data, args.save_file)


def train_collate_fn(batch):
    source, target_input, target_output, src_mask = batch[0]
    return {'mode': 'train', 'source': source, 'target': target_input, 'src_mask': src_mask}, \
           target_output


def test_collate_fn(batch):
    rank, source, src_mask = batch
    return rank, {'source': source, 'src_mask': src_mask}


if __name__ == '__main__':

    preprocess()
