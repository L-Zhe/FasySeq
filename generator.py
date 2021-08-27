import  torch
from    utils.lang import translate2word
from    utils import constants
from    tqdm import tqdm
from    torch.utils.data import DataLoader
from    utils.args import get_generate_config
from    utils.tools import load_vocab
from    utils.makeModel import make_model
from    utils.tools import save2file
from    utils.checkpoint import load_model
from    utils.eval import Eval
from    preprocess import data_process, get_data, restore_rank
import  os


def _batch(args, st, ed):
    try:
        length = (args.source[st:ed] != args.PAD_index).sum(dim=-1)
        max_src_len = length.max().item()
        max_length = min(max_src_len * args.max_alpha, args.max_length)
        max_length = min(max_src_len + args.max_add_token, max_length)
        output = args.model(source=args.source[st:ed, :max_src_len],
                            src_mask=args.src_mask[st:ed, :max_src_len],
                            mode='test',
                            max_length=max_length)
        output = output.tolist()
        for i in range(len(output)):
            output[i] = output[i][1:]
            if args.EOS_index in output[i]:
                end_index = output[i].index(args.EOS_index)
                index = min(int(args.max_alpha * length[i].item()), end_index)
                index = min(index, length[i] + args.max_add_token)
                output[i] = output[i][:index]

    except RuntimeError:
        if ed - st == 1:
            raise RuntimeError
        print('==>Reduce Batch Size')
        torch.cuda.empty_cache()
        output = []
        length = max(int((ed - st) / 4), 1)
        while st < ed:
            _ed = min(st + length, ed)
            output.extend(_batch(args, st, _ed))
            st = _ed
    return output

@torch.no_grad()
def generate(args):
    outputs = []
    rank = []
    args.model.eval()
    print('===>Start Generate.')
    for r, source, src_mask in tqdm(args.data):
        if source.dim() == 3:
            source = source[0]
            src_mask = src_mask[0]
        setattr(args, 'source', source)
        setattr(args, 'src_mask', src_mask)
        del source, src_mask
        outputs.extend(_batch(args, 0, args.source.size(0)))
        rank.extend(r)
    return list(zip(rank, translate2word(outputs, args.tgt_index2word)))


def get_dataloader(args):
    data = None
    setattr(args, 'mode', 'test')
    if args.raw_file:
        source = data_process(filelist=[args.raw_file],
                              word2index=args.src_word2index,
                              lower=args.lower)
        del args.src_word2index
        max_src_len = max(len(seq) for seq in source)

        data = {'source': source,
                'max_src_len': max_src_len}

    dataset, batch_size = get_data(args=args,
                                   data=data)
    dataset = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True)
    setattr(args, 'data', dataset)


def get_vocab_info(args):
    if args.file:
        if args.share_embed:
            _, tgt_index2word, lower = load_vocab(args.vocab)
            assert len(tgt_index2word) == args.vocab_size
        else:
            _, tgt_index2word, lower = load_vocab(args.tgt_vocab)
            assert len(tgt_index2word) == args.tgt_vocab_size
    else:
        if args.share_embed:
            src_word2index, tgt_index2word, lower = load_vocab(args.vocab)
            assert (len(src_word2index) == args.vocab_size)

        else:
            src_word2index, _, lower = load_vocab(args.src_vocab)
            _, tgt_index2word, _ = load_vocab(args.tgt_vocab)
        assert (len(src_word2index) == args.src_vocab_size) 
        assert (len(tgt_index2word) == args.tgt_vocab_size)
        setattr(args, 'src_word2index', src_word2index)
    setattr(args, 'tgt_index2word', tgt_index2word)
    setattr(args, 'lower', lower)
    if args.position_method == 'Embedding':
        if args.share_embed:
            args.max_length = min(args.max_length, max(args.max_src_position, args.max_tgt_position))
        else:
            args.max_length = min(args.max_length, args.max_tgt_position)

def _main():

    args = get_generate_config()
    setattr(args, 'PAD_index', constants.PAD_index)
    setattr(args, 'BOS_index', constants.BOS_index)
    setattr(args, 'EOS_index', constants.EOS_index)
    setattr(args, 'rank', 0)
    assert (args.file is None) ^ (args.raw_file is None)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_num)
    model_state_dict, model_config = load_model(args.model_path)
    for key, value in model_config.items():
        setattr(args, key, value)
    print(args)
    get_vocab_info(args)
    model = make_model(args, model_state_dict, 0, False)
    setattr(args, 'model', model)
    get_dataloader(args)
    output = generate(args)
    output = restore_rank(output)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    save_file = os.path.join(args.output_path, 'result.txt')
    save2file(output, save_file)

    if args.ref_file is not None:
        eval = Eval(reference_file=args.ref_file)
        eval(save_file)


if __name__ == '__main__':
    _main()
