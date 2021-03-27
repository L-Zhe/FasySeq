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
from    preprocess import data_process, data_loader
import  os
from    os import environ
import  pickle


class generator:
    def __init__(self, *args, **kwargs):
        self.data = kwargs['data']
        self.index2word = kwargs['index2word']
        self.max_length = kwargs['max_length']
        self.model = kwargs['model']

    def _batch(self, st, ed):
        try:
            max_src_len = (self.source[st:ed] != constants.PAD_index).sum(dim=-1).max().item()
            output = self.model(source=self.source[st:ed, :max_src_len].cuda(),
                                mode='test',
                                max_length=self.max_length)
            output = output.tolist()
            for i in range(len(output)):
                output[i] = output[i][1:]
                if constants.EOS_index in output[i]:
                    end_index = output[i].index(constants.EOS_index)
                    output[i] = output[i][:end_index]
                # print(len(output[i]))

        except RuntimeError:
            if ed - st == 1:
                raise RuntimeError
            print('==>Reduce Batch Size')
            torch.cuda.empty_cache()
            output = []
            length = max(int((ed - st) / 4), 1)
            while st < ed:
                _ed = min(st + length, ed)
                output.extend(self._batch(st, _ed))
                st = _ed
        return output

    @torch.no_grad()
    def __call__(self):
        outputs = []
        self.model.eval()
        print('===>Start Generate.')
        for source in tqdm(self.data):

            if source.dim() == 3:
                source = source[0]
            max_src_len = (source != constants.PAD_index).sum(dim=-1).max().item()
            self.source = source[:, :max_src_len]
            del source
            # self.source = self.source.cuda()
            outputs.extend(self._batch(0, self.source.size(0)))
        return translate2word(outputs, self.index2word)


def _main():

    args = get_generate_config()
    setattr(args, 'PAD_index', constants.PAD_index)
    setattr(args, 'BOS_index', constants.BOS_index)
    setattr(args, 'EOS_index', constants.EOS_index)
    assert (args.file is None) ^ (args.raw_file is None)
    if args.cuda:
        environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_num)
    model_state_dict, model_config = load_model(args.model_path)
    for key, value in model_config.items():
        setattr(args, key, value)

    if args.share_embed:
        _, tgt_index2word = load_vocab(args.vocab)
        assert len(tgt_index2word) == args.vocab_size
    else:
        if args.raw_file:
            src_word2index, _ = load_vocab(args.src_vocab)
        _, tgt_index2word = load_vocab(args.tgt_vocab)
        assert (len(src_word2index) == args.src_vocab_size) and (len(tgt_index2word) == args.tgt_vocab_size)
    if args.position_method == 'Embedding':
        if args.share_embed:
            args.max_length = min(args.max_length,
                    max(args.max_src_position, args.max_tgt_position))
        else:
            args.max_length = min(args.max_length, args.max_tgt_position)
    model = make_model(args, model_state_dict, 0, False)
    if args.file is not None:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)

    else:
        src_word2index, _ = load_vocab(args.src_vocab)
        source = data_process(filelist=[args.raw_file],
                              word2index=src_word2index)

        data = data_loader(source=source,
                           PAD_index=constants.PAD_index)

    dataset, batch_size = data.set_param(False, args)
    dataset = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True)
    generate = generator(data=dataset,
                         index2word=tgt_index2word,
                         max_length=args.max_length,
                         model=model)

    outputs = generate()
    outputs = data.restore_rank(outputs)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    save_file = os.path.join(args.output_path, 'result.txt')
    save2file(outputs, save_file)

    if args.ref_file is not None:
        eval = Eval(reference_file=args.ref_file)
        eval(save_file)


if __name__ == '__main__':
    _main()
