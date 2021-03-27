import  os
import  torch
from    queue import Queue
from    math import inf
from    copy import deepcopy


class checkpoint:

    def __init__(self, save_path=None, checkpoint_num=inf, restore_file=None):
        self.save_path = save_path
        self.restore_file = restore_file
        if save_path is None:
            return
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.best_score = 0
        self.checkpoint_path = os.path.join(save_path, 'checkpoint.pkl')
        self.best_model_path = os.path.join(save_path, 'model.best')
        self.checkpoint_num = checkpoint_num
        self.checkpoint_file = None
        self.checkpoint_cnt = 0

        if checkpoint_num != inf and checkpoint_num != 1:
            self.checkpoint_file = Queue(maxsize=checkpoint_num)

    def add_params(self, args):
        param = ['embedding_dim', 'share_embed', 'num_layer_decoder', 'num_layer_encoder', 'src_vocab_size',
                 'd_ff', 'num_head', 'max_src_position', 'max_tgt_position', 'position_method', 'tgt_vocab_size',
                 'dropout_embed', 'dropout_sublayer', 'normalize_before', 'smoothing', 'warmup_steps', 'vocab_size',
                 'factor', 'learning_rate', 'min_learning_rate', 'beta_1', 'beta_2', 'eps', 'weight_decay']
        self.model_config = {}
        for item in param:
            self.model_config[item] = getattr(args, item)

    def save_point(self, model, optim, epoch):
        if self.save_path is None:
            return
        print('===> Save checkpoint.')
        self.checkpoint_cnt += 1
        point = {
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'epoch': epoch,
            'best_score': self.best_score,
            'checkpoint_cnt': self.checkpoint_cnt,
            'model_config': self.model_config
        }

        if self.checkpoint_num == 1:
            torch.save(point, self.checkpoint_path)
            self.restore_file = self.checkpoint_path
        else:
            file_name = list(deepcopy(self.checkpoint_path))
            file_name.insert(-4, str(self.checkpoint_cnt))
            file_name = ''.join(file_name)
            del_file = None
            if self.checkpoint_file is not None and self.checkpoint_file.full():
                del_file = self.checkpoint_file.get()
            torch.save(point, file_name)
            if self.checkpoint_file is not None:
                if del_file is not None and os.path.exists(del_file):
                    os.remove(del_file)
                self.checkpoint_file.put(file_name)

            self.restore_file = file_name

    def save_best(self, model, score):
        model = model.eval()
        if score > self.best_score:
            self.best_score = score
            torch.save(model.state_dict(), self.best_model_path)

    def restore(self):
        model_state_dict = None
        optim_state_dict = None
        model_config = {}
        start_epoch = 0
        if self.restore_file is not None and os.path.exists(self.restore_file):
            print('===> Restore from checkpoint.')
            point = torch.load(self.restore_file, map_location=torch.device('cpu'))
            model_state_dict = process_state_dict(point['model'], 1)
            self.best_score = point['best_score']
            self.checkpoint_cnt = point['checkpoint_cnt']
            start_epoch = point['epoch']
            optim_state_dict = point['optim']
            model_config = point['model_config']
            self.model_config = model_config
        return model_state_dict, optim_state_dict, start_epoch, model_config


def process_state_dict(state_dict, device_num):
    def add_module(k):
        if not k.startswith('module.'):
            return 'module.' + k
        return k
    if device_num <= 1:
        return {k.replace('module.', ''): v for k, v in
                state_dict.items()}
    else:
        return {add_module(k): v for k, v in
                state_dict.items()}


def load_model(path):
    assert os.path.exists(path)
    point = torch.load(path, map_location=torch.device('cpu'))
    state_dict = process_state_dict(point['model'], 1)
    return state_dict, point['model_config']
     
