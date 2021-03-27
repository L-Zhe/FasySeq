import  torch
from    torch import FloatTensor
import  time
import  torch.distributed as dist
import  functools


def sync_between_gpu(val):
    val = FloatTensor([val]).cuda(non_blocking=True)
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    return val.item()


def move2cuda(data, PAD_index):
    if data.dim() == 3:
        data = data[0]
    max_len = (data != PAD_index).sum(dim=-1).max().item()
    data = data[:, :max_len]
    return data.cuda(non_blocking=True)


def print_info(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        total_tok = kwargs['info']['total_tok']
        cnt = kwargs['info']['cnt']
        total_loss = kwargs['info']['total_loss']
        st_time = kwargs['info']['st_time']
        ntoken = (kwargs['source'] != kwargs['PAD_index']).sum().item()
        batch_size = kwargs['source'].size(0)
        total_tok += ntoken
        cnt += batch_size
        loss = func(self,
                    source=kwargs['source'],
                    target_input=kwargs['target_input'],
                    target_output=kwargs['target_output'])
        total_loss += loss * batch_size
        if kwargs['print_flag']:
            if kwargs['world_size'] > 1:
                total_loss = sync_between_gpu(total_loss)
                total_tok = sync_between_gpu(total_tok)
                cnt = sync_between_gpu(cnt)
            total_time = time.time() - st_time
            st_time = time.time()
            if kwargs['rank'] == 0:
                print('Batch: %d\tloss: %f\tTok pre Sec: %d\t\tTime: %d' %
                      (kwargs['batch'], total_loss / cnt, total_tok / total_time, total_time))
            total_loss = 0
            cnt = 0
            total_tok = 0
        return {'total_loss': total_loss,
                'cnt': cnt,
                'total_tok': total_tok,
                'st_time': st_time}

    return wrapper


class Fit:

    def __init__(self, args, params):

        self.train_data = params['train_data']
        self.model      = params['model']
        self.optimizer  = params['optimizer']
        self.criterion  = params['criterion']
        self.checkpoint = params['checkpoint']
        self.PAD_index  = args.PAD_index
        self.EPOCH      = args.epoch
        self.batchPrintInfo = args.batch_print_info
        self.rank = args.rank
        self.world_size = args.world_size

    @print_info
    def step(self, source, target_input, target_output):
        source = move2cuda(source, self.PAD_index)
        target_input = move2cuda(target_input, self.PAD_index)
        output = self.model(mode='train',
                            source=source,
                            target=target_input)
        del source, target_input
        target_output = move2cuda(target_output, self.PAD_index)
        loss = self.criterion(output, target_output)
        step_loss = loss.item()
        loss.backward()
        del loss, target_output
        return step_loss

    def run(self, data):
        self.model.train()
        info = {'total_loss': 0,
                'cnt': 0,
                'total_tok': 0,
                'st_time': time.time()}
        self.optimizer.zero_grad()
        for i, (source, target_input, target_output) in enumerate(data):
            info = self.step(source=source,
                             target_input=target_input,
                             target_output=target_output,
                             info=info,
                             batch=i,
                             PAD_index=self.PAD_index,
                             print_flag=i % self.batchPrintInfo == 0,
                             rank=self.rank,
                             world_size=self.world_size)
            self.optimizer.update_optim(self.model.parameters())

        self.optimizer.step(self.model.parameters())
        self.optimizer.zero_grad()

    def __call__(self, start_epoch):
        torch.backends.cudnn.benchmark = True
        for epoch in range(start_epoch, self.EPOCH):
            if self.rank == 0:
                print('+' * 80)
                print('EPOCH: %d' % (epoch + 1))
                print('-' * 80)
            self.run(self.train_data)
            self.train_data.sampler.set_epoch(epoch)
            if self.rank == 0 and self.checkpoint is not None:
                self.checkpoint.save_point(model=self.model,
                                           optim=self.optimizer,
                                           epoch=epoch + 1)
