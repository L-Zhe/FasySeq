import  torch
from    torch import FloatTensor
import  time
import  torch.distributed as dist
import  functools


def sync_between_gpu(val):
    val = FloatTensor([val]).cuda(non_blocking=True)
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    return val.item()


def move2cuda(data):
    if data.dim() == 3:
        data = data[0]
    # max_len = (data != PAD_index).sum(dim=-1).max().item()
    # data = data[:, :max_len]
    return data.cuda(non_blocking=True)


def print_info(func):
    @functools.wraps(func)
    def wrapper(model, criterion, source, target_input, target_output, 
                info, batch, PAD_index, print_flag, rank, world_size):
        total_tok = info['total_tok']
        cnt = info['cnt']
        total_loss = info['total_loss']
        st_time = info['st_time']
        ntoken = (source != PAD_index).sum().item()
        batch_size = source.size(0)
        total_tok += ntoken
        cnt += batch_size
        loss = func(source=source,
                    target_input=target_input,
                    target_output=target_output,
                    model=model,
                    criterion=criterion)
        total_loss += loss * batch_size
        if print_flag:
            if world_size > 1:
                total_loss = sync_between_gpu(total_loss)
                total_tok = sync_between_gpu(total_tok)
                cnt = sync_between_gpu(cnt)
            total_time = time.time() - st_time
            st_time = time.time()
            if rank == 0:
                print(f'Batch: {batch}\tloss: {round(total_loss / cnt, 4)}\tTok pre Sec: {int(total_tok / total_time)}\t\tTime: {int(total_time)}')
            total_loss = 0
            cnt = 0
            total_tok = 0
        return {'total_loss': total_loss,
                'cnt': cnt,
                'total_tok': total_tok,
                'st_time': st_time}

    return wrapper


@print_info
def step(source, target_input, target_output,
         model, criterion):
    source = move2cuda(source)
    target_input = move2cuda(target_input)
    output = model(mode='train',
                   source=source,
                   target=target_input)
    del source, target_input
    target_output = move2cuda(target_output)
    loss = criterion(output, target_output)
    step_loss = loss.item()
    loss.backward()
    del loss, target_output
    return step_loss


def run(model, optimizer, criterion, rank, world_size, 
        batch_print_info, PAD_index, train_data):
    model.train()
    info = {'total_loss': 0,
            'cnt': 0,
            'total_tok': 0,
            'st_time': time.time()}
    optimizer.zero_grad()
    for i, (source, target_input, target_output) in enumerate(train_data):
        info = step(model=model,
                    criterion=criterion,
                    source=source,
                    target_input=target_input,
                    target_output=target_output,
                    info=info,
                    batch=i,
                    PAD_index=PAD_index,
                    print_flag=i % batch_print_info == 0,
                    rank=rank,
                    world_size=world_size)
        optimizer.update_optim()
    optimizer.step()
    optimizer.zero_grad()


def fit(args, start_epoch):
    torch.backends.cudnn.benchmark = True
    rank = args.rank
    world_size = args.world_size
    PAD_index = args.PAD_index
    batch_print_info = args.batch_print_info
    EPOCH = args.epoch
    train_data = args.train_data
    checkpoint = args.checkpoint
    model = args.model
    optimizer = args.optimizer
    criterion = args.criterion
    for epoch in range(start_epoch, EPOCH):
        if rank == 0:
            print('+' * 80)
            print(f'EPOCH: {epoch + 1}')
            print('-' * 80)
        run(model=model,
            optimizer=optimizer,
            criterion=criterion,
            rank=rank,
            world_size=world_size,
            batch_print_info=batch_print_info,
            PAD_index=PAD_index,
            train_data=train_data)
        train_data.sampler.set_epoch(epoch + 1)
        if rank == 0 and checkpoint is not None:
            checkpoint.save_point(model=model,
                                  optim=optimizer,
                                  epoch=epoch + 1)
