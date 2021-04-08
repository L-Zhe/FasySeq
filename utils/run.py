import  torch
import  time
import  functools
from    .tools import sync_between_gpu, move2cuda


def print_info(func):
    @functools.wraps(func)
    def wrapper(model, criterion, source, target_input, target_output, src_mask,
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
                    src_mask=src_mask,
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
                print(f'Batch: {batch}\tloss: {round(total_loss / cnt, 5)}\t\
                        Tok pre Sec: {int(total_tok / total_time)}\t\tTime: {int(total_time)}')
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
         src_mask, model, criterion):
    output = model(mode='train',
                   source=source,
                   target=target_input,
                   src_mask=src_mask)
    loss = criterion(output, move2cuda(target_output))
    step_loss = loss.item()
    loss.backward()
    return step_loss


def run(model, optimizer, criterion, rank, world_size, 
        batch_print_info, PAD_index, train_data):
    model.train()
    info = {'total_loss': 0,
            'cnt': 0,
            'total_tok': 0,
            'st_time': time.time()}
    optimizer.zero_grad()
    for i, (source, target_input, target_output, src_mask) in enumerate(train_data):
        info = step(model=model,
                    criterion=criterion,
                    source=source,
                    target_input=target_input,
                    target_output=target_output,
                    src_mask=src_mask,
                    info=info,
                    batch=i,
                    PAD_index=PAD_index,
                    print_flag=i % batch_print_info == 0,
                    rank=rank,
                    world_size=world_size)
        optimizer.update_optim()
    optimizer.step()
    optimizer.zero_grad()


def fit(args, epoch):
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
    while epoch < EPOCH:
        epoch += 1
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
