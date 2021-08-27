import  torch
import  time
import  functools
from    .tools import sync_between_gpu, move2cuda


def print_info(func):
    @functools.wraps(func)
    def wrapper(args, model, info, input, target):
        total_tok = info['total_tok']
        cnt = info['cnt']
        total_loss = info['total_loss']
        st_time = info['st_time']
        ntoken = (input['source'] != args.PAD_index).sum().item()
        batch_size = input['source'].size(0)
        total_tok += ntoken
        cnt += batch_size
        loss = func(input=input,
                    target=target,
                    model=model,
                    criterion=args.criterion)
        total_loss += loss * batch_size
        if args.batch % args.batch_print_info == 0:
            if args.world_size > 1:
                total_loss = sync_between_gpu(total_loss)
                total_tok = sync_between_gpu(total_tok)
                cnt = sync_between_gpu(cnt)
            total_time = time.time() - st_time
            st_time = time.time()
            if args.rank == 0:
                print(f'Batch: {args.batch}\tloss: {round(total_loss / cnt, 5)}\t\
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
def step(input, target, model, criterion):
    output = model(**input)
    loss = criterion(output, move2cuda(target))
    step_loss = loss.item()
    loss.backward()
    return step_loss


def run(args):
    train_data = args.train_data
    optimizer = args.optimizer
    model = args.model
    model.train()
    info = {'total_loss': 0,
            'cnt': 0,
            'total_tok': 0,
            'st_time': time.time()}
    optimizer.zero_grad()

    for i, (input, target) in enumerate(train_data):
        setattr(args, 'batch', i)
        info = step(args,
                    model=model,
                    info=info,
                    input=input,
                    target=target)
        optimizer.update_optim()
    optimizer.step()
    optimizer.zero_grad()


def fit(args, epoch):
    torch.backends.cudnn.benchmark = True
    rank = args.rank
    EPOCH = args.epoch
    checkpoint = args.checkpoint
    while epoch < EPOCH:
        if rank == 0:
            print('+' * 80)
            print(f'EPOCH: {epoch + 1}')
            print('-' * 80)
        run(args)
        epoch += 1
        args.train_data.sampler.set_epoch(epoch)
        if rank == 0 and checkpoint is not None:
            checkpoint.save_point(model=args.model,
                                  optim=args.optimizer,
                                  epoch=epoch)
