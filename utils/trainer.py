from    .checkpoint import checkpoint
from    .makeModel import make_model
from    .run import fit
from    . import constants
from    torch.utils.data import DataLoader, distributed
import  torch.distributed as dist
import  torch
from    Transformer.Module import WarmUpOpt, LabelSmoothing
from    preprocess import get_data
from    utils.tool import show_info


def get_checkpoint(args):
    if args.rank == 0:
        print("===>Get Checkpoint...")
    check_point = checkpoint(save_path=args.checkpoint_path,
                             checkpoint_num=args.checkpoint_num,
                             restore_file=args.restore_file)
    setattr(args, 'checkpoint', check_point)
    model_state_dict = None
    optim_state_dict = None
    start_epoch = 0
    model_state_dict, optim_state_dict, start_epoch, model_config = check_point.restore()
    for key, value in model_config.items():
        setattr(args, key, value)
    dist.barrier()
    return model_state_dict, optim_state_dict, start_epoch


def get_dataloader(args, seed):
    train_dataset, batch_size = get_data(args=args)
    train_sampler = distributed.DistributedSampler(train_dataset,
                                                   shuffle=True,
                                                   seed=seed,
                                                   num_replicas=args.world_size,
                                                   rank=args.rank)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True,
                              sampler=train_sampler)
    setattr(args, 'train_data', train_loader)
    dist.barrier()


def get_model(args, model_state_dict, optim_state_dict):
    if args.rank == 0:
        print("==>Build Model...")
    args.checkpoint.add_params(args)
    criterion = LabelSmoothing(smoothing=args.smoothing,
                               ignore_index=constants.PAD_index).cuda(args.rank)
    setattr(args, 'criterion', criterion)
    model = make_model(args, model_state_dict, args.rank)
    optimizer = WarmUpOpt(parameters=model.parameters(),
                          state_dict=optim_state_dict,
                          args=args)
    setattr(args, 'model', model)
    setattr(args, 'optimizer', optimizer)


def trainer(gpu, args, seed):

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=gpu)
    torch.cuda.set_device(gpu)
    setattr(args, 'rank', gpu)
    setattr(args, 'BOS_index', constants.BOS_index)
    setattr(args, 'EOS_index', constants.EOS_index)
    setattr(args, 'PAD_index', constants.PAD_index)
    model_state_dict, optim_state_dict, start_epoch = get_checkpoint(args)
    get_dataloader(args, seed)
    get_model(args, model_state_dict, optim_state_dict)
    del model_state_dict, optim_state_dict
    show_info(args)
    fit(args=args, 
        start_epoch=start_epoch)
