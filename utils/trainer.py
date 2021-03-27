from    .checkpoint import checkpoint
from    .makeModel import make_model
from    .run import Fit
from    . import constants
from    torch.utils.data import DataLoader, distributed
import  torch.distributed as dist
import  torch
from    Transformer.Module import WarmUpOpt, LabelSmoothing
import  pickle


def get_checkpoint(args, params):
    check_point = checkpoint(save_path=args.checkpoint_path,
                             checkpoint_num=args.checkpoint_num,
                             restore_file=args.restore_file)
    check_point.add_params(args)
    params['checkpoint'] = check_point
    model_state_dict = None
    optim_state_dict = None
    start_epoch = 0
    model_state_dict, optim_state_dict, start_epoch, model_config = params['checkpoint'].restore()
    for key, value in model_config.items():
        setattr(args, key, value)
    return model_config, model_state_dict, optim_state_dict, start_epoch


def get_dataloader(args, params, seed):
    with open(args.train_file, 'rb') as f:
        train_data = pickle.load(f)
    train_dataset, batch_size = train_data.set_param(shuffle=True,
                                                     args=args,
                                                     seed=seed)
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
    params['train_data'] = train_loader
    return batch_size


def get_model(args, params, model_state_dict, optim_state_dict):
    criterion = LabelSmoothing(smoothing=args.smoothing,
                               ignore_index=constants.PAD_index).cuda(args.rank)
    params['criterion']  = criterion
    check_point = None
    dist.barrier()
    model = make_model(args, model_state_dict, args.rank)
    optimizer = WarmUpOpt(parameters=model.parameters(),
                          state_dict=optim_state_dict,
                          args=args)
    params['model'] = model
    params['optimizer'] = optimizer


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
    params = {}
    model_config, model_state_dict, optim_state_dict, start_epoch = get_checkpoint(args, params)
    batch_size = get_dataloader(args, params, seed)
    get_model(args, params, model_state_dict, optim_state_dict)
    del model_state_dict, optim_state_dict
    fit = Fit(args, params)
    fit(start_epoch=start_epoch)
