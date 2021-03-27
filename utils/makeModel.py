from    Transformer.Model import transformer
from    torch.nn.parallel import DistributedDataParallel
import  numpy as np


def make_model(config, model_state_dict, gpu, dist=True):
    model = transformer(config)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    model.cuda(gpu)
    if gpu == 0:
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in model.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue
        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')
    if dist:
        return DistributedDataParallel(model, device_ids=[gpu])
    return model
