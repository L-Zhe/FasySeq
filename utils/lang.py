import  torch
from    torch import LongTensor
from    torch.utils.data import DataLoader, TensorDataset
from    utils import constants


def lang(file):
    with open(file, 'r') as f:
        corpus = [[int(word) for word in seq.strip().split()] for seq in f.readlines()]
    return corpus
    
def get_dataloader(source, target_inputs=None, 
    target_outputs=None, batch_size=64, shuffle=False):
    source = LongTensor(source)
    if target_inputs is not None and target_outputs is not None:
        target_inputs = LongTensor(target_inputs)
        target_outputs = LongTensor(target_outputs)
        data = TensorDataset(source, target_inputs, target_outputs)
    else:
        data = source
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def translate2word(sequence, index2word):
    return [' '.join([index2word[index] for index in seq]).replace('@@', ' ').replace('  ', '')
             for seq in sequence]