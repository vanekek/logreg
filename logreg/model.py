import torch.nn as nn


def LogReg(input_size, num_classes):
    model = nn.Linear(input_size, num_classes)
    return model
