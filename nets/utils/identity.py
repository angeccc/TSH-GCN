import torch
import torch.nn as nn


class identity(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input):

        return input

