import torch
import torch.nn as nn
class MultiHeadedAttention(nn.module):
    def __init__(embed_size, downsample_size):
        super.__init__()
        self.w_q = nn.Linear(embed_size, downsample_size, bias=False)
        self.w_k = nn.Linear(embed_size, downsample_size, bias=False)
        self.w_v = nn.Linear(embed_size, downsample_size, bias=False)