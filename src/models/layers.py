import torch.nn as nn

class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, feature_dim) '''
        if self.batch_first:
            batch_size, time_steps, feature_dim = x.shape
        else:
            time_steps, batch_size, feature_dim = x.shape
        c_in = x.reshape((batch_size * time_steps, feature_dim))
        c_out = self.module(c_in)
        r_in = c_out.view(batch_size, time_steps, -1)
        if self.batch_first is False:
            r_in = r_in.permute(1, 0, 2)
        return r_in