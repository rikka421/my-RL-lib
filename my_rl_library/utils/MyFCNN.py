import torch
import torch.nn as nn

class MyFCNN(nn.Module):
    # 普通的NN网络. 使用全连接层和ReLu激活函数
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(MyFCNN, self).__init__()
        self.fcs = []
        if hidden_dims is None or len(hidden_dims) == 0:
            self.fcs.append(nn.Linear(input_dim, output_dim))
        else:
            self.fcs.append(nn.Linear(input_dim, hidden_dims[0]))
            for i in range(len(hidden_dims) - 1):
                self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.fcs.append(nn.Linear(hidden_dims[-1], output_dim))
        self.fcs = nn.ModuleList(self.fcs)
    def forward(self, x):
        for fc in self.fcs:
            x = torch.relu(fc(x))
        return x
