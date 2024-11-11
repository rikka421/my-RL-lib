import torch
import torch.nn as nn
from my_rl_library.utils.MyFCNN import MyFCNN

class MyDuelingNN(nn.Module):
    # 普通的NN网络. 使用全连接层和ReLu激活函数
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(MyDuelingNN, self).__init__()

        self.A = MyFCNN(input_dim, output_dim, hidden_dims)
        self.V = MyFCNN(input_dim, 1, hidden_dims)

        self.fcs = nn.ModuleList([self.V, self.A])
    def forward(self, x):
        v = self.V(x)
        a = self.A(x)
        mean_a = torch.mean(a, dim=1, keepdim=True)

        x = v + a - mean_a

        return x
