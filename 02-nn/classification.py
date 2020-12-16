import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 建立数据集

n_data = torch.ones(100, 2)
# print("n_data size: ", n_data.size())
x0 = torch.normal(2*n_data, 1)
# print('x0: ', x0)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)
x, y = Variable(x), Variable(y)

# data = x.data.numpy()
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)

print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
# loss_func = torch.nn.MSELoss() # 交叉熵损失函数

plt.ion()
plt.show()

for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.clf()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()


# 问题:
# 1. 为什么要用交叉熵损失函数，而不能用均方误差损失函数;
# 2. torch.max() 函数的用法;
