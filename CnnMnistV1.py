from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch):
    network.train()
    accum_loss=0.0
    n=1 # can you write it without global variable? retard
    for i,(data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        # predict=network.forward(data) # you have failed
        predict=network(data)
        loss=F.nll_loss(predict,target)
        # print(loss)
        accum_loss+=loss
        loss.backward()
        optimizer.step()
        n+=1
    accum_loss/=n
    print("epoch №",epoch, "error=", accum_loss)


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
#-----------------------------------------------
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

#-----------------------------------------------
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# print("print(example_data.shape) ", example_data.shape)
#-----------------------------------------------
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig
#-----------------------------------------------
network=Net()
optimizer=optim.SGD(network.parameters(),lr=learning_rate,momentum=momentum)
# network.forward(train_loader)
print("please in the value of epochs")
epochs=int(input())
for epoch in range(1, (epochs+1)):
    train(epoch)
print("here is the end")
