# Absolut Halal
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
#----------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import random


train_dataset=dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset=dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
# print(train_dataset.test_data.size()) # old name - data
print(train_dataset.data.size())  # old name - targets
print(train_dataset.train_labels.size())
print(test_dataset.data.size())
batch_size=100
epochs=10
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv2d1=nn.Conv2d(in_channels=1,out_channels=20,kernel_size=4,stride=1,padding=3)
        self.relu1=nn.ReLU()
        self.maxPool1=nn.MaxPool2d(kernel_size=2,stride=1)
        self.conv2d2=nn.Conv2d(in_channels=20,out_channels=10,kernel_size=4,stride=1,padding=3)
        self.relu2=nn.ReLU()
        self.maxPool2=nn.MaxPool2d(kernel_size=2,stride=1)
        self.fc1=nn.Linear(10*32*32,10)
    def forward(self,x):
        out=self.conv2d1(x)
        out=self.relu1(out)
        out=self.maxPool1(out)
        out=self.conv2d2(out)
        out=self.relu2(out)
        out=self.maxPool2(out)
        # print("print(out.size(0)) ", out.size(0))
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        return out
myModel=Model()
cost_func=nn.CrossEntropyLoss()
learn_rate=0.01
optimizer=torch.optim.SGD(myModel.parameters(), lr=learn_rate)

for n_epoch in range (epochs):
    if((n_epoch%1)==0): # change the denominator
        # total=len(test_loader)
        # print("print(len(test_loader)) ", len(test_loader))
        predicted=0
        total=0

        with torch.no_grad():
            for c, (input_test,targets) in enumerate (test_loader):
                size_labels = len(targets)
                # print("print(targets.size(0))", targets.size(0))  # 20 workable original
                # print("print(targets.size())", targets.size()[0])  # 20 - workable
                # # print("print(targets.size())", targets.size().item())  # 20 -doesnt work
                output = myModel(input_test)
                _, predict=torch.max(output,1)
                # predicted=(predict==targets).sum()
                for x in range(size_labels):
                    total+=1
                    # print("print(targets.data[x].item())= ", targets.data[x].item())
                    if(targets.data[x].item()==predict.data[x].item()):
                        predicted+=1
            accuracy=(predicted/total)*100
            print("accuracy= ",accuracy, " after epoch ", n_epoch)
    print("train started")
    for i,(input_train,targets) in enumerate(train_loader):
        if(i%100):
            print("iteration: ",i)
        input_train=input_train.requires_grad_()
        optimizer.zero_grad()
        predict=myModel(input_train)
        loss=cost_func(predict,targets)
        # print("loss.item() ", loss.item())
        loss.backward()
        optimizer.step()
    print("training at epoch№",n_epoch, " ended")

# print(myModel.parameters())
# print("parameters in first 0 tensor of parameters ", list(myModel.parameters())[0].size())
# print("------------------------------------")
# print("Ere we go 0! : list(myModel.parameters())[0] ", list(myModel.parameters())[0])
# print("------------------------------------")
# print("parameters in second 1 tensor of parameters ", list(myModel.parameters())[1].size())
# print("Ere we go 1! : list(myModel.parameters())[1] ", list(myModel.parameters())[1])

print("end")



















