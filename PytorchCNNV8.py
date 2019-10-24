
# Absolut Halal
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
#----------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import random
import os


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
    # base_file="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1\Cnn1V1Parameters"
    def load_weights(self, weights_file):
        other, ext = os.path.splitext(weights_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(weights_file, map_location=lambda storage, loc: storage))
            #stirct false\true - можно параметры загружать с определённого места
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

myModel=Model()
cost_func=nn.CrossEntropyLoss()
learn_rate=0.01
optimizer=torch.optim.SGD(myModel.parameters(), lr=learn_rate)
#---------------------------------------------------------------------
# path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'
test_loader_manual=torch.utils.data.DataLoader(dataset=test_dataset)
# save_folder="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1"

save_folder="C:\\Users\\user\\Desktop\\DontTouchPLSpytorch\\Notes\\SavedParameters\\Cnn1\\"

file="Cnn1V1Parameters.pth"

torch.save(myModel.state_dict(), os.path.join(save_folder, file))
myModel.load_weights(save_folder+file)
#-------------------extract from enumerator (dataset) to a list for convenient access in further
# maybe batch sie in "dataloader" affects to it
size=test_dataset.__len__()
list_data_manual=[]
print("print(type(size)) = ", type(size))
print("print(size) = ", size)
for i,(image, target) in enumerate(test_loader_manual):
    list_data_manual.append((image,target))
    if (i==size):
        print("I == 1000")

print("out of cycle")



#-------------------extract from enumerator (dataset) to a list for convenient access in further
img_num=4 # manual input index of an image

#-------chk------------->>>>>>>>>>>>>>>HERE CONTINUE <<<<<<<-------------------



#-----------------ploting an image---------------
plt.ion # disable interactive mod
target=str(list_data_manual[img_num][1].item())
fig=plt.figure(0)
fig=plt.imshow(list_data_manual[img_num][0][0][0].numpy(), cmap='grey')
plt.show # maybe not need
fig.canvas.set_window_title(target)

# fig.canvas.set_window_title(list_data_manual[img_num].item()[1])





#-----------------ploting an image---------------


#----------------------ere we go1------------------
c=1
print (type(test_loader_manual))
while(c>=0):
    print("enter the index of image that you want to check in range of 0 ",len(test_dataset))
    c=int(input())
    print("c=",c)
    with torch.no_grad():

        pass
#----------------------ere we go1------------------

#----------------------
# test_loader_manual=torch.utils.data.DataLoader(dataset=test_dataset)
# print (type(test_loader_manual))
# print (type(test_loader_manual))
# while(c>=0):
#     print("enter the index of image that you want to check in range of 0 ",len(test_dataset))
#     c=int(input())
#     with torch.no_grad():
#         pass

#----------------------



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

# -----------saving parameters-----------
# print("want to save parametes? Y/N")
print("want to save parametes? 1 for Yes / 0 for No")
# answer=str(input())
answer=int(input())


# save_folder="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1"
# file="Cnn1V1Parameters"
if(answer):
    torch.save(myModel.state_dict(), os.path.join(save_folder, file))

#-----------saving parameters-----------


print("enter negative value for exit")
c=0
test_loader_manual=torch.utils.data.DataLoader(dataset=test_dataset)





#---------ere we go 2
print (type(test_loader_manual))
while(c>=0):
    print("enter the index of image that you want to check in range of 0 ",len(test_dataset))
    c=int(input())
    with torch.no_grad():
        pass
#---------ere we go 2


# print(myModel.parameters())
# print("parameters in first 0 tensor of parameters ", list(myModel.parameters())[0].size())
# print("------------------------------------")
# print("Ere we go 0! : list(myModel.parameters())[0] ", list(myModel.parameters())[0])
# print("------------------------------------")
# print("parameters in second 1 tensor of parameters ", list(myModel.parameters())[1].size())
# print("Ere we go 1! : list(myModel.parameters())[1] ", list(myModel.parameters())[1])

print("end")



















