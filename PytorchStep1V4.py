# Absolut Halal
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/






#----------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

train_dataset=dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset=dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

print("Please enter the size of a batch:")
# batch_size=int(input())
batch_size=6
print("Enter the number of epochs for learn:")
# epochs=int(input())
epochs=40
#-------------
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
#-------------

#------plot image----not even to close to Slaanesh beauty----
print("print(train_loader.shape)")
print(train_loader)
images_in_batch,labels_in_batch=iter(train_loader).next()
print(images_in_batch)
print(type(images_in_batch))
print(labels_in_batch)
print(type(labels_in_batch))
image_plt =images_in_batch[0]
print("print(image_plt)")
print(image_plt)
print("print(labels_in_batch[0])")
print(labels_in_batch[0])

label_type=labels_in_batch[0]
print(type(labels_in_batch[0]))
print(type(label_type))
# print (label_type.data[0])
fig=plt.figure(0)

number=str(labels_in_batch[0].item())
print(number)
print(type(number))
fig.canvas.set_window_title(number)
fig=plt.imshow(image_plt.numpy()[0],cmap='gray')

# fig=plt.figure(0)
# fig.canvas.set_window_title(labels_in_batch[0])
# fig.canvas.set_window_title('torpedo')
# plt.show()
# plt.draw()
plt.ion()
plt.show()

#-----------------------------


class ForwardNetModel(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim):
        super(ForwardNetModel,self).__init__()
        self.fullyConnectedLayer1=nn.Linear(input_dim,hidden_dim)
        self.sigmoid=nn.Sigmoid()
        self.fullyConnectedLayer2=nn.Linear(hidden_dim,output_dim)
        #fc->f(x)->fc->out
    def forward(self,x):
        out=self.fullyConnectedLayer1(x)
        out=self.sigmoid(out)
        out=self.fullyConnectedLayer2(out)
        return out
# input Dimension Widht and Height in pixels of one image 28x28=784 - flattened 2d array by my OWN SLAANESH!
# output dimension ~ number of classes at my task - 10 types of handwritten images
# Hidden dimension - tuned\custom parameter
input_dim=int(28*28)
hidden_dim=int(input_dim/4)
output_dim=10 # number of classes in this task
netModel=ForwardNetModel(input_dim,hidden_dim,output_dim)
#-----------------------------------
#type of cost function or at normies's language - "criterion"
cost_func=nn.CrossEntropyLoss()
#-----------------------------
learning_rate=0.1
optimizer=torch.optim.SGD(netModel.parameters(),lr=learning_rate)
#----------------------just for sake show what kind of type these objects are
print("Model.parameters")
print(netModel.parameters)
print("Model.parameters()")
print(netModel.parameters())

#----------------------just for sake show what kind of type these objects are
print("\t---------")
print("len(list(netModel.parameters()))")
print(len(list(netModel.parameters()))) #   if print(len(list(netModel.parameters))) -TypeError: 'method' object is not iterable
#----------------------------Slaanesh approves it
# size of first layer parameters -"Weights", without "bias" "vector"
print("print(list(netModel.parameters())[0].size())")
print(list(netModel.parameters())[0].size()) #size of first= hidden neurons, second=input neurons
# size of first layer parameters - "Biases", which is exactly the size of hidden layer neurons
print(list(netModel.parameters())[1].size())
# second (last) layer parameters
# weights of connections, size of first num - neurons in hidden, second num - count of neurons in output
print(list(netModel.parameters())[2].size())
# biases of output layer, size is equal the number of neurons in that layer
print(list(netModel.parameters())[3].size())
#-------------before train process, but the same data structures
# trying to draw a picture from these data structures
# val=0
# for images, labels in test_loader:  # in test loader!!!!
#     # images = images.view(-1, 28 * 28).requires_grad_()
#     images = images.view(-1, 28 * 28)
#     label_n=str(labels[val].item())
#     # tmp_img=images[val].view(28, 28).requires_grad_()
#     tmp_img=images[val].view(28, 28)
#     fig = plt.figure(0)
#     fig.canvas.set_window_title(label_n)
#     fig=plt.imshow(tmp_img.numpy(),cmap='gray')
#     plt.show()

#-------------training process-----------------------
print("train Process")
WTF=test_loader
print(type(WTF))
# print("TEST LOADER test_loader.size()=", test_loader.size(0)) # we dont need it boy, cause it accum anyway
iter = 0
for epoch in range(epochs):#num_epochs
    for i, (images,labels) in enumerate(train_loader): # in train loader !!!
        images=images.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
        output=netModel(images)
        error=cost_func(output,labels)
        error.backward()
        optimizer.step()
        iter+=1
#------------test----------
        if(iter%500):
            total=0
            correct=0
            img_i=0 #iterator for indexing plotting images from test set
            for images,labels in test_loader: # in test loader!!!!

                #-------draw the current image and its label from test loader----------
                print("----------follow me ladz---------")
                # t_plt=images.data[img_i]
                t_plt=images[img_i].data
                t_plt_label=str(labels[img_i].item())
                fig=plt.figure(0)
                fig.canvas.set_window_title(t_plt_label)
                fig=plt.imshow(t_plt.numpy()[0], cmap='gray')
                plt.show()
                #-------draw the current image and its label from test loader----------
                # print("labels.data")
                # print(labels.data)
                # print("labels.data[0]")
                # print(labels.data[0])
                # print("labels.data[0].data")
                # print(labels.data[0].data)
                # outputs_test=netModel(images)
                # print("outputs_test")
                # print(outputs_test)
                # print("outputs_test.data")
                # print(outputs_test.data)
                # print("outputs_test.data[0]")
                # print(outputs_test.data[0])
                #-----checker-----------
                img_i+=1
                #-----------------------

                images = images.view(-1, 28 * 28).requires_grad_()
                # accum=0.0
                # total=len(labels)
                print("total=len(labels)=", total)
                total=labels.size(0)
                print("total=labels.size(0)=",total)
                out_test=netModel(images)
                _, out_test=torch.max(out_test.data, 1)
                print(labels[0].item())

                # -----checker-----------

                for index in range (len(labels)): #labels.size(0)
                    print("out_test.data[index].item()",out_test.data[index].item()) # works - scalar, class type
                    print("out_test.data[index]",out_test.data[index]) # tensor and value, but not desired output
                    print("out_test[index].item()",out_test[index].item()) # works - scalar, class type
                    value=int(out_test.data[index].item()==labels[index].item())
                    total+=1
                    correct+=value
            print("total error ",(accum/total))
















##----------------------------------------------------
# import matplotlib
# # %matplotlib inline
# import numpy as np
# import matplotlib.pyplot as plt
# from six.moves import cPickle
#
# f = open('data/cifar10/cifar-10-batches-py/data_batch_1', 'rb')
# datadict = cPickle.load(f,encoding='latin1')
# f.close()
# X = datadict["data"]
# Y = datadict['labels']
# X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
# Y = np.array(Y)
#
# #Visualizing CIFAR 10
# fig, axes1 = plt.subplots(5,5,figsize=(3,3))
# for j in range(5):
#     for k in range(5):
#         i = np.random.choice(range(len(X)))
#         axes1[j][k].set_axis_off()
#         axes1[j][k].imshow(X[i:i+1][0])



##----------------------------------------------------
# import torch
# import torchvision
# import torchvision.transforms as transfroms

# trainset=torchvision.datasets.CIFAR100(root='./data', download=True, transfrom=transfroms.ToTensor())
# trainset=torchvision.datasets.CIFAR100(root='C:\Users\user\Desktop\pytorch\data\faces', download=True, transfrom=transfroms.ToTensor())
# trainset=torchvision.datasets.CIFAR100(root='./data', download=True, transfrom=transfroms.ToTensor())






















##----------------------------------------------------
# import torch
# import torchvision
#
# n_epochs=3
# batch_size_train=64
# batch_size_test=1000
# learning_rate=0.01
# momentum=0.5
# log_interval=10
#
# random_seed=1
# torch.backends.cudnn.enabled=False
# torch.manual_seed(random_seed)



##----------------------------------------------------


# import torch
# from torch.autograd import Variable
#
# x_data = [1.0, 2.0, 3.0]
# y_data = [2.0, 4.0, 6.0]
# lrate=0.01
#
# w=Variable(torch.Tensor([1.0]), requires_grad=True)
#
# def forward(x):
#     return x*w
# def loss(x,y):
#     y_pred=forward(x)
#     return (y_pred-y)*(y_pred-y)
# # print("predict (before training)", 4, forward(4).data[0])
# print("predict (before training)", 4, forward(4))
# print("predict (before training)", 4, forward(4).data[0])
# criterion=torch.nn.MSELoss(size_average=False)
# optimizer=torch.optim.SGD(model.parameters) # we should initialise class neural network
#
# #
# for epoch in range(100):
#     for x_val, y_val in zip(x_data,y_data):
#         l=loss(x_val, y_val)
#         l.backward()
#         print("\tgrad: ", x_val,y_val,w.grad.data[0])
#         w.data=w.data-lrate*w.grad.data
#         w.grad.data.zero_()
#     print("progress:", epoch, l.data[0])
# print("predict (after training)", 4, forward(4).data[0])
# #




        #
        # print("w.grad.data[0]=",w.grad.data[0])
        # print("w=",w)
        # print("w.grad=",w.grad)
        # print("w.grad.data[0]=",w.grad.data[1])

##----------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 3x3 square convolution
#         # kernel
#         # self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv1 = nn.Conv2d()
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#
#
# net = Net()
# # print(net)
# params=list(net.parameters())
# # print(len((net.parameters()))) #TypeError: object of type 'generator' has no len()
# print(len(params))

##-------------------------------------------------
# import torch
#
#
# a=torch.rand(2,2)
# print("a=",a)
# a1=(a*3)
# a2=(a-1)
# print("a1=",a1)
# print("a2=",a2)
# a3=a1/a2
# print("(a*3)/(a-1)=",a3)

#----------------------------------------------------
#
# from __future__ import print_function
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#
# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))
#
#
# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#
#
# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#
#     parser.add_argument('--save-model', action='store_true', default=False,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#
#     torch.manual_seed(args.seed)
#
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#         batch_size=args.test_batch_size, shuffle=True, **kwargs)
#
#     model = Net().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(args, model, device, test_loader)
#
#     if (args.save_model):
#         torch.save(model.state_dict(), "mnist_cnn.pt")
#
#
# if __name__ == '__main__':
#     main()
