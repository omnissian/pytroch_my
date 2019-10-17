# Absolut Halal
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

train_dataset=dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset=dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

print("Please enter the size of a batch:")
# batch_size=int(input())
batch_size=3000
print("Enter the number of epochs for learn:")
# epochs=int(input())
epochs=1
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
print (label_type.item())
fig=plt.figure(0)
number=labels_in_batch[0].item()
print(type(number))
print("int number=",number)
str_number=str(number)
print("str_number",str_number)

fig=plt.imshow(image_plt.numpy()[0],cmap='gray')
fig=plt.figure(0)
# fig.canvas.set_window_title(labels_in_batch[0])
fig.canvas.set_window_title(str_number)

fig.plt
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

#-------------training process-----------------------
print("train Process")
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
    total=0
    correct=0
    for images,labels in test_loader: # in test loader!!!!
        images = images.view(-1, 28 * 28).requires_grad_()
        print("labels.data")
        print(labels.data)
        print("labels.data[0]")
        print(labels.data[0])
        print("labels.data[0].data")
        print(labels.data[0].data)
        outputs_test=netModel(images)
        print("outputs_test")
        print(outputs_test)
        print("outputs_test.data")
        print(outputs_test.data)
        print("outputs_test.data[0]")
        print(outputs_test.data[0])

        _, out_test=torch.max(outputs_test.data, 1)

