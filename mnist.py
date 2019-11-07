#%%
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm 


#%%

train_data=MNIST("~/Data/mnist",train=True,download=True,transform=transforms.ToTensor())
train_loader=DataLoader(train_data,batch_size=1000,shuffle=True)

val_data=MNIST("~/Data/mnist",train=False,download=True,transform=transforms.ToTensor())
val_loader=DataLoader(val_data,batch_size=1000,shuffle=True)


dataloaders_dict={"train":train_loader,"val":val_loader}

# %%
class LSTMNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_dim):
        super(LSTMNet,self).__init__()
        self.rnn=nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=2,
                        batch_first=True)
        self.output_layers=nn.Linear(hidden_size,output_dim)

    def forward(self,inputs,hidden0=None):
        output,(hidden,cell)=self.rnn(inputs,hidden0)
        output=self.output_layers(output[:,-1,:])

        return output

net=LSTMNet(28,256,10)
print(net)

# %%
epoch_num=100
criterion=nn.MSELoss()
optimizer=optim.SGD(net.parameters(),lr=0.01)

device=torch.device("cuda:0"if torch.cuda.is_available() else"cpu")
print("device is ",device)
net.to(device)

for epoch in range(epoch_num):
    print("Epoch {}/{}".format(epoch+1,epoch_num))
    running_loss=0.0
    total=0
    correct=0
    training_accuracy=0.0
    for i,(data,labels) in enumerate(tqdm(train_loader)):
        data=data.to(device)
        labels=labels.to(device)
        onehot=torch.eye(10)[labels]
        onehot=onehot.to(device)
        data=data.view(-1,28,28)
        output=net(data)
        loss=criterion(output,onehot)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _,predicted=torch.max(output.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        if i%5==0:
            print("train_accuracy:{:.2f}%".format(100*float(correct/total)))
            print(correct)
            print(total)


        
    test_accuracy=0.0
    for i,(data,labels) in enumerate(tqdm(val_loader)):
        data=data.to(device)
        labels=labels.to(device)
        print("raw_labels_size:",labels.size())
        onehot=torch.eye(10)[labels]
        onehot=onehot.to(device)
        data=data.view(-1,28,28)
        output=net(data)
        print("inputs_size:",data.size())
        print("labels_size:",labels.size())
        print("outputs_size:",output.size())
        running_loss+=loss.item()
        _,predicted=torch.max(output.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        if i%5==0:
            print("val_accuracy:{:.2f}%".format(100*float(correct/total)))

torch.save(net.state_dict(),"./weight.pth")


        
        





