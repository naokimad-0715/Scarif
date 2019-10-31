import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm 

#making datasets
train_data=MNIST("~/Data/mnist",train=True,download=True,transform=transforms.ToTensor())
train_loader=DataLoader(train_data,batch_size=1000,shuffle=True)

val_data=MNIST("~/Data/mnist",train=False,download=True,transform=transforms.ToTensor())
val_loader=DataLoader(val_data,batch_size=1000,shuffle=True)

dataloaders_dict={"train":train_loader,"val":val_loader}

#defining model
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
print("MODEL---------------------------------------------------------------------------------------------")
print(net)
print("--------------------------------------------------------------------------------------------------")

#train or validataion
epoch_num=100
criterion=nn.MSELoss()
optimizer=optim.SGD(net.parameters(),lr=0.01)

device=torch.device("cuda:0"if torch.cuda.is_available() else"cpu")
print("device is ",device)
net.to(device)

for epoch in range(epoch_num):
    print("Epoch {}/{}".format(epoch+1,epoch_num))
    print("---------------------------------------------------------------------------------------------")
    
    for phase in ["train","val"]:
        if phase=="train":
            net.train()
        else:
            net.eval()
        total=0
        epoch_loss=0.0
        epoch_correct=0

        if (epoch==0) and (phase=="train"):
            continue
        for data,number in tqdm(dataloaders_dict[phase]):
            
            data=data.to(device)
            inputs=data.view(-1,28,28).to(device)
            onehot=torch.eye(10)[number].to(device)
            labels=number.to(device)

            with torch.set_grad_enabled(phase=="train"):
                outputs=net(inputs)
                
                loss=criterion(outputs,onehot)
                _,predicted=torch.max(outputs.data,1)
                
                if phase=="train":
                    loss.backward()
                    optimizer.step()
                
                total+=labels.size(0)
                epoch_loss+=loss.item()*inputs.size(0)
                
                epoch_correct+=torch.sum(predicted==labels)
                
        epoch_loss=epoch_loss/len(dataloaders_dict[phase].dataset)
        epoch_acc=epoch_correct.double()/total

        print("phase:{},Loss:{:.4f},Accuracy:{:.4f}".format(phase,epoch_loss,epoch_acc))

torch.save(net.state_dict(),"./weight.pth")