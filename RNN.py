import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

class RNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(RNN, self).__init__()
        self.w_xh = nn.Parameter(torch.rand(in_size, hidden_size))
        self.w_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.w_hy = nn.Parameter(torch.rand(hidden_size, out_size))
        self.hidden_size = hidden_size

    def forward(self, x, bz):
        h = F.tanh(torch.mm(x,self.w_xh) + torch.mm(bz,self.w_hh))
        y = torch.mm(h,self.w_hy)
        return y, h

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

class RandomData(Dataset):
    def __init__(self, data_size=10000000, in_size=1, batch_size=16):
        self.x = torch.rand(data_size, 1, in_size)
        #self.x = self.x.round().reshape(data_size//10, 10)
        #self.labels = torch.fmod(self.x.sum(axis=1), 2).long()
        self.labels = torch.zeros(data_size, 1).long()
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        idx = min(idx, len(self.x)-self.batch_size)
        return self.x[idx:idx+self.batch_size], self.labels[idx:idx+self.batch_size]
            

def train_rnn():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 16
    epochs = 5
    lr = 0.01
    momentum = 0.5
    input_size = 3
    output_size = 5
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    rnn = RNN(input_size, 32, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=lr, momentum=momentum)
    train_data = RandomData(10000, input_size, batch_size)
    hidden = rnn.init_hidden().to(device)
    for epoch in range(epochs):
        for idx in range(len(train_data)):
            x_in, label = train_data[idx]
            x_in, label = x_in.to(device), label.to(device)
            output = torch.rand(batch_size, output_size)	
            for i in range(x_in.size()[0]):	
                y,hidden = rnn(x_in[i], hidden)	
                output[i] = y
            optimizer.zero_grad()
            label = label.squeeze()
            loss = criterion(output, label)
            print(loss)
            loss.backward(retain_graph=True)
            optimizer.step()
            if idx % 50 == 0:
                print("train Epochs %d %d/%d loss %.6f"%(epoch, idx, len(train_data), loss.item()))


def test_rnn():
    h = torch.zeros(1, 5)
    rnn = RNN(3, 5, 2)
    for i in range(10):
        x = torch.rand(1, 3)
        h, y = rnn(x, h)
        print(i, y)
        

train_rnn()
        
