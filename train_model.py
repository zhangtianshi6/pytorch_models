import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models import AlexNet as Net
# from models import VGGNet16 as Net
# from models import Resnet18 as Net
# from models import Resnet34 as Net
# from models import Resnet50 as Net


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 4
test_batch_size = 4
input_size = 224

def load_data():

    pin_memory = True if use_cuda else False
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', \
        train=True, download=True, transform=transforms.Compose([\
        transforms.RandomResizedCrop(input_size), transforms.ToTensor()])),\
        batch_size = batch_size, shuffle=True, num_workers=1, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', \
        train=False, transform=transforms.Compose([\
        transforms.RandomResizedCrop(input_size), transforms.ToTensor()])),\
        batch_size = test_batch_size, shuffle=True, num_workers=1, pin_memory=pin_memory)
    return train_loader, test_loader

def train():
    # load dataset
    train_loader, test_loader = load_data()
    channels = 1
    class_num = 10
    # params
    lr = 0.01
    momentum = 0.5
    model = Net(channels, class_num).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    epochs = 300


    # train data
    model.train()
    for epoch in range(epochs):
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero to gradient
            output = model(data)
            loss = F.cross_entropy(output, target) # loss
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % 50 == 0:
                print("train Epochs %d %d/%d loss %.6f"%(epoch, batch_idx, len(train_loader), loss.item()))
        test_loss = 0
        test_correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            print("Test average loss %.6f test correct %.6f train correct %.6f"%(test_loss/len(test_loader), test_correct*1.0/len(test_loader*test_batch_size), train_correct*1.0/(batch_idx*batch_size)))
        torch.save(model.state_dict(),"model_cnn_epoch_"+str(epoch)+".pt")


train()
