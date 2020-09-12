import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models import Net as Net
# from models import AlexNet as Net
# from models import VGGNet16 as Net
# from models import Resnet18 as Net
# from models import Resnet34 as Net
# from models import Resnet50 as Net


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 4
test_batch_size = 4
input_size = 28

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
    epochs = 30


    # train data
    model.train()
    all_train_correct = []
    all_test_correct = []
    all_train_loss = []
    all_test_loss = []
    for epoch in range(epochs):
        train_correct = 0
        train_loss = 0
        print(len(train_loader))
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero to gradient
            output = model(data)
            loss = F.cross_entropy(output, target) # loss
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss += loss.item()
            if batch_idx % 50 == 0:
                print("train Epochs %d %d/%d loss %.6f"%(epoch, batch_idx, len(train_loader), loss.item()))
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for batch_idx,(data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        print("Test average loss %.6f test correct %.6f train correct %.6f"%(test_loss/len(test_loader), test_correct*1.0/(len(test_loader)*test_batch_size), train_correct*1.0/(len(train_loader)*batch_size)))
        torch.save(model.state_dict(),"model/model_cnn_epoch_"+str(epoch)+'_'+str(test_correct*1.0/(len(test_loader)*test_batch_size))+".pt")
        all_train_correct.append(train_correct*1.0/(len(train_loader)*batch_size))
        all_test_correct.append(test_correct*1.0/(len(test_loader)*test_batch_size))
        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)
        print('all_train_correct', all_train_correct)
        print('all_test_correct', all_test_correct)
        print('all_train_loss', all_train_loss)
        print('all_test_loss', all_test_loss)

def draw():
    import matplotlib.pyplot as plt
    all_train_correct = [0.6788666666666666, 0.78755, 0.8122666666666667, 0.8195333333333333, 0.8281, 0.838, 0.84115, 0.8458166666666667, 0.8478833333333333, 0.8485833333333334, 0.8508, 0.8526666666666667, 0.8528833333333333, 0.8556833333333334, 0.8557333333333333, 0.8598, 0.8596666666666667]
    all_test_correct = [0.7744, 0.7953, 0.8219, 0.8379, 0.838, 0.8444, 0.8462, 0.8422, 0.8516, 0.8579, 0.8626, 0.8541, 0.8471, 0.8533, 0.8611, 0.8578, 0.8555]
    all_train_loss = [14319.63852810234, 9363.197002579858, 8285.029175705473, 7820.660119522956, 7503.216359365597, 7140.601851054889, 6969.394840877674, 6737.789949233205, 6682.2538625822335, 6626.132391326902, 6554.214370633703, 6409.276927270399, 6430.483607136456, 6288.054740178739, 6291.715443706668, 6147.085014998472, 6122.345562560802]
    all_test_loss = [1681.7422, 1508.3678, 1286.3157, 1194.9720, 1175.6904, 1110.8330, 1098.5044, 1163.5863, 1084.9124, 1030.0759, 996.7437, 1034.5934, 1079.7917, 1051.9266, 1052.6558, 986.8037, 1050.8832]
    all_train_loss = [i/15000 for i in all_train_loss]
    all_test_loss = [i/5000 for i in all_test_loss]
    epochs = range(len(all_train_correct))
    plt.figure()
    plt.plot(epochs, all_train_correct, 'r-', label="train_correct")
    plt.plot(epochs, all_test_correct, '-', label="test_correct")
    #plt.plot(epochs, all_train_loss, '-', label="train_loss")
    #plt.plot(epochs, all_test_loss, '-', label="test_loss")
    plt.title('correct')
    plt.legend(loc="lower right")
    plt.show()


# train()
# draw()
