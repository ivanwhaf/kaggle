import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms, utils

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


batch_size = 3000
epochs = 100
lr = 0.001


class Net(nn.Module):
    """
    customized neural network
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(25, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.tanh(x)
        return out


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(25, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        out = self.tanh(x)
        return out


def load_dataset():
    train_data = pd.read_csv("pubg/train_V2.csv")
    print('train data describe:', train_data.describe())
    print('train data skew:', train_data.skew())
    print('train data null:', train_data.isnull().any())

    # ===== preprocess data =====

    # fill NaN value
    train_data['winPlacePerc'].fillna(value=0, inplace=True)
    # print(train_data['matchType'].value_counts())

    # label encoder -- 'matchType'
    le = LabelEncoder()
    le = le.fit(train_data['matchType'].unique())
    matchType = le.transform(train_data['matchType'])
    train_data['matchType'] = matchType

    # select features -- 25
    X_train = train_data[['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
                          'killStreaks', 'longestKill', 'matchDuration', 'matchType', 'maxPlace', 'numGroups', 'rankPoints', 'revives',
                          'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]
    y = train_data['winPlacePerc']

    # normalize train data
    minMaxScaler = MinMaxScaler()
    minMaxScaler.fit(X_train)
    # standardScaler = StandardScaler()
    # standardScaler.fit(X_train)

    X_train = minMaxScaler.transform(X_train)
    # X_train=standardScaler.transform(X_train)

    # convert dataframe to numpy ndarray then to pytorch tensor
    # X = torch.tensor(X_train.values.astype('float32'))
    # y = torch.tensor(y.values.astype('float32'))
    X = torch.tensor(X_train.astype('float32'))
    y = torch.tensor(y.astype('float32'))

    # trainset and testset
    train_X = X[:4000000]
    train_y = y[:4000000]
    val_X = X[4000000:]
    val_y = y[4000000:]

    print('train_X:', train_X)
    print('train_y:', train_y)

    # build dataset from dataset
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)

    # buile dataloader from dataset
    train_loader = DataLoader(

        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train(model, train_loader, optimizer, epoch, device, train_loss_lst):
    model.train()  # Set the module in training mode
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate loss and back prop
        outputs = model(inputs)
        outputs = outputs.squeeze(-1)
        loss = F.mse_loss(outputs, labels)  # mean square error loss
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()

        # print loss and accuracy every 100 iter
        if(batch_idx+1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss_lst.append(loss.item())
    return train_loss_lst


def validate(model, val_loader, device, val_loss_lst):
    model.eval()  # Set the module in evaluation mode
    val_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # add one batch loss
            output = output.squeeze(-1)
            val_loss += F.mse_loss(output, target, reduction='sum').item()

    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}\n'.format(val_loss))
    val_loss_lst.append(val_loss)
    return val_loss_lst


def plot_loss(epochs, train_loss_lst, val_loss_lst):
    # plot loss and accuracy
    fig = plt.figure()
    plt.plot(range(epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(epochs), val_loss_lst, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    plt.savefig('pubg/output/'+now + '.jpg')
    plt.show()


def predict(device):
    # predict
    model = torch.load("pubg/output/pubg.pth").to(device)

    test_data = pd.read_csv("pubg/test_V2.csv")
    # ===== preprocess data =====

    # label encoder -- 'matchType'
    le = LabelEncoder()
    le = le.fit(test_data['matchType'].unique())
    matchType = le.transform(test_data['matchType'])
    test_data['matchType'] = matchType

    # select feature -- 25
    X_test = test_data[['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
                        'killStreaks', 'longestKill', 'matchDuration', 'matchType', 'maxPlace', 'numGroups', 'rankPoints', 'revives',
                        'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]

    # convert dataframe to numpy values then to pytorch tensor
    X = torch.tensor(X_test.values.astype('float32')).to(device)

    # trainset and testset
    outputs = model(X)

    print(outputs)
    print(outputs.shape)
    outputs = outputs.squeeze(-1)

    output = pd.DataFrame(
        {'Id': test_data.Id, 'winPlacePerc': predictions})

    output.to_csv('pubg/submission.csv', index=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    train_loader, val_loader = load_dataset()

    net = Net2().to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train_loss_lst, val_loss_lst = [], []

    # train and validate
    for epoch in range(epochs):
        train_loss_lst = train(net, train_loader, optimizer,
                               epoch, device, train_loss_lst)
        val_loss_lst = validate(net, val_loader, device, val_loss_lst)

    plot_loss(epochs, train_loss_lst, val_loss_lst)

    # save model
    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    torch.save(net, "pubg/output/pubg_"+now+".pth")

    # predict(device)
