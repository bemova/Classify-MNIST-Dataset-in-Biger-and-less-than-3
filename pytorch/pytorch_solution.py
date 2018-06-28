import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils as utils
import torchvision
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import argparse
import time

transforms = transforms.Compose([transforms.Resize(14), transforms.ToTensor()])

dataset = torchvision.datasets.MNIST(root="pytorch_data", transform=transforms, download=True, train=True)
train_data = utils.data.DataLoader(dataset, shuffle=True, batch_size=32, num_workers=2)

test_dataset = torchvision.datasets.MNIST(root="pytorch_data", transform=transforms, download=True, train=False)
test_data = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=2)


class OneLayerModel(nn.Module):
    def __init__(self):
        super(OneLayerModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=2, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(96, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class TwoLayersModel(nn.Module):
    def __init__(self):
        super(TwoLayersModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=2, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, stride=2, padding=2)
        self.fc1 = nn.Linear(144, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class ThreeLayersModel(nn.Module):
    def __init__(self):
        super(ThreeLayersModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=2, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=2)
        self.fc1 = nn.Linear(288, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


# Convert Labels
def convert_labels(l_tensor):
    """
    this method is responsible for converting input vector for example from
    [ 3, 2, 0, 2, 7, 1, 6, 8, 1, 0, 4, 7, 0, 0, 1, 0, 5, 8, 1, 4, 5, 3, 9, 0, 9, 3, 6, 6, 4, 1, 4, 5]
    to
    [ 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0].
    in this method if the label is used for showing the handwriting number which is less than three, we convert it to 1.
    otherwise, we convert it to 0 which means the label is responsible for a handwritten number which is greater than or equal to three.
    :param l_tensor: the labels tensor for a batch size.
    :return: the converted label as a numpy array. in order to be able to have binary classification.
    """
    result = np.zeros((l_tensor.shape[0]), dtype=np.long)
    for index, data in enumerate(l_tensor.data.numpy(), 0):
        if data <= 2:
            result[index] = 1
        else:
            result[index] = 0
    return result


def model_test(model, test_data):
    """
    it is responsible for testing the model
    :param model: CNN model
    :param test_data: test dataset. in this example we are using MNIST test dataset.
    :return: accuracy of the model on the test dataset.
    """
    correct = 0
    total = 0
    for (images, labels) in test_data:

        outputs = model(images)
        labels = torch.from_numpy(convert_labels(labels))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (accuracy))
    print('Testing is Done!')
    return accuracy


def train(net):
    """
    this method is responsible to train and test the model with one or two or three convolutional layers
    :param net: model which is ThreeLayersModel or TwoLayersModel or OneLayerModel
    :return: None
    """
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # augmuntation with blur filter on the batch.
    conv0 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
    conv0.weight.data = torch.tensor([[[[0.0625, 0.1250, 0.0625],
                                        [0.1250, 0.2500, 0.1250],
                                        [0.0625, 0.1250, 0.0625]]]])
    conv0.bias.data = torch.tensor([0.0])

    for epoch in range(30):
        print("Epoch: {}".format(epoch + 1))
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_data, 0):
            labels = Variable(torch.from_numpy(convert_labels(labels)))
            inputs = conv0(inputs)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Finished Training')
    end = time.time()
    print("Training time is: {}".format(end - start))
    test_start_time = time.time()
    model_test(net, test_data)
    test_end_time = time.time()
    print("Testing time is: {}".format(test_end_time - test_start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="classifies the MNIST dataset into two classes (less than 3 and equal or greater than 3")

    parser.add_argument("--con_layer_count", help="the number of convolutional layers which is needed", default=3, type=int, required=False)

    args = parser.parse_args()

    if args.con_layer_count == 1:
        print("################# Training is started for one convolutional layer #################")
        net_1 = OneLayerModel()
        train(net=net_1)
    if args.con_layer_count == 2:
        print("################# Training is started for two convolutional layers #################")
        net_2 = TwoLayersModel()
        train(net=net_2)
    elif args.con_layer_count == 3:
        print("################# Training is started for three convolutional layers #################")
        net_3 = ThreeLayersModel()
        train(net=net_3)

