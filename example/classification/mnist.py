import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys

sys.path.append("../../")
from framwork.experience_framwork import Experiment


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 10, 24x24
        self.conv2 = nn.Conv2d(10, 20, 3)  # 128, 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


def get_data(batch_size):
    """
    获取训练和验证数据
    :param batch_size: 每个batch的大小
    :return: train_loader, test_loader
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_setting(cuda_index):
    """
    获取实验所需要的配置
    :param cuda_index: 决定使用哪个GPU进行实验
    :return: batch_size, epochs, device, model, optimizer, criterion
    """
    batch_size = 512  # 每一个batch的大小
    epochs = 5  # 总共训练批次
    device = torch.device(
        "cuda:" + cuda_index + "" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    model = ConvNet().to(device)  # 配置模型
    optimizer = optim.Adam(model.parameters())
    criterion = F.nll_loss
    return batch_size, epochs, device, model, optimizer, criterion


if __name__ == '__main__':
    cuda_index = sys.argv[1]  # 决定使用哪个GPU进行实验
    batch_size, epoch, device, model, optimizer, criterion = get_setting(cuda_index)  # 准备实验设置
    train_loader, test_loader = get_data(batch_size)  # 获取实验数据
    class_experiment = Experiment(model, optimizer, criterion, batch_size, epoch, device)  # 实例化实验
    class_experiment.fit_model(train_loader=train_loader, evaluation_loader=test_loader, log_freq=batch_size)  # 训练模型
    class_experiment.load_model(model_path="output/trained.model.ep5")
    class_experiment.test_model(data_loader=test_loader)
