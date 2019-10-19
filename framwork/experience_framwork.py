import torch
import tqdm
import os


class Experiment(object):
    """实验的实例化类

    通过该类，可以进行实验过程配置的封装。

    Attributes:
        model:  模型
        optimizer: 优化函数
        criterion: 损失函数
        batch_size: 每个批次的大小
        epoch: 迭代几次
        device: 使用哪个设备进行运行
    """

    def __init__(self, model,
                 optimizer, criterion, batch_size, epoch, device):
        """
        初始化函数
        :param model: 需要训练的模型
        :param optimizer: 优化器
        :param criterion: 损失函数
        :param batch_size: 每个批次的大小
        :param epoch: 迭代几次
        :param device: 需要运行的设备
        """
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion = criterion

    def __iteration(self, epoch, data_loader, train=True, log_freq=10):
        """
        训练和测试的迭代过程
        :param epoch: 当前迭代次数
        :param data_loader: 数据集
        :param train: 是否是训练过程
        :return: None
        """
        str_code = "train" if train else "test"

        if train:
            self.model.train()
        else:
            self.model.eval()

        # 设置一个进度条函数
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        # 训练和测试的具体过程
        for i, (data, target) in data_iter:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model.forward(data)
            loss = self.criterion(output, target)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            correct = output.argmax(dim=-1).eq(target).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += target.nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i != 0 and i % log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def test_model(self, data_loader):
        """
        测试模型
        :param data_loader: 测试集
        :return: None
        """
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct += pred.eq(target.view_as(pred)).sum().item()
        print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(data_loader.dataset),
                100. * correct / len(data_loader.dataset)))

    def fit_model(self, train_loader, evaluation_loader, save_model=True, log_freq=10,
                  model_path="output/", model_name="trained.model"):
        """
        训练并评估过程
        :param train_loader: 训练集
        :param evaluation_loader: 评估集
        :param save_model: 是否保存模型
        :param log_freq: 刷新频率
        :param model_path: 保存模型路径
        :param model_name: 保存模型名
        :return: None
        """
        for epoch in range(1, self.epoch + 1):
            self.__iteration(epoch, train_loader, log_freq=log_freq)
            self.__iteration(epoch, evaluation_loader, train=False, log_freq=log_freq)
            if save_model is True:
                self.save_model(epoch, model_path, model_name)
            print("\n")

    def save_model(self, epoch, file_path="output/", file_name="trained.model"):
        """
        保存模型
        :param epoch: 当前迭代次数
        :param file_path: 保存的路径
        :param file_name: 保存的文件名
        :return: None
        """
        if os.path.exists(file_path):
            pass
        else:
            os.mkdir(file_path)
        output_path = file_path+file_name + ".ep%d" % epoch
        torch.save(self.model.state_dict(), output_path)  # 保存模型的参数
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load_model(self, model_path):
        """
        加载模型
        :param model_path: 模型的路径
        :return: None
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
