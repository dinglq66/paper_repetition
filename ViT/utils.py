import os
import sys
import json
import pickle
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，每个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存放训练集的所有图片路径
    train_images_label = []  # 存放训练集所有图片对应索引信息
    val_images_path = []  # 存放验证集的所有图片路径
    val_images_label = []  # 存放验证集的所有图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    supported = [".jpg", ".png", ".JPG", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获得supported支持的所有文件路径
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        images_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按照比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(images_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(images_class)

    print("{} images found in dataset".format(sum(every_class_num)))
    print("{} images for training".format(len(train_images_path)))
    print("{} images for validation".format(len(val_images_path)))

    plot_image = False  # 若为True，则展示数据集信息;若为False，则不显示
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0，1，2，3，4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v+5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


class MyDataSet(Dataset):
    """
    自定义数据集
    """
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图像，L为灰度图像
        if img.mode != 'RGB':
            raise ValueError("image:{} isn't RGB mode".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_fucntion = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累积损失
    accu_num = torch.zeros(1).to(device)  # 累积预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]  # 每次iteration都自加batch_size

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_fucntion(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss:{:.3f}, acc:{:.3f}".format(epoch,
                                                                             accu_loss.item() / (step+1),
                                                                             accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累积损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss:{:.3f}, acc:{:.3f}".format(epoch,
                                                                             accu_loss.item() / (step + 1),
                                                                             accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
