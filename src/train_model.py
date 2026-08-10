import os
import sys
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '动物识别数据集', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '训练输出')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES_ZH = ['猫', '牛', '鸡', '狗', '鸭', '狮子', '羊', '虎']
CLASSES_EN = ['cat', 'cattle', 'chicken', 'dog', 'duck', 'lion', 'sheep', 'tiger']

HYPER_PARAMS = {
    'learning_rate': [0.001, 0.0005],
    'batch_size': [16, 32]
}
EPOCHS_PER_TRIAL = 10
NUM_WORKERS = 0

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def verify_dataset():
    """预处理数据集并进行质量验证"""
    print('\n' + '=' * 60)
    print('  数据集预处理与质量验证')
    print('=' * 60)

    corrupt_files = []
    stats = {}
    total_valid = 0

    for phase in ['train', 'val']:
        phase_dir = os.path.join(DATA_DIR, phase)
        if not os.path.exists(phase_dir):
            print(f'  [警告] 目录不存在: {phase_dir}')
            continue

        for cls_en in CLASSES_EN:
            cls_dir = os.path.join(phase_dir, cls_en)
            if not os.path.exists(cls_dir):
                continue

            valid_count = 0
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    continue

                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    with Image.open(fpath) as img:
                        w, h = img.size
                        if w < 100 or h < 100:
                            corrupt_files.append((fpath, f'分辨率过低 {w}x{h}'))
                            continue
                        mode = img.mode
                        if mode not in ('RGB', 'RGBA'):
                            corrupt_files.append((fpath, f'色彩模式异常 {mode}'))
                            continue
                    valid_count += 1
                except Exception as e:
                    corrupt_files.append((fpath, str(e)))

            key = cls_en
            if key not in stats:
                stats[key] = {}
            stats[key][phase] = valid_count
            total_valid += valid_count

    print(f'\n  有效文件总数: {total_valid}')
    print(f'  损坏文件数: {len(corrupt_files)}')
    return stats


def get_dataloaders(batch_size):
    """构建数据预处理管道与数据加载器"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=NUM_WORKERS)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes


def build_model(num_classes=8):
    """构建基于 ResNet18 的迁移学习模型"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)


def train_and_evaluate(lr, batch_size, epochs=None):
    """执行模型训练与验证"""
    if epochs is None:
        epochs = EPOCHS_PER_TRIAL

    dataloaders, dataset_sizes = get_dataloaders(batch_size)
    model = build_model(num_classes=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs} | lr={lr} batch_size={batch_size}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_corrects = 0
            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_corrects += torch.sum(preds == labels.data)
                running_loss += loss.item() * inputs.size(0)

            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
                    print(f'  [保存] 验证集准确率创新高: {best_acc:.4f}')

    return best_acc, val_acc_history, train_loss_history, val_loss_history


if __name__ == '__main__':
    print('=' * 60)
    print('  基于CNN的常见动物种类识别系统 - 训练脚本')
    print('=' * 60)
    print(f'当前设备: {DEVICE}')

    verify_dataset()

    keys, values = zip(*HYPER_PARAMS.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    global_best_acc = 0.0

    for params in param_combinations:
        lr = params['learning_rate']
        batch_size = params['batch_size']
        cfg_name = f"lr={lr}, bs={batch_size}"

        print(f'\n>>> 开始训练配置: {cfg_name}')
        best_acc, _, _, _ = train_and_evaluate(lr=lr, batch_size=batch_size)

        if best_acc > global_best_acc:
            global_best_acc = best_acc

    print(f'\n>>> 训练完成，全局最佳验证准确率: {global_best_acc:.4f}')
    print(f'>>> 最佳模型已保存至: {os.path.join(OUTPUT_DIR, "best_model.pth")}')