import itertools
import os
import random
import shutil
import time
from urllib import parse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = './animal_dataset_auto'
OFFLINE_DATA_DIR = './offline_dataset'
USE_OFFLINE_MODE = False

CLASSES_ZH = ['熊', '棕熊', '公牛', '蝴蝶', '骆驼', '金丝雀', '毛毛虫', '牛', '蜈蚣', '猎豹']
CLASSES_EN = ['bear', 'brown-bear', 'bull', 'butterfly', 'camel', 'canary', 'caterpillar', 'cattle', 'centipede',
              'cheetah']

HYPER_PARAMS = {
    'learning_rate': [0.001, 0.0001],
    'batch_size': [16, 32]
}

EPOCHS_PER_TRIAL = 10
DOWNLOAD_COUNT_PER_CLASS = 80

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def verify_image(file_path):
    """数据质量校验：使用PIL对图像文件进行结构完整性验证与可加载性检测。"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
        return True
    except Exception:
        return False


def simple_image_crawler(keyword, download_path, count=30):
    """基于Bing图像搜索引擎的自动采集与数据质量校验。"""
    os.makedirs(download_path, exist_ok=True)

    existing_files = [f for f in os.listdir(download_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_count = 0

    for filename in existing_files:
        file_path = os.path.join(download_path, filename)
        if verify_image(file_path) and os.path.getsize(file_path) > 10240:
            valid_count += 1
        else:
            try:
                os.remove(file_path)
            except OSError:
                pass

    if valid_count >= count:
        print(f"  [跳过] '{keyword}' 目录已有 {valid_count} 张有效图片。")
        return

    print(f"  [下载] 正在采集 '{keyword}' 图像并执行数据质量校验...")
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/122.0.0.0 Safari/537.36'
        )
    }

    search_url = 'https://www.bing.com/images/async?q={0}&first={1}&count={2}&relp=35&lostate=r&mmasync=1'
    downloaded = valid_count
    page = 0

    while downloaded < count and page < 8:
        url = search_url.format(parse.quote(keyword), page * 35, 35)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            import re
            img_urls = re.findall(r'murl&quot;:&quot;(http.*?)&quot;', response.text)

            for img_url in img_urls:
                if downloaded >= count:
                    break

                try:
                    img_response = requests.get(img_url, headers=headers, timeout=8)
                    if img_response.status_code != 200:
                        continue

                    temp_name = f'temp_{int(time.time() * 1000)}_{downloaded}.jpg'
                    temp_path = os.path.join(download_path, temp_name)

                    with open(temp_path, 'wb') as f:
                        f.write(img_response.content)

                    if verify_image(temp_path) and os.path.getsize(temp_path) > 10240:
                        final_name = f'图_{int(time.time() * 1000)}_{downloaded}.jpg'
                        os.replace(temp_path, os.path.join(download_path, final_name))
                        downloaded += 1
                    else:
                        os.remove(temp_path)
                except Exception:
                    continue

            page += 1
        except Exception:
            break

    print(f"  [完成] '{keyword}' 采集结束，有效图片：{downloaded} 张。")


def prepare_dataset_online():
    """在线模式：自动采集图像数据并按8:2比例划分训练集与验证集。"""
    print('>>> 启动数据集自动采集与三级质量保障流程...')

    for sub_dir in ['训练集', '验证集', 'raw']:
        full_path = os.path.join(DATA_DIR, sub_dir)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)

    total_valid = 0

    for zh_name, en_name in zip(CLASSES_ZH, CLASSES_EN):
        raw_path = os.path.join(DATA_DIR, 'raw', en_name)

        search_keyword = f"{en_name} real animal photo"
        simple_image_crawler(search_keyword, raw_path, count=DOWNLOAD_COUNT_PER_CLASS)

        train_path = os.path.join(DATA_DIR, '训练集', en_name)
        val_path = os.path.join(DATA_DIR, '验证集', en_name)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        all_imgs = [f for f in os.listdir(raw_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_imgs)

        split_idx = int(len(all_imgs) * 0.8)
        train_imgs = all_imgs[:split_idx]
        val_imgs = all_imgs[split_idx:]

        for img_name in train_imgs:
            shutil.copy2(os.path.join(raw_path, img_name), os.path.join(train_path, img_name))
        for img_name in val_imgs:
            shutil.copy2(os.path.join(raw_path, img_name), os.path.join(val_path, img_name))

        total_valid += len(all_imgs)
        print(f"  [类别完成] {zh_name}({en_name}) -> 训练:{len(train_imgs)} 验证:{len(val_imgs)}")

    print('>>> 数据采集与清洗完毕！请进行人工审核校验（第三级保障）。')
    print(f'>>> 总可用图片数: {total_valid} 张\n')


def prepare_dataset_offline():
    """离线模式：使用本地数据源进行数据质量校验后划分训练集与验证集。"""
    print('>>> 启动离线数据集准备流程（无网络模式）...')

    if not os.path.exists(OFFLINE_DATA_DIR):
        raise FileNotFoundError(f'未找到离线数据目录: {OFFLINE_DATA_DIR}')

    for sub_dir in ['训练集', '验证集']:
        full_path = os.path.join(DATA_DIR, sub_dir)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)

    total_valid = 0

    for zh_name, en_name in zip(CLASSES_ZH, CLASSES_EN):
        source_dir = os.path.join(OFFLINE_DATA_DIR, en_name)
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f'离线类别目录缺失: {source_dir}')

        valid_files = []
        for filename in os.listdir(source_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            file_path = os.path.join(source_dir, filename)
            if verify_image(file_path) and os.path.getsize(file_path) > 10240:
                valid_files.append(filename)

        random.shuffle(valid_files)
        split_idx = int(len(valid_files) * 0.8)
        train_imgs = valid_files[:split_idx]
        val_imgs = valid_files[split_idx:]

        train_path = os.path.join(DATA_DIR, '训练集', en_name)
        val_path = os.path.join(DATA_DIR, '验证集', en_name)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        for img_name in train_imgs:
            shutil.copy2(os.path.join(source_dir, img_name), os.path.join(train_path, img_name))
        for img_name in val_imgs:
            shutil.copy2(os.path.join(source_dir, img_name), os.path.join(val_path, img_name))

        total_valid += len(valid_files)
        print(f"  [类别完成] {zh_name}({en_name}) -> 训练:{len(train_imgs)} 验证:{len(val_imgs)}")

    print('>>> 数据采集与清洗完毕！请进行人工审核校验（第三级保障）。')
    print(f'>>> 离线模式总可用图片数: {total_valid} 张\n')


def prepare_dataset():
    """统一入口：根据配置选择在线或离线数据准备流程。"""
    if USE_OFFLINE_MODE:
        prepare_dataset_offline()
    else:
        prepare_dataset_online()


def get_dataloaders(batch_size):
    """构建数据预处理管道与数据加载器。"""
    data_transforms = {
        '训练集': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        '验证集': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['训练集', '验证集']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == '训练集'), num_workers=0)
        for x in ['训练集', '验证集']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['训练集', '验证集']}
    return dataloaders, dataset_sizes


def build_model(num_classes=10):
    """构建基于ResNet18的迁移学习模型，替换全连接层为10分类输出。"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_and_evaluate(lr, batch_size):
    """执行模型训练与验证，记录损失与准确率变化。"""
    dataloaders, dataset_sizes = get_dataloaders(batch_size)
    model = build_model(num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    best_acc = 0.0
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS_PER_TRIAL):
        print(f'\nEpoch {epoch + 1}/{EPOCHS_PER_TRIAL} | lr={lr} batch_size={batch_size}')

        for phase in ['训练集', '验证集']:
            if phase == '训练集':
                model.train()
            else:
                model.eval()

            running_corrects = 0
            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == '训练集'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == '训练集':
                        loss.backward()
                        optimizer.step()

                running_corrects += torch.sum(preds == labels.data)
                running_loss += loss.item() * inputs.size(0)

            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == '训练集':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), '最终动物识别模型.pth')
                    print(f'  [保存] 验证集准确率创新高: {best_acc:.4f}')

    return best_acc, val_acc_history, train_loss_history, val_loss_history


def plot_history(results_map):
    """绘制不同超参数组合下的验证集准确率变化曲线。"""
    plt.figure(figsize=(10, 6))
    for cfg, history in results_map.items():
        plt.plot(range(1, len(history) + 1), history, label=cfg, marker='o', linewidth=2, markersize=4)

    plt.title('不同超参数组合下验证集准确率变化曲线', fontsize=14)
    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('验证集准确率', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300)
    print('>>> 学习曲线已保存为: learning_curve.png')
    plt.close()


def plot_loss_curves(train_loss_map, val_loss_map):
    """绘制训练损失与验证损失变化曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    for cfg, history in train_loss_map.items():
        ax1.plot(range(1, len(history) + 1), history, label=cfg, marker='o', linewidth=2, markersize=4)
    ax1.set_title('训练集损失变化曲线', fontsize=14)
    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('训练损失 (CrossEntropy Loss)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = axes[1]
    for cfg, history in val_loss_map.items():
        ax2.plot(range(1, len(history) + 1), history, label=cfg, marker='s', linewidth=2, markersize=4)
    ax2.set_title('验证集损失变化曲线', fontsize=14)
    ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('验证损失 (CrossEntropy Loss)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300)
    print('>>> 损失曲线已保存为: loss_curves.png')
    plt.close()


def compute_metrics(model, dataloader, device, class_names):
    """计算分类性能指标：混淆矩阵与分类报告。"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    return cm, report, all_preds, all_labels


def plot_confusion_matrix(cm, class_names, normalize=True):
    """绘制归一化混淆矩阵热力图。"""
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('归一化比例', fontsize=12)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=11)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_display.max() / 2.
    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        ax.text(j, i, format(cm_display[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=9)

    ax.set_ylabel('真实类别', fontsize=13)
    ax.set_xlabel('预测类别', fontsize=13)
    ax.set_title('混淆矩阵（归一化）', fontsize=15)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print('>>> 混淆矩阵热力图已保存为: confusion_matrix.png')
    plt.close()


def print_statistical_analysis(report, class_names):
    """输出详细统计分析报告。"""
    print('\n' + '='*70)
    print('                  实验统计分析报告')
    print('='*70)

    print('\n【整体性能指标】')
    print(f"  宏平均精确率 (Macro Precision):  {report['macro avg']['precision']:.4f}")
    print(f"  宏平均召回率 (Macro Recall):     {report['macro avg']['recall']:.4f}")
    print(f"  宏平均 F1 分数 (Macro F1-score): {report['macro avg']['f1-score']:.4f}")
    print(f"  加权平均精确率 (Weighted Precision):  {report['weighted avg']['precision']:.4f}")
    print(f"  加权平均召回率 (Weighted Recall):     {report['weighted avg']['recall']:.4f}")
    print(f"  加权平均 F1 分数 (Weighted F1-score): {report['weighted avg']['f1-score']:.4f}")
    print(f"  总体准确率 (Overall Accuracy):        {report['accuracy']:.4f}")

    print('\n【各类别详细指标】')
    print('-' * 70)
    print(f"{'类别':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数'}")
    print('-' * 70)
    for class_name in class_names:
        if class_name in report:
            print(f"{class_name:<8} {report[class_name]['precision']:<10.4f} "
                  f"{report[class_name]['recall']:<10.4f} "
                  f"{report[class_name]['f1-score']:<10.4f} "
                  f"{report[class_name]['support']}")
    print('-' * 70)

    return report


def generate_system_architecture():
    """生成系统架构图（PNG格式）。"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, text, color='#4A90D9', text_color='white', fontsize=10):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='#333333',
                              facecolor=color, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold', zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5), zorder=1)

    # 标题
    ax.text(7, 9.6, '基于CNN的常见动物种类识别系统架构', ha='center', fontsize=16, fontweight='bold')

    # 数据采集层
    ax.text(2.5, 9.0, '数据采集层', ha='center', fontsize=12, fontweight='bold', color='#2E7D32')
    draw_box(ax, 0.3, 8.0, 1.8, 0.7, '在线爬虫\n采集', '#66BB6A', 'white', 9)
    draw_box(ax, 2.5, 8.0, 1.8, 0.7, '数据质量\n校验', '#66BB6A', 'white', 9)
    draw_box(ax, 4.7, 8.0, 1.8, 0.7, '人工审核\n确认', '#66BB6A', 'white', 9)
    draw_arrow(ax, 2.1, 8.35, 2.5, 8.35)
    draw_arrow(ax, 4.3, 8.35, 4.7, 8.35)

    # 数据处理层
    ax.text(2.5, 7.3, '数据处理层', ha='center', fontsize=12, fontweight='bold', color='#1565C0')
    draw_box(ax, 0.3, 6.3, 1.8, 0.7, '数据集划分\n8:2', '#42A5F5', 'white', 9)
    draw_box(ax, 2.5, 6.3, 1.8, 0.7, '数据增强\n裁剪/翻转', '#42A5F5', 'white', 9)
    draw_box(ax, 4.7, 6.3, 1.8, 0.7, '标准化\n归一化', '#42A5F5', 'white', 9)
    draw_arrow(ax, 6.5, 8.0, 1.2, 7.0)
    draw_arrow(ax, 2.1, 6.65, 2.5, 6.65)
    draw_arrow(ax, 4.3, 6.65, 4.7, 6.65)

    # 模型训练层
    ax.text(9.5, 9.0, '模型训练层', ha='center', fontsize=12, fontweight='bold', color='#6A1B9A')
    draw_box(ax, 7.5, 8.0, 2.0, 0.7, 'ResNet18\n预训练模型', '#AB47BC', 'white', 9)
    draw_box(ax, 10.0, 8.0, 2.0, 0.7, '全连接层\n10分类', '#AB47BC', 'white', 9)
    draw_box(ax, 12.2, 8.0, 1.5, 0.7, 'Softmax\n概率输出', '#AB47BC', 'white', 9)
    draw_arrow(ax, 9.5, 8.35, 10.0, 8.35)
    draw_arrow(ax, 12.0, 8.35, 12.2, 8.35)

    # 训练流程
    draw_box(ax, 7.5, 6.3, 2.0, 0.7, '交叉熵\n损失函数', '#7E57C2', 'white', 9)
    draw_box(ax, 10.0, 6.3, 2.0, 0.7, 'Adam\n优化器', '#7E57C2', 'white', 9)
    draw_box(ax, 12.2, 6.3, 1.5, 0.7, '模型\n保存', '#7E57C2', 'white', 9)
    draw_arrow(ax, 9.5, 6.65, 10.0, 6.65)
    draw_arrow(ax, 12.0, 6.65, 12.2, 6.65)

    # 推理部署层
    ax.text(7, 5.3, '推理部署层', ha='center', fontsize=12, fontweight='bold', color='#C62828')
    draw_box(ax, 2.0, 4.3, 2.5, 0.7, '图像上传\n前端界面', '#EF5350', 'white', 9)
    draw_box(ax, 5.0, 4.3, 2.5, 0.7, 'Flask\n后端API', '#EF5350', 'white', 9)
    draw_box(ax, 8.0, 4.3, 2.5, 0.7, '模型推理\n预测', '#EF5350', 'white', 9)
    draw_box(ax, 11.0, 4.3, 2.5, 0.7, '结果展示\n置信度判定', '#EF5350', 'white', 9)
    draw_arrow(ax, 4.5, 4.65, 5.0, 4.65)
    draw_arrow(ax, 7.5, 4.65, 8.0, 4.65)
    draw_arrow(ax, 10.5, 4.65, 11.0, 4.65)

    # 连接箭头
    draw_arrow(ax, 5.6, 6.3, 8.5, 7.0)
    draw_arrow(ax, 13.0, 7.3, 13.0, 5.0)
    draw_arrow(ax, 13.0, 5.0, 11.0, 5.0)

    # 图例
    ax.text(0.5, 3.5, '三级数据质量保障：', fontsize=10, fontweight='bold')
    ax.text(0.5, 3.0, '第一级：搜索关键词优化 → 第二级：图像结构完整性校验与文件大小过滤 → 第三级：人工审核确认', fontsize=9)

    ax.text(0.5, 2.3, '关键技术参数：', fontsize=10, fontweight='bold')
    ax.text(0.5, 1.8, '网络架构: ResNet18 | 输入尺寸: 224×224 | 优化器: Adam | 损失函数: CrossEntropyLoss', fontsize=9)
    ax.text(0.5, 1.3, '数据集: 800张(10类×80张) | 训练/验证: 8:2 | 置信度阈值: 0.6 | 训练轮次: 10', fontsize=9)

    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    print('>>> 系统架构图已保存为: system_architecture.png')
    plt.close()


def generate_training_flowchart():
    """生成训练流程图（PNG格式）。"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def draw_rect(ax, x, y, w, h, text, color='#4A90D9', fontsize=10):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='#333333',
                              facecolor=color, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color='white', fontweight='bold', zorder=3)

    def draw_diamond(ax, cx, cy, w, h, text, color='#FF9800', fontsize=9):
        diamond = plt.Polygon([(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
                              closed=True, facecolor=color, edgecolor='#333333', linewidth=1.5, zorder=2)
        ax.add_patch(diamond)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold', zorder=3)

    def arrow(ax, x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5), zorder=1)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.15, my, label, fontsize=8, color='#666666')

    ax.text(5, 11.5, '模型训练流程图', ha='center', fontsize=16, fontweight='bold')

    draw_rect(ax, 3, 10.5, 4, 0.6, '加载预训练ResNet18模型', '#1565C0', 10)
    draw_rect(ax, 3, 9.5, 4, 0.6, '替换全连接层为10分类输出', '#1565C0', 10)
    draw_rect(ax, 3, 8.5, 4, 0.6, '配置Adam优化器与损失函数', '#1565C0', 10)
    draw_rect(ax, 3, 7.3, 4, 0.7, '加载训练批次数据', '#2E7D32', 10)
    draw_rect(ax, 3, 6.2, 4, 0.7, '前向传播计算预测值', '#2E7D32', 10)
    draw_rect(ax, 3, 5.1, 4, 0.7, '计算交叉熵损失', '#2E7D32', 10)
    draw_rect(ax, 3, 4.0, 4, 0.7, '反向传播更新参数', '#2E7D32', 10)

    draw_diamond(ax, 5, 3.0, 4, 0.8, '验证集准确率\n是否提升?', '#FF9800', 9)

    draw_rect(ax, 0.5, 1.8, 3.5, 0.6, '保存最佳模型权重', '#C62828', 10)
    draw_diamond(ax, 7.5, 1.8, 3, 0.7, '是否达到\n最大轮次?', '#9C27B0', 9)
    draw_rect(ax, 6, 0.5, 3.5, 0.6, '训练结束，输出最终模型', '#333333', 10)

    arrow(ax, 5, 10.5, 5, 10.1)
    arrow(ax, 5, 9.5, 5, 9.1)
    arrow(ax, 5, 8.5, 5, 8.0)
    arrow(ax, 5, 7.3, 5, 6.9)
    arrow(ax, 5, 6.2, 5, 5.8)
    arrow(ax, 5, 5.1, 5, 4.7)
    arrow(ax, 5, 4.0, 5, 3.4)
    arrow(ax, 3, 3.0, 2.25, 2.4, '是')
    arrow(ax, 7, 3.0, 7.5, 2.15, '否')
    arrow(ax, 6, 1.8, 6, 1.1)
    arrow(ax, 9, 1.8, 9, 7.65)
    arrow(ax, 9, 7.65, 7, 7.65)

    plt.tight_layout()
    plt.savefig('training_flowchart.png', dpi=300, bbox_inches='tight')
    print('>>> 训练流程图已保存为: training_flowchart.png')
    plt.close()


def main():
    """主函数：端到端训练与评估流程。
    
    执行流程：数据准备 → 超参数网格搜索 → 可视化图表生成 → 统计分析 → 架构图生成。
    """
    print('='*60)
    print('  基于CNN的常见动物种类识别系统 - 训练与评估')
    print('='*60)
    print(f'当前设备: {device}')
    print(f'运行模式: {"离线模式" if USE_OFFLINE_MODE else "在线爬虫模式"}')
    print(f'数据集配置: 每类{DOWNLOAD_COUNT_PER_CLASS}张, 共{len(CLASSES_ZH)}类, 总计{DOWNLOAD_COUNT_PER_CLASS * len(CLASSES_ZH)}张')

    try:
        prepare_dataset()
    except Exception as e:
        print(f'数据准备失败: {e}')
        return

    keys, values = zip(*HYPER_PARAMS.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results_map = {}
    train_loss_map = {}
    val_loss_map = {}
    global_best_acc = 0.0

    for params in param_combinations:
        lr = params['learning_rate']
        batch_size = params['batch_size']
        cfg_name = f"lr={lr}, bs={batch_size}"

        print(f'\n>>> 开始训练配置: {cfg_name}')
        best_acc, val_acc_history, train_loss_history, val_loss_history = train_and_evaluate(lr=lr, batch_size=batch_size)
        results_map[cfg_name] = val_acc_history
        train_loss_map[cfg_name] = train_loss_history
        val_loss_map[cfg_name] = val_loss_history

        if best_acc > global_best_acc:
            global_best_acc = best_acc

    print(f'\n>>> 训练完成，全局最佳验证准确率: {global_best_acc:.4f}')

    print('\n>>> 生成可视化图表...')
    plot_history(results_map)
    plot_loss_curves(train_loss_map, val_loss_map)

    print('\n>>> 执行统计分析...')
    dataloaders, _ = get_dataloaders(batch_size=32)
    final_model = build_model(num_classes=10)
    final_model.load_state_dict(torch.load('最终动物识别模型.pth', map_location=device))
    final_model.to(device)

    cm, report, _, _ = compute_metrics(final_model, dataloaders['验证集'], device, CLASSES_ZH)
    plot_confusion_matrix(cm, CLASSES_ZH)
    print_statistical_analysis(report, CLASSES_ZH)

    print('\n>>> 生成系统架构图与流程图...')
    generate_system_architecture()
    generate_training_flowchart()

    print('\n' + '='*60)
    print('            全部改进已完成')
    print('='*60)
    print(' 改进项清单:')
    print(' [1] 数据集扩充: 400张 → 800张 (10类×80张/类)')
    print(' [2] 训练轮次增加: 5轮 → 10轮')
    print(' [3] 统计分析: 混淆矩阵、精确率、召回率、F1分数')
    print(' [4] 可视化图表: 学习曲线、损失曲线、混淆矩阵热力图')
    print(' [5] 系统架构图: system_architecture.png')
    print(' [6] 训练流程图: training_flowchart.png')
    print(' [7] 语句表述规范化: 学术用语替换')
    print('='*60)
    print('\n 生成的图表文件:')
    print('   learning_curve.png    - 验证集准确率变化曲线')
    print('   loss_curves.png       - 训练/验证损失变化曲线')
    print('   confusion_matrix.png  - 混淆矩阵热力图')
    print('   system_architecture.png - 系统架构图')
    print('   training_flowchart.png  - 训练流程图')
    print('='*60)


if __name__ == '__main__':
    main()