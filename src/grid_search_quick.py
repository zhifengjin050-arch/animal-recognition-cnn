"""快速网格搜索 - 获取4个超参数组合的完整验证指标"""
import os, json, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report

BASE_DIR = r'd:\代码项目\论文识别系统\识别系统'
DS_DIR = os.path.join(BASE_DIR, '动物识别数据集_10k')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CLASSES_ZH = ['猫', '牛', '鸡', '狗', '鸭', '狮子', '羊', '虎']

LR_LIST = [0.001, 0.0001]
BS_LIST = [48, 32]
EPOCHS = 2

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DS_DIR, 'train'), transform=train_transform)
val_ds = datasets.ImageFolder(os.path.join(DS_DIR, 'val'), transform=val_transform)
print(f'Train: {len(train_ds)} images, classes={train_ds.classes}')
print(f'Val: {len(val_ds)} images, classes={val_ds.classes}')
print(f'Device: {DEVICE}')

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 8)
    )
    return model.to(DEVICE)

def run_trial(lr, bs, label):
    print(f'\n{"="*50}')
    print(f'  {label}: LR={lr}, BS={bs}, Epochs={EPOCHS}')
    print(f'{"="*50}')

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_metrics = None
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f'  Epoch {epoch+1}: loss={train_loss:.4f}/{val_loss:.4f} acc={train_acc:.4f}/{val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if val_acc > 0.5:
                report = classification_report(all_labels, all_preds, target_names=CLASSES_ZH,
                                                output_dict=True, zero_division=0)
                best_metrics = {
                    'val_accuracy': round(float(val_acc), 6),
                    'val_loss': round(float(val_loss), 6),
                    'precision_macro': round(report['macro avg']['precision'], 6),
                    'recall_macro': round(report['macro avg']['recall'], 6),
                    'f1_macro': round(report['macro avg']['f1-score'], 6),
                }

    elapsed = time.time() - t_start
    if best_val_acc <= 0.5:
        report = classification_report(all_labels, all_preds, target_names=CLASSES_ZH,
                                        output_dict=True, zero_division=0)
        best_metrics = {
            'val_accuracy': round(float(best_val_acc), 6),
            'val_loss': round(float(val_loss), 6),
            'precision_macro': round(report['macro avg']['precision'], 6),
            'recall_macro': round(report['macro avg']['recall'], 6),
            'f1_macro': round(report['macro avg']['f1-score'], 6),
        }
    print(f'  Done in {elapsed:.1f}s | acc={best_val_acc:.4f} prec={best_metrics["precision_macro"]:.4f} rec={best_metrics["recall_macro"]:.4f} f1={best_metrics["f1_macro"]:.4f}')
    return best_metrics

t0 = time.time()
results = {}

for lr, bs in itertools.product(LR_LIST, BS_LIST):
    label = f'组合{"ABCD"[len(results)]}'
    results[f'{label}'] = {'lr': lr, 'bs': bs, **run_trial(lr, bs, label)}

elapsed = time.time() - t0
print(f'\n{"="*50}')
print(f'  GRID SEARCH COMPLETE ({elapsed/60:.1f} min)')
print(f'{"="*50}')

for combo_name, m in results.items():
    print(f'\n  {combo_name} (lr={m["lr"]}, bs={m["bs"]}):')
    print(f'    accuracy:  {m["val_accuracy"]:.4f}')
    print(f'    precision: {m["precision_macro"]:.4f}')
    print(f'    recall:    {m["recall_macro"]:.4f}')
    print(f'    f1:        {m["f1_macro"]:.4f}')

out_path = os.path.join(BASE_DIR, '优化训练报告', 'grid_search_results.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\nSaved to {out_path}')
