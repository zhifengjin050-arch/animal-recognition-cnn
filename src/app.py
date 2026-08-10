import io
import os
import sys
import time

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from data_manager import (
        get_dataset_summary, run_full_quality_check, check_class_balance,
        list_snapshots, get_training_history, get_data_lineage,
        generate_dashboard_data, get_augmentation_summary, scan_dataset
    )
    HAS_DATA_MANAGER = True
except Exception:
    HAS_DATA_MANAGER = False

try:
    from viz_utils import setup_academic_style
    setup_academic_style()
    HAS_VIZ = True
except Exception:
    HAS_VIZ = False

CLASSES_ZH = ['猫', '牛', '鸡', '狗', '鸭', '狮子', '羊', '虎']
CLASSES_EN = ['cat', 'cattle', 'chicken', 'dog', 'duck', 'lion', 'sheep', 'tiger']

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', '最终动物识别模型_10k.pth')
UNKNOWN_THRESHOLD = 0.60
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DATASET_INFO = {
    'classes': [
        {'zh': '猫', 'en': 'cat', 'train': 1130, 'val': 60, 'test': 60, 'source': 'Bing图像搜索'},
        {'zh': '牛', 'en': 'cattle', 'train': 1130, 'val': 60, 'test': 60, 'source': 'Bing图像搜索'},
        {'zh': '鸡', 'en': 'chicken', 'train': 1130, 'val': 60, 'test': 60, 'source': 'Bing图像搜索'},
        {'zh': '狗', 'en': 'dog', 'train': 1130, 'val': 60, 'test': 60, 'source': 'Bing图像搜索'},
        {'zh': '鸭', 'en': 'duck', 'train': 1130, 'val': 60, 'test': 60, 'source': 'Bing图像搜索'},
        {'zh': '狮子', 'en': 'lion', 'train': 1130, 'val': 60, 'test': 60, 'source': 'Bing图像搜索'},
        {'zh': '羊', 'en': 'sheep', 'train': 1130, 'val': 60, 'test': 60, 'source': 'Bing图像搜索'},
        {'zh': '虎', 'en': 'tiger', 'train': 1130, 'val': 60, 'test': 60, 'source': 'Bing图像搜索'},
    ],
    'total_train': 9040,
    'total_val': 480,
    'total_test': 480,
    'total': 10000,
    'split_ratio': '90.4:4.8:4.8',
    'model': 'ResNet18',
    'epochs': 15,
    'batch_size': 48,
    'optimizer': 'Adam',
    'lr': 0.001,
    'lr_schedule': 'CosineAnnealingLR',
    'loss_fn': 'CrossEntropyLoss',
    'input_size': '224×224',
    'threshold': UNKNOWN_THRESHOLD
}

MODEL_METRICS = {
    'accuracy': 0.9104,
    'precision': 0.9117,
    'recall': 0.9104,
    'f1': 0.9100
}

HISTORY_LOG = []


def build_model():
    """构建 ResNet18 模型并加载预训练权重"""
    model_file = MODEL_PATH
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'未找到模型文件: {model_file}')

    state_dict = torch.load(model_file, map_location=device)
    
    fc_keys = [k for k in state_dict.keys() if k.startswith('fc.')]
    if not fc_keys:
        raise ValueError(f'模型文件中未找到fc相关权重')
    
    last_linear_weight = None
    for k in sorted(fc_keys, reverse=True):
        if k.endswith('.weight') and 'running' not in k and 'num_batches' not in k:
            last_linear_weight = k
            break
    
    if last_linear_weight is None:
        raise ValueError('无法找到最后一个Linear层权重')
    
    num_classes = state_dict[last_linear_weight].shape[0]
    
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


try:
    model = build_model()
    print(f'模型加载成功，设备: {device}')
except Exception as e:
    print(f'模型加载失败: {e}，请将模型权重文件放入 models/ 目录')
    model = None

app = Flask(__name__, static_folder=BASE_DIR)


@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """动物识别接口：上传图片进行识别"""
    if model is None:
        return jsonify({'ok': False, 'error': '模型未加载'}), 503
    
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify({'ok': False, 'error': '请选择图片后再提交。'}), 400

    try:
        start_time = time.time()
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = infer_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, k=min(5, len(CLASSES_ZH)))
        inference_time = (time.time() - start_time) * 1000

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            predictions.append({
                'zh': CLASSES_ZH[idx],
                'en': CLASSES_EN[idx],
                'probability': float(prob)
            })

        top1_prob = predictions[0]['probability']
        is_unknown = top1_prob < UNKNOWN_THRESHOLD

        all_probs = {CLASSES_ZH[i]: float(probs[i]) for i in range(len(CLASSES_ZH))}

        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'result': '未知' if is_unknown else predictions[0]['zh'],
            'confidence': top1_prob,
            'inference_time': round(inference_time, 2),
            'is_correct': None
        }
        HISTORY_LOG.append(log_entry)

        return jsonify({
            'ok': True,
            'source': 'backend_resnet18',
            'is_unknown': is_unknown,
            'unknown_threshold': UNKNOWN_THRESHOLD,
            'message': '非论文8类 / 无法可靠识别' if is_unknown else '识别成功',
            'predictions': predictions,
            'all_probs': all_probs,
            'inference_time': round(inference_time, 2)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': f'识别失败: {e}'}), 500


@app.route('/api/dataset_info', methods=['GET'])
def dataset_info():
    return jsonify(DATASET_INFO)


@app.route('/api/model_metrics', methods=['GET'])
def model_metrics():
    return jsonify(MODEL_METRICS)


@app.route('/api/history', methods=['GET'])
def history():
    return jsonify({'history': HISTORY_LOG[-50:]})


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    HISTORY_LOG.clear()
    return jsonify({'ok': True})


@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    if HAS_DATA_MANAGER:
        try:
            data = generate_dashboard_data()
            return jsonify({'ok': True, 'data': data})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': False, 'error': '数据管理模块未加载'})


@app.route('/api/data_quality', methods=['GET'])
def data_quality():
    if HAS_DATA_MANAGER:
        try:
            run_full = request.args.get('full', '0') == '1'
            if run_full:
                report = run_full_quality_check()
            else:
                stats = scan_dataset()
                balance = check_class_balance(stats)
                summary = get_dataset_summary()
                report = {'summary': summary, 'balance': balance}
            return jsonify({'ok': True, 'data': report})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': False, 'error': '数据管理模块未加载'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)