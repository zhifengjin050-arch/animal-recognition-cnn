import json
import os

# ============================================================================
#  基础路径配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
RAW_DATA_DIR = os.path.join(BASE_DIR, '动物识别数据集', '原始数据')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '动物识别数据集', 'data')

# 模型保存路径
MODEL_SAVE_DIR = os.path.join(BASE_DIR, '优化训练报告')
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, '最终动物识别模型.pth')
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_temp.pth')

# 图表输出路径
CHART_OUTPUT_DIR = MODEL_SAVE_DIR
TRAIN_CHART_DIR = os.path.join(BASE_DIR, '训练报告')

# 指标文件路径
METRICS_PATH = os.path.join(MODEL_SAVE_DIR, 'metrics.json')

# ============================================================================
#  8 类动物中英文名称映射
# ============================================================================
CLASSES_ZH = ['牛', '狗', '狮子', '猫', '羊', '虎', '鸡', '鸭']
CLASSES_EN = ['cattle', 'dog', 'lion', 'cat', 'sheep', 'tiger', 'chicken', 'duck']

ZH_TO_EN = dict(zip(CLASSES_ZH, CLASSES_EN))
EN_TO_ZH = dict(zip(CLASSES_EN, CLASSES_ZH))

NUM_CLASSES = len(CLASSES_ZH)

# ============================================================================
#  超参数默认值（来自最佳实验结果）
# ============================================================================
DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 48
DEFAULT_EPOCHS = 15
DEFAULT_OPTIMIZER = 'Adam'
DEFAULT_LOSS_FN = 'CrossEntropyLoss'
DEFAULT_SCHEDULER = 'CosineAnnealingLR'

# ============================================================================
#  从 metrics.json 动态加载模型指标
# ============================================================================
def _load_metrics():
    """从 metrics.json 加载最新模型训练指标，若文件不存在则返回 None"""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

_metrics = _load_metrics()

if _metrics is not None:
    TOTAL_TRAIN = _metrics.get('total_train', 9040)
    TOTAL_VAL = _metrics.get('total_val', 480)
    TOTAL_TEST = _metrics.get('total_test', 480)
    TOTAL_SAMPLES = TOTAL_TRAIN + TOTAL_VAL + TOTAL_TEST

    TRAIN_COUNTS = _metrics.get('train_counts', {})
    VAL_COUNTS = _metrics.get('val_counts', {})
    TEST_COUNTS = _metrics.get('test_counts', {})

    BEST_ACCURACY = round(_metrics.get('best_accuracy', 0.9229) * 100, 2)
    PRECISION_MACRO = round(_metrics.get('precision_macro', 0.9117) * 100, 2)
    RECALL_MACRO = round(_metrics.get('recall_macro', 0.9104) * 100, 2)
    F1_MACRO = round(_metrics.get('f1_macro', 0.9100) * 100, 2)

    PER_CLASS_ACCURACY = _metrics.get('per_class_accuracy', [])
    PER_CLASS_PRECISION = _metrics.get('per_class_precision', [])
    PER_CLASS_RECALL = _metrics.get('per_class_recall', [])
    PER_CLASS_F1 = _metrics.get('per_class_f1', [])

    CONFUSION_MATRIX = _metrics.get('confusion_matrix', [])

    HISTORY_ACC = _metrics.get('history_acc', [])
    HISTORY_LOSS = _metrics.get('history_loss', [])
    HISTORY_VAL_ACC = _metrics.get('history_val_acc', [])
    HISTORY_VAL_LOSS = _metrics.get('history_val_loss', [])

    ACTUAL_LR = _metrics.get('learning_rate', DEFAULT_LR)
    ACTUAL_BATCH_SIZE = _metrics.get('batch_size', DEFAULT_BATCH_SIZE)
    ACTUAL_EPOCHS = _metrics.get('epochs', DEFAULT_EPOCHS)
else:
    TOTAL_TRAIN = 9040
    TOTAL_VAL = 480
    TOTAL_TEST = 480
    TOTAL_SAMPLES = 10000
    TRAIN_COUNTS = {}
    VAL_COUNTS = {}
    TEST_COUNTS = {}
    BEST_ACCURACY = 92.29
    PRECISION_MACRO = 91.17
    RECALL_MACRO = 91.04
    F1_MACRO = 91.00
    PER_CLASS_ACCURACY = []
    PER_CLASS_PRECISION = []
    PER_CLASS_RECALL = []
    PER_CLASS_F1 = []
    CONFUSION_MATRIX = []
    HISTORY_ACC = []
    HISTORY_LOSS = []
    HISTORY_VAL_ACC = []
    HISTORY_VAL_LOSS = []
    ACTUAL_LR = DEFAULT_LR
    ACTUAL_BATCH_SIZE = DEFAULT_BATCH_SIZE
    ACTUAL_EPOCHS = DEFAULT_EPOCHS

# ============================================================================
#  统一学术配色方案（8 色调色板）
# ============================================================================
COLOR_BLUE = '#2E86C1'
COLOR_RED = '#E74C3C'
COLOR_GREEN = '#27AE60'
COLOR_ORANGE = '#F39C12'
COLOR_PURPLE = '#8E44AD'
COLOR_TEAL = '#17A589'
COLOR_PINK = '#E91E63'
COLOR_DARK = '#2C3E50'
COLOR_GRAY = '#7F8C8D'

PALETTE_8 = [
    '#2E86C1', '#E74C3C', '#27AE60', '#F39C12',
    '#8E44AD', '#17A589', '#E91E63', '#D35400',
]

# ============================================================================
#  统一字体配置
# ============================================================================
FONT_SANS_SERIF = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
FONT_SERIF = ['Times New Roman', 'DejaVu Serif']

FONT_SIZE_TITLE = 16
FONT_SIZE_SUBTITLE = 14
FONT_SIZE_AXIS = 13
FONT_SIZE_TICK = 11
FONT_SIZE_LEGEND = 11
FONT_SIZE_ANNOTATION = 10
FONT_SIZE_NOTE = 10

FONT_WEIGHT_BOLD = 'bold'
FONT_WEIGHT_NORMAL = 'normal'

# ============================================================================
#  图表通用配置
# ============================================================================
CHART_DPI = 300
CHART_FACE_COLOR = 'white'
CHART_SPINE_COLOR = '#cccccc'
CHART_GRID_ALPHA = 0.25
CHART_GRID_STYLE = '--'
BAR_EDGE_COLOR = 'white'
BAR_EDGE_WIDTH = 0.6
SCATTER_EDGE_COLOR = 'white'
SCATTER_EDGE_WIDTH = 0.5

# ============================================================================
#  数据管理工具配置
# ============================================================================
DATASET_SPLITS = ['train', 'val', 'test']
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
VALID_COLOR_MODES = ('RGB', 'RGBA')
SUPPORTED_FORMATS = ['JPEG', 'PNG', 'BMP', 'WEBP', 'GIF']

SNAPSHOT_DIR = os.path.join(BASE_DIR, '数据集快照')
REPORT_DIR = os.path.join(BASE_DIR, '数据集报告')

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def print_config():
    """打印当前配置信息，用于调试确认"""
    print('=' * 60)
    print('  统一配置信息')
    print('=' * 60)
    print(f'  基础目录:         {BASE_DIR}')
    print(f'  类别数:           {NUM_CLASSES}')
    print(f'  中文类别:         {CLASSES_ZH}')
    print(f'  英文类别:         {CLASSES_EN}')
    print(f'  训练集样本数:     {TOTAL_TRAIN}')
    print(f'  验证集样本数:     {TOTAL_VAL}')
    print(f'  测试集样本数:     {TOTAL_TEST}')
    print(f'  总样本数:         {TOTAL_SAMPLES}')
    print(f'  超参数:           lr={ACTUAL_LR}, batch_size={ACTUAL_BATCH_SIZE}, epochs={ACTUAL_EPOCHS}')
    print(f'  最佳准确率:       {BEST_ACCURACY}%')
    print(f'  宏平均精确率:     {PRECISION_MACRO}%')
    print(f'  宏平均召回率:     {RECALL_MACRO}%')
    print(f'  宏平均 F1:        {F1_MACRO}%')
    print('=' * 60)


if __name__ == '__main__':
    print_config()