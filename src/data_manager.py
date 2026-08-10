import os
import json
import hashlib
import time
import csv
import shutil
from collections import defaultdict, OrderedDict
from datetime import datetime
from PIL import Image, UnidentifiedImageError

from config import (
    BASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, SNAPSHOT_DIR, REPORT_DIR,
    CLASSES_ZH, CLASSES_EN, DATASET_SPLITS,
    VALID_IMAGE_EXTENSIONS, MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT,
    VALID_COLOR_MODES, SUPPORTED_FORMATS
)

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

QUALITY_LOG_PATH = os.path.join(REPORT_DIR, 'quality_log.json')
TRAINING_LOG_PATH = os.path.join(REPORT_DIR, 'training_history.json')
AUGMENTATION_LOG_PATH = os.path.join(REPORT_DIR, 'augmentation_log.json')
DATA_LINEAGE_PATH = os.path.join(REPORT_DIR, 'data_lineage.json')


# ============================================================================
#  一、数据集统计与扫描
# ============================================================================

def scan_dataset(base_path=None):
    """扫描数据集统计各类别样本数量。
    
    遍历 train/val/test 三个子目录，统计每个类别在各划分中的图片数量。
    
    Args:
        base_path: 数据集根目录路径，默认为 PROCESSED_DATA_DIR
    
    Returns:
        dict: {类别英文名: {'train': 数量, 'val': 数量, 'test': 数量, 'total': 总数}}
    """
    if base_path is None:
        base_path = PROCESSED_DATA_DIR
    stats = defaultdict(lambda: defaultdict(int))
    for split in DATASET_SPLITS:
        split_dir = os.path.join(base_path, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            count = 0
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    count += 1
            stats[class_name][split] = count
            stats[class_name]['total'] = stats[class_name].get('total', 0) + count
    return dict(stats)


def get_dataset_summary(base_path=None):
    """获取数据集汇总信息，包含各类别中英文名和各划分样本数。
    
    Args:
        base_path: 数据集根目录路径
    
    Returns:
        dict: 包含 classes 列表、各划分总数、总计和划分比例
    """
    if base_path is None:
        base_path = PROCESSED_DATA_DIR
    stats = scan_dataset(base_path)
    total_train = 0
    total_val = 0
    total_test = 0
    class_list = []
    for cls_name_en, cls_data in sorted(stats.items()):
        cls_name_zh = CLASSES_ZH[CLASSES_EN.index(cls_name_en)] if cls_name_en in CLASSES_EN else cls_name_en
        train_cnt = cls_data.get('train', 0)
        val_cnt = cls_data.get('val', 0)
        test_cnt = cls_data.get('test', 0)
        total_train += train_cnt
        total_val += val_cnt
        total_test += test_cnt
        class_list.append({
            'zh': cls_name_zh,
            'en': cls_name_en,
            'train': train_cnt,
            'val': val_cnt,
            'test': test_cnt,
            'total': train_cnt + val_cnt + test_cnt
        })
    total_all = total_train + total_val + total_test
    return {
        'classes': class_list,
        'total_train': total_train,
        'total_val': total_val,
        'total_test': total_test,
        'total': total_all,
        'split_ratio': f'{total_train/total_all:.1%}:{total_val/total_all:.1%}:{total_test/total_all:.1%}'
    }


# ============================================================================
#  二、数据质量验证
# ============================================================================

def verify_image(filepath):
    """验证单张图片的完整性和质量。
    
    检查项：文件存在性、扩展名、图片可解码性、分辨率、色彩模式。
    
    Args:
        filepath: 图片文件路径
    
    Returns:
        dict: 包含 valid, error, width, height, format, mode, file_size 字段
    """
    result = {'valid': True, 'error': None, 'width': 0, 'height': 0,
              'format': '', 'mode': '', 'file_size': 0}
    if not os.path.exists(filepath):
        result['valid'] = False
        result['error'] = '文件不存在'
        return result
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        result['valid'] = False
        result['error'] = f'不支持的图片格式: {ext}'
        return result
    result['file_size'] = os.path.getsize(filepath)
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            result['width'], result['height'] = img.size
            result['format'] = img.format if img.format else ''
            result['mode'] = img.mode
        if result['width'] < MIN_IMAGE_WIDTH or result['height'] < MIN_IMAGE_HEIGHT:
            result['valid'] = False
            result['error'] = f'分辨率过低: {result["width"]}x{result["height"]}, 最低要求 {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}'
        if result['mode'] not in VALID_COLOR_MODES:
            result['valid'] = False
            result['error'] = f'色彩模式异常: {result["mode"]}, 仅支持 {", ".join(VALID_COLOR_MODES)}'
    except UnidentifiedImageError:
        result['valid'] = False
        result['error'] = '无法识别图片格式，文件可能已损坏'
    except Exception as e:
        result['valid'] = False
        result['error'] = f'图片读取异常: {str(e)}'
    return result


def run_full_quality_check(base_path=None):
    """执行全量数据质量检查，生成质量报告。
    
    遍历数据集中所有图片文件，逐一验证完整性，记录损坏文件和警告项。
    
    Args:
        base_path: 数据集根目录路径
    
    Returns:
        dict: 质量检查报告，包含总文件数、有效文件数、损坏文件列表、警告列表
    """
    if base_path is None:
        base_path = PROCESSED_DATA_DIR
    print('\n' + '=' * 60)
    print('  全量数据质量检查')
    print('=' * 60)
    total = 0
    valid_count = 0
    corrupt_files = []
    warnings = []
    for split in DATASET_SPLITS:
        split_dir = os.path.join(base_path, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    continue
                fpath = os.path.join(class_dir, fname)
                total += 1
                result = verify_image(fpath)
                if result['valid']:
                    valid_count += 1
                    if result['width'] < 224 or result['height'] < 224:
                        warnings.append((fpath, f'分辨率偏低 {result["width"]}x{result["height"]}'))
                else:
                    corrupt_files.append((fpath, result['error']))
                if total % 500 == 0:
                    print(f'  已检查 {total} 个文件...')

    quality_report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': total,
        'valid_files': valid_count,
        'corrupt_files': corrupt_files,
        'corrupt_count': len(corrupt_files),
        'warnings': warnings,
        'warning_count': len(warnings),
        'valid_rate': round(valid_count / total * 100, 2) if total > 0 else 0
    }

    report_path = os.path.join(REPORT_DIR, f'quality_check_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)
    with open(QUALITY_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    print(f'\n  检查完成: {total} 个文件中 {valid_count} 个有效 ({quality_report["valid_rate"]}%)')
    print(f'  损坏文件: {len(corrupt_files)} 个')
    print(f'  警告: {len(warnings)} 个')
    print(f'  报告已保存: {report_path}')
    print('=' * 60)
    return quality_report


def calculate_dataset_quality_score(base_path=None):
    """计算数据集综合质量评分（0-100分）。
    
    评分维度：文件有效性(50%)、类别平衡性(30%)、警告数量(20%)。
    
    Args:
        base_path: 数据集根目录路径
    
    Returns:
        dict: 包含 overall_score, grade, sub_scores, details 字段
    """
    if base_path is None:
        base_path = PROCESSED_DATA_DIR
    report = run_full_quality_check(base_path)
    stats = scan_dataset(base_path)
    total = report['total_files']
    valid = report['valid_files']
    corrupt = report['corrupt_count']
    warnings = report['warning_count']

    score_validity = (valid / total * 100) if total > 0 else 0
    class_counts = [cls_data.get('total', 0) for cls_data in stats.values()]
    if class_counts and max(class_counts) > 0:
        balance_ratio = min(class_counts) / max(class_counts)
    else:
        balance_ratio = 0
    score_balance = balance_ratio * 100
    score_warnings = max(0, 100 - warnings * 2)
    overall_score = round(score_validity * 0.5 + score_balance * 0.3 + score_warnings * 0.2, 1)

    quality_score = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall_score': overall_score,
        'grade': 'A' if overall_score >= 90 else 'B' if overall_score >= 75 else 'C' if overall_score >= 60 else 'D',
        'sub_scores': {
            'file_validity': round(score_validity, 1),
            'class_balance': round(score_balance, 1),
            'no_warnings': round(score_warnings, 1)
        },
        'details': {
            'total_files': total,
            'valid_files': valid,
            'corrupt_files': corrupt,
            'warnings': warnings,
            'num_classes': len(class_counts),
            'balance_ratio': round(balance_ratio, 4)
        }
    }
    score_path = os.path.join(REPORT_DIR, f'dataset_quality_score_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(score_path, 'w', encoding='utf-8') as f:
        json.dump(quality_score, f, ensure_ascii=False, indent=2)
    print(f'\n  数据集质量评分: {overall_score}/100 (等级: {quality_score["grade"]})')
    return quality_score


# ============================================================================
#  三、MD5去重检查
# ============================================================================

def deduplicate_by_md5(path=None):
    """基于MD5哈希值检测并报告数据集中的重复文件。
    
    遍历数据集中所有图片文件，计算MD5哈希值，识别具有相同哈希值的重复文件。
    
    Args:
        path: 数据集根目录路径
    
    Returns:
        dict: {哈希值: [文件路径列表]}，仅包含重复项
    """
    if path is None:
        path = PROCESSED_DATA_DIR
    print('\n' + '=' * 60)
    print('  MD5去重检查')
    print('=' * 60)
    print(f'  扫描路径: {path}')
    file_hashes = defaultdict(list)
    total_files = 0
    for root, dirs, files in os.walk(path):
        for fname in files:
            if not fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                continue
            fpath = os.path.join(root, fname)
            total_files += 1
            md5_hash = _calculate_file_hash(fpath)
            if md5_hash:
                file_hashes[md5_hash].append(fpath)
            if total_files % 200 == 0:
                print(f'  已扫描 {total_files} 个文件...')
    duplicates = {}
    duplicate_file_count = 0
    for hash_val, file_list in file_hashes.items():
        if len(file_list) > 1:
            duplicates[hash_val] = file_list
            duplicate_file_count += len(file_list) - 1
    print(f'  扫描文件总数: {total_files}')
    print(f'  重复哈希值: {len(duplicates)} 个')
    print(f'  重复文件数: {duplicate_file_count} 个')
    if duplicates:
        print(f'\n  重复文件明细:')
        for i, (hash_val, file_list) in enumerate(duplicates.items()):
            if i >= 5:
                print(f'  ... 及其他 {len(duplicates) - 5} 组重复')
                break
            print(f'    MD5: {hash_val[:12]}... ({len(file_list)} 个文件)')
            for fp in file_list[:3]:
                print(f'      - {os.path.relpath(fp, path)}')
    print('=' * 60)
    return duplicates


# ============================================================================
#  四、类别平衡性分析
# ============================================================================

def check_class_balance(stats=None):
    """分析数据集类别平衡性，识别样本不足的类别并生成改进建议。
    
    判断标准：最大类别样本数 / 最小类别样本数 <= 2.0 视为平衡。
    
    Args:
        stats: scan_dataset() 的返回结果，为 None 时自动扫描
    
    Returns:
        dict: 包含 is_balanced, max_ratio, per_split_balance, recommendations 等字段
    """
    if stats is None:
        stats = scan_dataset()
    recommendations = []
    totals = {}
    for cls_name, cls_data in stats.items():
        totals[cls_name] = cls_data.get('total', 0)
    if not totals:
        return {'is_balanced': True, 'max_ratio': 1.0, 'recommendations': ['数据集为空，无法分析']}
    max_class = max(totals, key=totals.get)
    min_class = min(totals, key=totals.get)
    max_count = totals[max_class]
    min_count = totals[min_class]
    max_ratio = max_count / min_count if min_count > 0 else float('inf')
    imbalances = []
    avg_count = sum(totals.values()) / len(totals)
    for cls_name, count in totals.items():
        if count < avg_count * 0.5:
            imbalances.append((cls_name, count, '严重不足'))
        elif count < avg_count * 0.8:
            imbalances.append((cls_name, count, '偏少'))
    per_split_balance = {}
    for split in DATASET_SPLITS:
        split_counts = {}
        for cls_name, cls_data in stats.items():
            split_counts[cls_name] = cls_data.get(split, 0)
        if not any(split_counts.values()):
            continue
        split_max = max(split_counts, key=split_counts.get)
        split_min = min(split_counts, key=split_counts.get)
        split_max_val = split_counts[split_max]
        split_min_val = split_counts[split_min]
        split_ratio = split_max_val / split_min_val if split_min_val > 0 else float('inf')
        per_split_balance[split] = {
            'ratio': round(split_ratio, 2),
            'max_class': split_max, 'max_count': split_max_val,
            'min_class': split_min, 'min_count': split_min_val
        }
    is_balanced = max_ratio <= 2.0 and not imbalances
    if max_ratio > 3.0:
        recommendations.append(
            f'严重类别不平衡: "{max_class}"({max_count}张) 是 "{min_class}"({min_count}张) 的 {max_ratio:.1f} 倍。'
        )
    elif max_ratio > 2.0:
        recommendations.append(
            f'轻度类别不平衡: "{max_class}"({max_count}张) 与 "{min_class}"({min_count}张) 比例为 {max_ratio:.1f}:1。'
        )
    else:
        recommendations.append(f'类别分布较为均匀，最大比例为 {max_ratio:.1f}:1')
    for cls_name, count, level in imbalances:
        recommendations.append(f'类别 "{cls_name}" 样本数量{level}({count}张)，建议补充样本')
    if is_balanced:
        recommendations.append('数据集类别基本平衡，可直接用于训练')
    return {
        'is_balanced': is_balanced,
        'max_ratio': round(max_ratio, 2),
        'max_class': max_class, 'max_count': max_count,
        'min_class': min_class, 'min_count': min_count,
        'average_count': round(avg_count, 1),
        'per_split_balance': per_split_balance,
        'recommendations': recommendations
    }


# ============================================================================
#  五、数据集报告生成与导出
# ============================================================================

def generate_dataset_report(base_path=None, check_integrity=True):
    """生成完整的数据集统计报告，包含类别统计、去重检查和完整性验证。
    
    Args:
        base_path: 数据集根目录路径
        check_integrity: 是否执行文件完整性校验
    
    Returns:
        dict: 完整的数据集报告
    """
    if base_path is None:
        base_path = PROCESSED_DATA_DIR
    print('\n' + '=' * 60)
    print('  数据集统计报告生成')
    print('=' * 60)
    print(f'  数据集路径: {base_path}')
    category_stats = scan_dataset(base_path)
    total_stats = defaultdict(int)
    for cls_name, cls_data in category_stats.items():
        for split in DATASET_SPLITS:
            total_stats[split] += cls_data.get(split, 0)
    total_stats['total'] = sum(total_stats[s] for s in DATASET_SPLITS)
    print(f'\n  总体统计:')
    for split in DATASET_SPLITS:
        print(f'    {split}: {total_stats.get(split, 0)} 张')
    print(f'    总计: {total_stats["total"]} 张')
    duplicate_info = deduplicate_by_md5(base_path)
    dup_count = sum(len(v) - 1 for v in duplicate_info.values())
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'base_path': os.path.abspath(base_path),
        'category_stats': category_stats,
        'total_stats': dict(total_stats),
        'duplicate_info': {hash_val: file_list for hash_val, file_list in duplicate_info.items()},
        'duplicate_count': dup_count,
        'corrupt_files': [], 'total_corrupt': 0
    }
    if check_integrity:
        print(f'\n  执行文件完整性校验...')
        corrupt_files = _batch_verify_integrity(base_path)
        report['corrupt_files'] = corrupt_files
        report['total_corrupt'] = len(corrupt_files)
        if corrupt_files:
            print(f'  [警告] 发现 {len(corrupt_files)} 个损坏文件')
        else:
            print(f'  所有文件完整性校验通过')
    balance_result = check_class_balance(category_stats)
    report['balance_analysis'] = balance_result
    report_path = os.path.join(REPORT_DIR, f'dataset_report_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n  报告已保存: {report_path}')
    print('=' * 60)
    return report


def export_dataset_manifest(base_path=None, output_path=None):
    """导出数据集清单（JSON + CSV 双格式），包含每个文件的元数据。
    
    Args:
        base_path: 数据集根目录路径
        output_path: 输出 JSON 文件路径
    
    Returns:
        str: 生成的 JSON 清单文件路径
    """
    if base_path is None:
        base_path = PROCESSED_DATA_DIR
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, f'dataset_manifest_{time.strftime("%Y%m%d_%H%M%S")}.json')
    print('\n' + '=' * 60)
    print('  导出数据集清单')
    print('=' * 60)
    manifest = []
    file_count = 0
    for split in DATASET_SPLITS:
        split_dir = os.path.join(base_path, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    continue
                fpath = os.path.join(class_dir, fname)
                file_count += 1
                file_size = os.path.getsize(fpath)
                md5_hash = _calculate_file_hash(fpath)
                try:
                    with Image.open(fpath) as img:
                        width, height = img.size
                except Exception:
                    width, height = 0, 0
                manifest.append({
                    'file_path': os.path.relpath(fpath, BASE_DIR),
                    'file_name': fname, 'class_label': class_name,
                    'split': split, 'file_size_bytes': file_size,
                    'file_size_mb': round(file_size / (1024 * 1024), 4),
                    'md5': md5_hash, 'width': width, 'height': height
                })
                if file_count % 500 == 0:
                    print(f'  已处理 {file_count} 个文件...')
    manifest_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'base_path': os.path.abspath(base_path),
        'total_files': file_count, 'items': manifest
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)
    print(f'  共导出 {file_count} 条记录')
    print(f'  清单已保存: {output_path}')
    csv_path = output_path.replace('.json', '.csv')
    _export_manifest_csv(manifest, csv_path)
    print('=' * 60)
    return output_path


def _export_manifest_csv(manifest, csv_path):
    """内部函数：将清单数据导出为 CSV 格式"""
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['文件路径', '文件名', '类别标签', '数据集划分', '文件大小(字节)', '文件大小(MB)', 'MD5哈希', '宽度', '高度'])
        for item in manifest:
            writer.writerow([
                item['file_path'], item['file_name'], item['class_label'],
                item['split'], item['file_size_bytes'], item['file_size_mb'],
                item['md5'], item['width'], item['height']
            ])
    print(f'  CSV清单已导出: {csv_path}')


# ============================================================================
#  六、数据集快照与版本管理
# ============================================================================

def create_dataset_snapshot(source=None, snapshot_name=None):
    """创建数据集快照，记录当前数据集的完整文件结构和MD5哈希。
    
    用于数据集版本管理，支持后续差异比较。
    
    Args:
        source: 数据源路径
        snapshot_name: 快照名称，默认为 snapshot_时间戳
    
    Returns:
        str: 快照文件路径
    """
    if source is None:
        source = PROCESSED_DATA_DIR
    if snapshot_name is None:
        snapshot_name = f'snapshot_{time.strftime("%Y%m%d_%H%M%S")}'
    snapshot_path = os.path.join(SNAPSHOT_DIR, f'{snapshot_name}.json')
    print('\n' + '=' * 60)
    print('  创建数据集快照')
    print('=' * 60)
    print(f'  数据源: {source}')
    print(f'  快照名称: {snapshot_name}')
    snapshot = {
        'snapshot_name': snapshot_name,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'created_timestamp': time.time(),
        'source_path': os.path.abspath(source),
        'structure': {}, 'files': [],
        'total_files': 0, 'total_size_bytes': 0
    }
    file_count = 0
    total_size = 0
    for root, dirs, files in sorted(os.walk(source)):
        rel_dir = os.path.relpath(root, source)
        if rel_dir == '.':
            rel_dir = ''
        dir_files = []
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel_path = os.path.join(rel_dir, fname) if rel_dir else fname
            try:
                file_size = os.path.getsize(fpath)
                md5_hash = _calculate_file_hash(fpath)
                dir_files.append({'file_name': fname, 'relative_path': rel_path,
                                  'file_size_bytes': file_size, 'md5': md5_hash})
                file_count += 1
                total_size += file_size
            except Exception as e:
                print(f'  [警告] 跳过文件 {rel_path}: {e}')
        if dir_files:
            snapshot['structure'][rel_dir if rel_dir else '/'] = {
                'file_count': len(dir_files), 'files': dir_files
            }
        if file_count % 500 == 0 and file_count > 0:
            print(f'  已处理 {file_count} 个文件...')
    snapshot['total_files'] = file_count
    snapshot['total_size_bytes'] = total_size
    snapshot['total_size_mb'] = round(total_size / (1024 * 1024), 2)
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f'  记录文件数: {file_count}')
    print(f'  数据集总大小: {snapshot["total_size_mb"]:.2f} MB')
    print(f'  快照已保存: {snapshot_path}')
    print('=' * 60)
    return snapshot_path


def compare_snapshots(snap1, snap2):
    """比较两个数据集快照的差异，识别新增、删除和修改的文件。
    
    Args:
        snap1: 快照1（路径字符串或已加载的字典）
        snap2: 快照2（路径字符串或已加载的字典）
    
    Returns:
        dict: 包含 added_files, removed_files, modified_files 等字段
    """
    if isinstance(snap1, str):
        with open(snap1, 'r', encoding='utf-8') as f:
            snap1 = json.load(f)
    if isinstance(snap2, str):
        with open(snap2, 'r', encoding='utf-8') as f:
            snap2 = json.load(f)
    snap1_files = {}
    if snap1.get('files'):
        for f in snap1['files']:
            snap1_files[f['relative_path']] = f['md5']
    else:
        for dir_info in snap1.get('structure', {}).values():
            for f in dir_info.get('files', []):
                snap1_files[f['relative_path']] = f['md5']
    snap2_files = {}
    if snap2.get('files'):
        for f in snap2['files']:
            snap2_files[f['relative_path']] = f['md5']
    else:
        for dir_info in snap2.get('structure', {}).values():
            for f in dir_info.get('files', []):
                snap2_files[f['relative_path']] = f['md5']
    paths1 = set(snap1_files.keys())
    paths2 = set(snap2_files.keys())
    common_paths = paths1 & paths2
    added_paths = paths2 - paths1
    removed_paths = paths1 - paths2
    added_files = [{'relative_path': p, 'md5': snap2_files[p]} for p in sorted(added_paths)]
    removed_files = [{'relative_path': p, 'md5': snap1_files[p]} for p in sorted(removed_paths)]
    modified_files = [
        {'relative_path': p, 'md5_before': snap1_files[p], 'md5_after': snap2_files[p]}
        for p in sorted(common_paths) if snap1_files[p] != snap2_files[p]
    ]
    unchanged_files = len(common_paths) - len(modified_files)
    print('\n' + '=' * 60)
    print('  快照差异比较')
    print('=' * 60)
    print(f'  快照1: {snap1.get("snapshot_name", "")} ({snap1.get("created_at", "")})')
    print(f'  快照2: {snap2.get("snapshot_name", "")} ({snap2.get("created_at", "")})')
    print(f'\n  差异摘要:')
    print(f'    新增文件: {len(added_files)}')
    print(f'    删除文件: {len(removed_files)}')
    print(f'    修改文件: {len(modified_files)}')
    print(f'    未变文件: {unchanged_files}')
    print(f'    快照1总文件数: {len(paths1)}')
    print(f'    快照2总文件数: {len(paths2)}')
    print('=' * 60)
    return {
        'added_files': added_files, 'removed_files': removed_files,
        'modified_files': modified_files, 'unchanged_files': unchanged_files,
        'snap1_info': {
            'name': snap1.get('snapshot_name', ''),
            'created_at': snap1.get('created_at', ''),
            'total_files': snap1.get('total_files', len(paths1)),
            'total_size_mb': snap1.get('total_size_mb', 0)
        },
        'snap2_info': {
            'name': snap2.get('snapshot_name', ''),
            'created_at': snap2.get('created_at', ''),
            'total_files': snap2.get('total_files', len(paths2)),
            'total_size_mb': snap2.get('total_size_mb', 0)
        }
    }


def list_snapshots():
    """列出所有已创建的数据集快照。
    
    Returns:
        list: 快照信息列表，每项包含 name, created_at, total_files, total_size_mb
    """
    if not os.path.exists(SNAPSHOT_DIR):
        return []
    snapshots = []
    for fname in sorted(os.listdir(SNAPSHOT_DIR)):
        if fname.endswith('.json'):
            fpath = os.path.join(SNAPSHOT_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    snap = json.load(f)
                snapshots.append({
                    'name': snap.get('snapshot_name', fname.replace('.json', '')),
                    'created_at': snap.get('created_at', ''),
                    'total_files': snap.get('total_files', 0),
                    'total_size_mb': snap.get('total_size_mb', 0)
                })
            except Exception:
                snapshots.append({'name': fname, 'created_at': '', 'total_files': 0, 'total_size_mb': 0})
    return snapshots


# ============================================================================
#  七、训练历史日志管理
# ============================================================================

def log_training_session(session_data):
    """记录一次训练会话的指标数据。
    
    Args:
        session_data: 包含训练指标（best_accuracy, precision, recall, f1, train_loss, val_loss 等）的字典
    
    Returns:
        dict: 记录的训练会话条目
    """
    if os.path.exists(TRAINING_LOG_PATH):
        with open(TRAINING_LOG_PATH, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {'sessions': [], 'total_sessions': 0}
    session_entry = {
        'session_id': len(history['sessions']) + 1,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'timestamp_unix': time.time(),
        **session_data
    }
    history['sessions'].append(session_entry)
    history['total_sessions'] = len(history['sessions'])
    with open(TRAINING_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'  训练会话已记录: session_id={session_entry["session_id"]}')
    return session_entry


def get_training_history():
    """获取所有训练历史记录。
    
    Returns:
        dict: 包含 sessions 列表和 total_sessions 计数的训练历史
    """
    if not os.path.exists(TRAINING_LOG_PATH):
        return {'sessions': [], 'total_sessions': 0}
    with open(TRAINING_LOG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_training_sessions(session_ids=None):
    """比较多个训练会话的性能指标，找出各指标的最佳会话。
    
    Args:
        session_ids: 要比较的会话ID列表，为 None 时比较所有会话
    
    Returns:
        dict: 各指标的比较结果
    """
    history = get_training_history()
    sessions = history.get('sessions', [])
    if session_ids:
        sessions = [s for s in sessions if s['session_id'] in session_ids]
    if len(sessions) < 2:
        print('  需要至少2个训练会话进行比较')
        return None
    comparison = {
        'sessions_compared': [s['session_id'] for s in sessions],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics_comparison': []
    }
    metric_keys = ['best_accuracy', 'precision', 'recall', 'f1', 'train_loss', 'val_loss']
    for key in metric_keys:
        values = []
        for s in sessions:
            val = s.get(key)
            if val is not None:
                values.append({'session_id': s['session_id'], 'value': val})
        if values:
            best_session = max(values, key=lambda x: x['value']) if key != 'train_loss' and key != 'val_loss' else min(values, key=lambda x: x['value'])
            comparison['metrics_comparison'].append({
                'metric': key,
                'values': values,
                'best_session': best_session['session_id'],
                'range': f'{min(v["value"] for v in values):.4f} - {max(v["value"] for v in values):.4f}'
            })
    print('\n  训练会话比较:')
    for mc in comparison['metrics_comparison']:
        print(f'    {mc["metric"]}: 最佳=session_{mc["best_session"]}, 范围={mc["range"]}')
    return comparison


# ============================================================================
#  八、数据增强追踪
# ============================================================================

def log_augmentation(aug_data):
    """记录数据增强操作日志。
    
    Args:
        aug_data: 包含 original_count, augmented_count 等字段的字典
    
    Returns:
        dict: 记录的增强条目
    """
    if os.path.exists(AUGMENTATION_LOG_PATH):
        with open(AUGMENTATION_LOG_PATH, 'r', encoding='utf-8') as f:
            aug_log = json.load(f)
    else:
        aug_log = {'records': [], 'total_records': 0, 'statistics': {}}
    record = {
        'record_id': len(aug_log['records']) + 1,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        **aug_data
    }
    aug_log['records'].append(record)
    aug_log['total_records'] = len(aug_log['records'])
    total_original = sum(r.get('original_count', 0) for r in aug_log['records'])
    total_augmented = sum(r.get('augmented_count', 0) for r in aug_log['records'])
    aug_log['statistics'] = {
        'total_original_images': total_original,
        'total_augmented_images': total_augmented,
        'total_output_images': total_original + total_augmented,
        'augmentation_ratio': round(total_augmented / total_original, 2) if total_original > 0 else 0
    }
    with open(AUGMENTATION_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(aug_log, f, ensure_ascii=False, indent=2)
    return record


def get_augmentation_summary():
    """获取数据增强汇总统计。
    
    Returns:
        dict: 增强统计信息，包含原始图片数、增强图片数、增强比例
    """
    if not os.path.exists(AUGMENTATION_LOG_PATH):
        return None
    with open(AUGMENTATION_LOG_PATH, 'r', encoding='utf-8') as f:
        aug_log = json.load(f)
    return aug_log.get('statistics', {})


# ============================================================================
#  九、数据溯源管理
# ============================================================================

def init_data_lineage(source_description=None):
    """初始化数据溯源信息，记录数据集的来源和处理步骤。
    
    遵循数据血缘追踪最佳实践，记录从原始采集到最终训练数据的完整处理链。
    
    Args:
        source_description: 数据来源描述
    
    Returns:
        dict: 数据溯源信息
    """
    lineage = {
        'dataset_name': '动物识别数据集',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source_description': source_description or '通过Bing图片搜索引擎爬取，经三级数据准确性保障机制过滤',
        'processing_steps': [
            {'step': 1, 'name': '网络爬虫采集', 'description': '使用Python Requests库模拟浏览器请求，从Bing图片搜索获取原始图像', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
            {'step': 2, 'name': '第一级过滤：关键词优化', 'description': '构建包含真实性约束的复合检索短语，从源头减少噪声数据', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
            {'step': 3, 'name': '第二级过滤：文件完整性校验', 'description': '基于Pillow库校验图像文件头部魔数和可解码性', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
            {'step': 4, 'name': '第三级过滤：人工抽检复审', 'description': '人工随机抽检各类别样本，确认标签语义正确性', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
            {'step': 5, 'name': '数据集划分', 'description': '按90.4%:4.8%:4.8%比例划分为训练集、验证集、测试集', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
            {'step': 6, 'name': '数据增强', 'description': '对训练集应用随机裁剪、水平翻转、归一化等增强操作', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
        ],
        'data_characteristics': {
            'num_classes': 8,
            'class_names_zh': CLASSES_ZH,
            'class_names_en': CLASSES_EN,
            'image_format': 'JPEG',
            'input_resolution': '224×224',
            'color_mode': 'RGB'
        }
    }
    with open(DATA_LINEAGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(lineage, f, ensure_ascii=False, indent=2)
    print(f'  数据溯源信息已初始化: {DATA_LINEAGE_PATH}')
    return lineage


def add_lineage_step(step_name, description):
    """向数据溯源记录中添加新的处理步骤。
    
    Args:
        step_name: 步骤名称
        description: 步骤描述
    
    Returns:
        dict: 更新后的数据溯源信息
    """
    if not os.path.exists(DATA_LINEAGE_PATH):
        init_data_lineage()
    with open(DATA_LINEAGE_PATH, 'r', encoding='utf-8') as f:
        lineage = json.load(f)
    new_step = {
        'step': len(lineage['processing_steps']) + 1,
        'name': step_name,
        'description': description,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    lineage['processing_steps'].append(new_step)
    with open(DATA_LINEAGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(lineage, f, ensure_ascii=False, indent=2)
    return lineage


def get_data_lineage():
    """获取数据溯源信息。
    
    Returns:
        dict: 数据溯源信息，如果文件不存在返回 None
    """
    if not os.path.exists(DATA_LINEAGE_PATH):
        return None
    with open(DATA_LINEAGE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
#  十、综合数据管理面板数据生成
# ============================================================================

def generate_dashboard_data():
    """生成综合数据管理面板所需的所有数据。
    
    聚合数据集摘要、质量评分、增强统计、快照列表、训练历史、类别平衡等数据。
    
    Returns:
        dict: 综合仪表盘数据
    """
    summary = get_dataset_summary()
    quality_score = None
    quality_log_exists = os.path.exists(QUALITY_LOG_PATH)
    if quality_log_exists:
        with open(QUALITY_LOG_PATH, 'r', encoding='utf-8') as f:
            quality_log = json.load(f)
        quality_score = {
            'valid_rate': quality_log.get('valid_rate', 100),
            'corrupt_count': quality_log.get('corrupt_count', 0),
            'warning_count': quality_log.get('warning_count', 0)
        }
    aug_summary = get_augmentation_summary()
    snapshots = list_snapshots()
    training_history = get_training_history()
    balance = check_class_balance()
    dashboard = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_summary': summary,
        'quality': quality_score,
        'augmentation': aug_summary,
        'snapshots': snapshots,
        'training_sessions': len(training_history.get('sessions', [])),
        'latest_training': training_history['sessions'][-1] if training_history.get('sessions') else None,
        'class_balance': {
            'is_balanced': balance.get('is_balanced', True),
            'max_ratio': balance.get('max_ratio', 1.0),
            'recommendations': balance.get('recommendations', [])
        }
    }
    dashboard_path = os.path.join(REPORT_DIR, 'dashboard_data.json')
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2)
    return dashboard


# ============================================================================
#  内部辅助函数
# ============================================================================

def _calculate_file_hash(file_path):
    """计算文件的 MD5 哈希值。
    
    Args:
        file_path: 文件路径
    
    Returns:
        str: MD5 哈希值（32位十六进制字符串），失败返回 None
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None


def _batch_verify_integrity(base_path):
    """内部函数：批量验证数据集中所有图片文件的完整性。
    
    Args:
        base_path: 数据集根目录路径
    
    Returns:
        list: 损坏文件列表 [(文件路径, 错误信息)]
    """
    corrupt_files = []
    total = 0
    for split in DATASET_SPLITS:
        split_dir = os.path.join(base_path, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    continue
                fpath = os.path.join(class_dir, fname)
                total += 1
                result = verify_image(fpath)
                if not result['valid']:
                    corrupt_files.append((fpath, result['error']))
    return corrupt_files


# ============================================================================
#  演示入口
# ============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('  论文识别系统 — 训练数据管理工具')
    print('=' * 60)

    print('\n[功能1] 扫描数据集统计...')
    stats = scan_dataset()
    for cls_name, cls_data in sorted(stats.items()):
        detail = ', '.join(f'{s}: {cls_data.get(s, 0)}张' for s in DATASET_SPLITS if cls_data.get(s, 0) > 0)
        print(f'  {cls_name}: {cls_data["total"]}张 ({detail})')

    print('\n[功能2] 类别平衡性检查...')
    balance_result = check_class_balance(stats)

    print('\n[功能3] 生成数据集报告...')
    if os.path.exists(PROCESSED_DATA_DIR):
        report = generate_dataset_report(check_integrity=False)

    print('\n[功能4] 初始化数据溯源...')
    init_data_lineage()

    print('\n[功能5] 生成仪表盘数据...')
    dashboard = generate_dashboard_data()
    print(f'  仪表盘数据已生成: {os.path.join(REPORT_DIR, "dashboard_data.json")}')