# -*- coding: utf-8 -*-
"""图像文件验证脚本：检查数据集完整性、格式合规性和文件损坏情况"""
import os
import json
import time
from collections import defaultdict

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] PIL not installed, skipping image integrity checks")


def validate_dataset(data_dir, backup_dir=None):
    """
    验证数据集完整性
    
    Args:
        data_dir: 数据集目录路径
        backup_dir: 备份目录路径（可选）
    
    Returns:
        dict: 验证报告
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    total_files = 0
    format_stats = defaultdict(int)
    corrupted_files = []
    invalid_format_files = []
    size_stats = {"min": float("inf"), "max": 0, "total": 0}
    dimension_issues = []
    per_class_stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                ext = os.path.splitext(fname)[1].lower()
                total_files += 1
                format_stats[ext] += 1
                per_class_stats[cls][split] += 1

                if ext not in valid_exts:
                    invalid_format_files.append(fpath)
                    continue

                if HAS_PIL:
                    try:
                        with Image.open(fpath) as img:
                            img.verify()
                        with Image.open(fpath) as img:
                            w, h = img.size
                            file_size = os.path.getsize(fpath)
                            size_stats["max"] = max(size_stats["max"], file_size)
                            size_stats["min"] = min(size_stats["min"], file_size)
                            size_stats["total"] += file_size
                            if w < 100 or h < 100:
                                dimension_issues.append((fpath, w, h))
                    except Exception as e:
                        corrupted_files.append((fpath, str(e)))

    all_ok = (len(corrupted_files) == 0 and len(invalid_format_files) == 0)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": data_dir,
        "total_images": total_files,
        "train_images": sum(v["train"] for v in per_class_stats.values()),
        "val_images": sum(v["val"] for v in per_class_stats.values()),
        "test_images": sum(v["test"] for v in per_class_stats.values()),
        "num_classes": len(per_class_stats),
        "per_class": dict(per_class_stats),
        "format_distribution": dict(format_stats),
        "corrupted_count": len(corrupted_files),
        "invalid_format_count": len(invalid_format_files),
        "dimension_issues_count": len(dimension_issues),
        "total_size_mb": round(size_stats["total"] / (1024*1024), 1) if size_stats["total"] > 0 else 0,
        "overall_status": "PASS" if all_ok else "WARN"
    }

    return report


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '动物识别数据集', 'data')
    report = validate_dataset(data_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))