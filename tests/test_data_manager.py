"""
单元测试：数据管理模块 (data_manager.py)

测试覆盖：数据集扫描、图片验证、类别平衡分析、
MD5去重、快照管理、数据溯源
"""
import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDataManager(unittest.TestCase):
    """数据管理模块单元测试"""

    def setUp(self):
        """创建临时测试数据集"""
        self.temp_dir = tempfile.mkdtemp()
        # 创建模拟数据集结构
        for split in ['train', 'val', 'test']:
            for cls in ['cat', 'dog']:
                cls_dir = os.path.join(self.temp_dir, split, cls)
                os.makedirs(cls_dir, exist_ok=True)
                # 创建虚拟图片文件
                for i in range(3):
                    fpath = os.path.join(cls_dir, f'img_{i}.jpg')
                    with open(fpath, 'wb') as f:
                        f.write(b'\xff\xd8\xff\xe0' + b'\x00' * 100)

    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scan_dataset_structure(self):
        """测试数据集扫描：验证目录结构统计正确"""
        from data_manager import scan_dataset
        stats = scan_dataset(self.temp_dir)
        self.assertIn('cat', stats)
        self.assertIn('dog', stats)
        self.assertEqual(stats['cat']['train'], 3)
        self.assertEqual(stats['cat']['val'], 3)
        self.assertEqual(stats['cat']['test'], 3)
        self.assertEqual(stats['cat']['total'], 9)

    def test_get_dataset_summary(self):
        """测试数据集摘要：验证汇总信息正确"""
        from data_manager import get_dataset_summary
        summary = get_dataset_summary(self.temp_dir)
        self.assertEqual(summary['total_train'], 6)
        self.assertEqual(summary['total_val'], 6)
        self.assertEqual(summary['total_test'], 6)
        self.assertEqual(summary['total'], 18)
        self.assertEqual(len(summary['classes']), 2)

    def test_check_class_balance(self):
        """测试类别平衡性分析"""
        from data_manager import check_class_balance
        stats = {
            'cat': {'train': 50, 'val': 5, 'test': 5, 'total': 60},
            'dog': {'train': 50, 'val': 5, 'test': 5, 'total': 60},
        }
        result = check_class_balance(stats)
        self.assertTrue(result['is_balanced'])
        self.assertEqual(result['max_ratio'], 1.0)

    def test_check_class_imbalance(self):
        """测试类别不平衡检测"""
        from data_manager import check_class_balance
        stats = {
            'cat': {'train': 100, 'val': 10, 'test': 10, 'total': 120},
            'dog': {'train': 10, 'val': 1, 'test': 1, 'total': 12},
        }
        result = check_class_balance(stats)
        self.assertFalse(result['is_balanced'])
        self.assertGreater(result['max_ratio'], 3.0)

    def test_verify_image_valid(self):
        """测试图片验证：有效JPEG文件"""
        from data_manager import verify_image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'\xff\xd8\xff\xe0' + b'\x00' * 1000)
            temp_path = f.name
        try:
            result = verify_image(temp_path)
            self.assertTrue(result['valid'])
            self.assertIsNone(result['error'])
        finally:
            os.unlink(temp_path)

    def test_verify_image_nonexistent(self):
        """测试图片验证：不存在的文件"""
        from data_manager import verify_image
        result = verify_image('/nonexistent/file.jpg')
        self.assertFalse(result['valid'])
        self.assertEqual(result['error'], '文件不存在')

    def test_deduplicate_by_md5(self):
        """测试MD5去重：相同内容文件应被检测为重复"""
        from data_manager import deduplicate_by_md5
        import shutil
        # 创建两个相同内容的文件
        cls_dir = os.path.join(self.temp_dir, 'train', 'cat')
        with open(os.path.join(cls_dir, 'dup_1.jpg'), 'wb') as f:
            f.write(b'unique_duplicate_content_12345')
        shutil.copy2(
            os.path.join(cls_dir, 'dup_1.jpg'),
            os.path.join(cls_dir, 'dup_2.jpg')
        )
        duplicates = deduplicate_by_md5(self.temp_dir)
        self.assertGreater(len(duplicates), 0)

    def test_create_dataset_snapshot(self):
        """测试数据集快照创建"""
        from data_manager import create_dataset_snapshot
        import tempfile
        snapshot_dir = tempfile.mkdtemp()
        try:
            # 临时修改 SNAPSHOT_DIR
            import data_manager as dm
            original_snapshot_dir = dm.SNAPSHOT_DIR
            dm.SNAPSHOT_DIR = snapshot_dir
            try:
                snapshot_path = create_dataset_snapshot(
                    source=self.temp_dir,
                    snapshot_name='test_snapshot'
                )
                self.assertTrue(os.path.exists(snapshot_path))
                with open(snapshot_path, 'r', encoding='utf-8') as f:
                    snap = json.load(f)
                self.assertEqual(snap['snapshot_name'], 'test_snapshot')
                self.assertGreater(snap['total_files'], 0)
            finally:
                dm.SNAPSHOT_DIR = original_snapshot_dir
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    def test_init_data_lineage(self):
        """测试数据溯源初始化"""
        from data_manager import init_data_lineage
        import tempfile
        import data_manager as dm
        lineage_dir = tempfile.mkdtemp()
        original_path = dm.DATA_LINEAGE_PATH
        dm.DATA_LINEAGE_PATH = os.path.join(lineage_dir, 'data_lineage.json')
        try:
            lineage = init_data_lineage('测试数据源')
            self.assertEqual(lineage['dataset_name'], '动物识别数据集')
            self.assertEqual(len(lineage['processing_steps']), 6)
            self.assertTrue(os.path.exists(dm.DATA_LINEAGE_PATH))
        finally:
            dm.DATA_LINEAGE_PATH = original_path
            shutil.rmtree(lineage_dir, ignore_errors=True)

    def test_scan_dataset_empty(self):
        """测试空数据集扫描"""
        from data_manager import scan_dataset
        import tempfile
        empty_dir = tempfile.mkdtemp()
        try:
            stats = scan_dataset(empty_dir)
            self.assertEqual(stats, {})
        finally:
            os.rmdir(empty_dir)


class TestConfig(unittest.TestCase):
    """配置模块单元测试"""

    def test_classes_mapping(self):
        """测试中英文类别映射一致性"""
        from config import CLASSES_ZH, CLASSES_EN, ZH_TO_EN, EN_TO_ZH, NUM_CLASSES
        self.assertEqual(len(CLASSES_ZH), len(CLASSES_EN))
        self.assertEqual(NUM_CLASSES, len(CLASSES_ZH))
        for zh, en in zip(CLASSES_ZH, CLASSES_EN):
            self.assertEqual(ZH_TO_EN[zh], en)
            self.assertEqual(EN_TO_ZH[en], zh)

    def test_hyperparameters_defaults(self):
        """测试超参数默认值合理性"""
        from config import DEFAULT_LR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
        self.assertGreater(DEFAULT_LR, 0)
        self.assertLess(DEFAULT_LR, 0.1)
        self.assertGreater(DEFAULT_BATCH_SIZE, 0)
        self.assertGreater(DEFAULT_EPOCHS, 0)

    def test_valid_image_extensions(self):
        """测试有效图片扩展名配置"""
        from config import VALID_IMAGE_EXTENSIONS
        self.assertIn('.jpg', VALID_IMAGE_EXTENSIONS)
        self.assertIn('.png', VALID_IMAGE_EXTENSIONS)

    def test_palette_8_length(self):
        """测试8色调色板长度"""
        from config import PALETTE_8
        self.assertEqual(len(PALETTE_8), 8)


if __name__ == '__main__':
    unittest.main()