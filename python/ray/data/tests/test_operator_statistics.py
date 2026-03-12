"""
Ray Data CBO Phase 1: 完整的单元测试

源码位置：python/ray/data/tests/test_operator_statistics.py

测试覆盖：
1. OperatorStatistics 数据结构
2. ColumnStatistics 基础操作
3. 统计缩放（scale）
4. 选择率估算
5. 统计合并（merge）
"""

import pytest
from ray.data._internal.cbo_stats.operator_statistics import (
    OperatorStatistics,
    ColumnStatistics,
    ConfidenceLevel,
    estimate_selectivity_from_column_stats,
)


class TestColumnStatistics:
    """ColumnStatistics 的单元测试"""

    def test_create_column_statistics(self):
        """测试基础的列统计创建"""
        col = ColumnStatistics(
            name='age',
            min_value=0,
            max_value=100,
            null_count=5,
            distinct_count=50,
        )

        assert col.name == 'age'
        assert col.min_value == 0
        assert col.max_value == 100
        assert col.null_count == 5
        assert col.distinct_count == 50
        assert col.is_available()

    def test_column_statistics_with_invalid_data(self):
        """测试处理无效数据"""
        # 负数的 null_count 应该被纠正
        col = ColumnStatistics(
            name='col1',
            null_count=-10,  # 无效
        )
        assert col.null_count == 0

        # 负数的 distinct_count 应该被设置为 None
        col2 = ColumnStatistics(
            name='col2',
            distinct_count=-5,  # 无效
        )
        assert col2.distinct_count is None

    def test_column_statistics_to_dict(self):
        """测试序列化"""
        col = ColumnStatistics(
            name='col1',
            min_value=10,
            max_value=100,
            null_count=5,
        )
        d = col.to_dict()

        assert d['name'] == 'col1'
        assert d['min_value'] == 10
        assert d['max_value'] == 100
        assert d['null_count'] == 5

    def test_column_statistics_from_dict(self):
        """测试反序列化"""
        data = {
            'name': 'col1',
            'min_value': 10,
            'max_value': 100,
            'null_count': 5,
            'distinct_count': 50,
            'encoding': 'PLAIN',
            'compression': 'SNAPPY',
            'confidence': 0.95,
        }
        col = ColumnStatistics.from_dict(data)

        assert col.name == 'col1'
        assert col.min_value == 10
        assert col.max_value == 100
        assert col.confidence == 0.95


class TestOperatorStatistics:
    """OperatorStatistics 的单元测试"""

    def test_create_operator_statistics(self):
        """测试基础的算子统计创建"""
        stats = OperatorStatistics(
            num_rows=1000,
            size_bytes=1024 * 1024,
            selectivity=0.5,
            confidence=ConfidenceLevel.HIGH.value,
        )

        assert stats.num_rows == 1000
        assert stats.size_bytes == 1024 * 1024
        assert stats.selectivity == 0.5
        assert stats.confidence == ConfidenceLevel.HIGH.value
        assert stats.is_available()

    def test_operator_statistics_avg_row_bytes_calculation(self):
        """测试自动计算 avg_row_bytes"""
        stats = OperatorStatistics(
            num_rows=1000,
            size_bytes=1024 * 1024,  # 1MB = 1024*1024 字节
        )

        # 应该自动计算：1MB / 1000 rows = 1024 bytes/row
        assert stats.avg_row_bytes == 1024

    def test_operator_statistics_scale(self):
        """测试统计缩放（如应用选择率）"""
        original = OperatorStatistics(
            num_rows=1000,
            size_bytes=1024 * 1024,
            avg_row_bytes=1024,
            confidence=ConfidenceLevel.HIGH.value,
        )

        # 应用 50% 的选择率（如 Filter）
        scaled = original.scale(0.5)

        assert scaled.num_rows == 500  # 1000 * 0.5
        assert scaled.size_bytes == 512 * 1024  # 1MB * 0.5
        assert scaled.avg_row_bytes == 1024  # 平均行大小不变
        assert scaled.selectivity == 0.5
        # 置信度应该下降
        assert scaled.confidence < original.confidence

    def test_operator_statistics_scale_with_zero_rows(self):
        """测试缩放时的边界情况"""
        stats = OperatorStatistics(num_rows=None, size_bytes=None)
        scaled = stats.scale(0.5)

        assert scaled.num_rows is None
        assert scaled.size_bytes is None

    def test_operator_statistics_zero(self):
        """测试创建零数据统计"""
        zero_stats = OperatorStatistics.zero()

        assert zero_stats.num_rows == 0
        assert zero_stats.size_bytes == 0
        assert zero_stats.confidence == ConfidenceLevel.EXACT.value

    def test_operator_statistics_unknown(self):
        """测试创建未知统计"""
        unknown_stats = OperatorStatistics.unknown()

        assert unknown_stats.num_rows is None
        assert unknown_stats.size_bytes is None
        assert unknown_stats.confidence == ConfidenceLevel.UNKNOWN.value

    def test_operator_statistics_merge(self):
        """测试统计合并（如 Union 操作）"""
        stats1 = OperatorStatistics(
            num_rows=1000,
            size_bytes=1024 * 1024,
            confidence=ConfidenceLevel.HIGH.value,
        )

        stats2 = OperatorStatistics(
            num_rows=500,
            size_bytes=512 * 1024,
            confidence=ConfidenceLevel.HIGH.value,
        )

        merged = stats1.merge(stats2)

        # 行数应该求和
        assert merged.num_rows == 1500
        # 字节数应该求和
        assert merged.size_bytes == 1536 * 1024
        # 置信度取较小值（保守）
        assert merged.confidence == ConfidenceLevel.HIGH.value

    def test_operator_statistics_merge_with_none(self):
        """测试与 None 的合并"""
        stats = OperatorStatistics(num_rows=1000, size_bytes=1024 * 1024)
        merged = stats.merge(None)

        assert merged.num_rows == 1000

    def test_operator_statistics_to_log_string(self):
        """测试日志字符串格式"""
        stats = OperatorStatistics(
            num_rows=1000,
            size_bytes=1024 * 1024 * 1024,  # 1GB
            avg_row_bytes=1024,
            selectivity=0.5,
            confidence=0.95,
        )

        log_str = stats.to_log_string()

        assert 'rows=1000' in log_str
        assert 'size=1.00GB' in log_str
        assert 'avg_row=1024B' in log_str
        assert 'selectivity=50.00%' in log_str
        assert 'confidence=95.0%' in log_str

    def test_operator_statistics_with_column_stats(self):
        """测试包含列统计的算子统计"""
        col = ColumnStatistics(
            name='age',
            min_value=0,
            max_value=100,
            distinct_count=50,
        )

        stats = OperatorStatistics(
            num_rows=1000,
            size_bytes=100 * 1024,
            column_stats={'age': col},
        )

        assert len(stats.column_stats) == 1
        assert stats.get_column_stat('age') is not None
        assert stats.get_column_stat('age').distinct_count == 50
        assert stats.get_column_stat('nonexistent') is None


class TestSelectivityEstimation:
    """选择率估算的单元测试"""

    def test_selectivity_eq(self):
        """测试等值谓词的选择率"""
        col = ColumnStatistics(
            name='col1',
            min_value=0,
            max_value=100,
            distinct_count=50,
        )

        # 等值选择率 = 1 / NDV
        sel = estimate_selectivity_from_column_stats('col1', 'EQ', 50, col)
        assert abs(sel - 1 / 50) < 0.01

    def test_selectivity_gt(self):
        """测试大于谓词的选择率"""
        col = ColumnStatistics(
            name='age',
            min_value=0,
            max_value=100,
            distinct_count=100,
        )

        # age > 50: 应该约 50%
        sel = estimate_selectivity_from_column_stats('age', 'GT', 50, col)
        assert 0.45 < sel < 0.55

    def test_selectivity_lt(self):
        """测试小于谓词的选择率"""
        col = ColumnStatistics(
            name='age',
            min_value=0,
            max_value=100,
        )

        # age < 25: 应该约 25%
        sel = estimate_selectivity_from_column_stats('age', 'LT', 25, col)
        assert 0.2 < sel < 0.3

    def test_selectivity_in(self):
        """测试 IN 谓词的选择率"""
        col = ColumnStatistics(
            name='category',
            distinct_count=10,
        )

        # IN (3 个值): 3/10 = 30%
        sel = estimate_selectivity_from_column_stats('category', 'IN', ['A', 'B', 'C'], col)
        assert abs(sel - 0.3) < 0.01

    def test_selectivity_no_stats(self):
        """测试没有统计时的降级行为"""
        # 没有列统计信息
        sel = estimate_selectivity_from_column_stats('col1', 'EQ', 50, None)

        # 应该返回保守值
        assert sel == 0.5

    def test_selectivity_invalid_operator(self):
        """测试无效操作符"""
        col = ColumnStatistics(name='col1', min_value=0, max_value=100)
        sel = estimate_selectivity_from_column_stats('col1', 'UNKNOWN_OP', 50, col)

        # 应该返回保守值
        assert sel == 0.5


class TestOperatorStatisticsIntegration:
    """集成测试：模拟完整的数据流统计传播"""

    def test_filter_push_down_statistics(self):
        """
        测试场景：Filter 算子推导统计

        管线：Read → Filter(age > 30) → Project(name, age)
        """
        # Step 1: Read 得到基础统计
        read_stats = OperatorStatistics(
            num_rows=10000,
            size_bytes=10 * 1024 * 1024,  # 10MB
            column_stats={
                'age': ColumnStatistics(
                    name='age',
                    min_value=0,
                    max_value=100,
                    distinct_count=100,
                ),
                'name': ColumnStatistics(name='name'),
            },
        )

        # Step 2: Filter 应用选择率
        # age > 30 的选择率约 70%
        filter_selectivity = estimate_selectivity_from_column_stats(
            'age', 'GT', 30,
            read_stats.get_column_stat('age'),
        )
        filter_stats = read_stats.scale(filter_selectivity)

        # 验证过滤后的统计
        assert filter_stats.num_rows == int(10000 * filter_selectivity)
        assert filter_stats.size_bytes == int(10 * 1024 * 1024 * filter_selectivity)
        assert filter_stats.selectivity == filter_selectivity

        # Step 3: Project(name, age)
        # 假设原始有 5 列，现在只选 2 列
        # 字节数应该约 2/5 = 40% 的原始大小
        project_stats = filter_stats.scale(2 / 5)

        assert project_stats.num_rows == filter_stats.num_rows  # 行数不变
        assert project_stats.size_bytes < filter_stats.size_bytes  # 字节数减少

    def test_confidence_degradation_through_pipeline(self):
        """
        测试场景：置信度在管线中逐步下降

        这反映了现实：预测的管线越长，不确定性越大
        """
        # Read 的置信度最高
        stats1 = OperatorStatistics(
            num_rows=1000,
            confidence=ConfidenceLevel.HIGH.value,  # 0.95
        )

        # Filter 应用选择率后，置信度下降
        stats2 = stats1.scale(0.5)
        assert stats2.confidence < stats1.confidence

        # 再应用一个 Filter
        stats3 = stats2.scale(0.5)
        assert stats3.confidence < stats2.confidence

        # 后续规则会看到越来越低的置信度，
        # 从而做出更保守的优化决策


class TestEdgeCases:
    """边界情况测试"""

    def test_scale_with_extreme_ratios(self):
        """测试极端的缩放比例"""
        stats = OperatorStatistics(num_rows=1000, size_bytes=1024 * 1024)

        # 非常小的缩放比例
        scaled_small = stats.scale(0.001)
        assert scaled_small.num_rows == 1  # int(1000 * 0.001)

        # 很大的缩放比例（如 explode）
        scaled_large = stats.scale(100)
        assert scaled_large.num_rows == 100000

    def test_statistics_with_very_large_data(self):
        """测试处理非常大的数据"""
        large_size = 1024 * 1024 * 1024 * 1024  # 1TB

        stats = OperatorStatistics(
            num_rows=1e12,  # 1 trillion rows
            size_bytes=large_size,
        )

        assert stats.is_available()
        log_str = stats.to_log_string()
        assert 'TB' in log_str or 'GB' in log_str

    def test_merge_with_partial_stats(self):
        """测试合并部分统计"""
        stats1 = OperatorStatistics(num_rows=1000, size_bytes=None)
        stats2 = OperatorStatistics(num_rows=None, size_bytes=500 * 1024)

        merged = stats1.merge(stats2)

        assert merged.num_rows == 1000
        assert merged.size_bytes == 500 * 1024


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
