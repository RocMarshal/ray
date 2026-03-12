"""
Ray Data CBO Phase 2: 完整的单元测试与集成测试

源码位置：python/ray/data/tests/test_cost_model.py

测试覆盖：
1. 代价估算的准确性
2. 预留比例推导的各种场景
3. 管线特征识别
4. 集成测试
"""

import pytest
from ray.data._internal.cbo_stats.cost_model import (
    OperatorCost,
    CostWeights,
    CostEstimator,
    PipelineProperties,
    ReservationRatioDeriver,
)


class TestOperatorCost:
    """OperatorCost 的单元测试"""

    def test_create_operator_cost(self):
        """测试基础代价创建"""
        cost = OperatorCost(
            cpu_cost=10.0,
            gpu_cost=5.0,
            peak_memory_bytes=1024 * 1024 * 1024,  # 1GB
        )

        assert cost.cpu_cost == 10.0
        assert cost.gpu_cost == 5.0
        assert cost.peak_memory_bytes == 1024 * 1024 * 1024

    def test_total_weighted_cost(self):
        """测试加权总代价计算"""
        cost = OperatorCost(
            cpu_cost=10.0,
            gpu_cost=2.0,
        )

        weights = CostWeights(cpu=1.0, gpu=10.0)
        total = cost.total_weighted_cost(weights)

        # total = 10 * 1.0 + 2 * 10.0 = 30
        assert total == 30.0

    def test_add_costs(self):
        """测试代价合并"""
        cost1 = OperatorCost(cpu_cost=5.0, peak_memory_bytes=100)
        cost2 = OperatorCost(cpu_cost=3.0, peak_memory_bytes=200)

        merged = cost1.add(cost2)

        assert merged.cpu_cost == 8.0
        assert merged.peak_memory_bytes == 200  # max(100, 200)

    def test_cost_weights_presets(self):
        """测试权重预设"""
        cpu_weights = CostWeights.cpu_intensive()
        assert cpu_weights.gpu == 0.0

        gpu_weights = CostWeights.gpu_intensive()
        assert gpu_weights.gpu > cpu_weights.gpu

        bandwidth_weights = CostWeights.bandwidth_constrained()
        assert bandwidth_weights.shuffle > gpu_weights.shuffle


class TestCostEstimator:
    """CostEstimator 的单元测试"""

    def test_estimate_read_cost(self):
        """测试 Read 算子的代价估算"""
        cost = CostEstimator.estimate_read_cost(
            num_rows=1_000_000,
            size_bytes=100 * 1024 * 1024,  # 100MB
        )

        assert cost.cpu_cost > 0
        assert cost.io_read_bytes == 100 * 1024 * 1024
        assert cost.object_store_bytes == 100 * 1024 * 1024

    def test_estimate_filter_cost(self):
        """测试 Filter 算子的代价估算"""
        cost = CostEstimator.estimate_filter_cost(
            num_rows=1_000_000,
            selectivity=0.5,
            input_size_bytes=100 * 1024 * 1024,
        )

        assert cost.cpu_cost > 0
        # 输出大小应该是输入的 50%
        assert cost.object_store_bytes == 50 * 1024 * 1024

    def test_estimate_map_cost_cpu(self):
        """测试 CPU Map 算子的代价估算"""
        cost = CostEstimator.estimate_map_cost(
            num_rows=1_000_000,
            amplification_ratio=2.0,
            input_size_bytes=100 * 1024 * 1024,
            has_gpu=False,
        )

        assert cost.cpu_cost > 0
        assert cost.gpu_cost == 0.0
        # 输出大小应该是输入的 2 倍
        assert cost.object_store_bytes == 200 * 1024 * 1024

    def test_estimate_map_cost_gpu(self):
        """测试 GPU Map 算子的代价估算"""
        cost = CostEstimator.estimate_map_cost(
            num_rows=1_000_000,
            amplification_ratio=1.0,
            input_size_bytes=100 * 1024 * 1024,
            has_gpu=True,
        )

        assert cost.cpu_cost == 0.0
        assert cost.gpu_cost > 0

    def test_estimate_join_cost(self):
        """测试 Join 算子的代价估算"""
        cost = CostEstimator.estimate_join_cost(
            left_rows=500_000,
            right_rows=500_000,
            left_size_bytes=50 * 1024 * 1024,
            right_size_bytes=50 * 1024 * 1024,
            output_rows=1_000_000,
            output_size_bytes=100 * 1024 * 1024,
        )

        assert cost.cpu_cost > 0
        # Join 内存应该是输入的 3 倍
        expected_memory = (50 + 50) * 1024 * 1024 * 3
        assert cost.peak_memory_bytes == expected_memory
        # Shuffle 成本应该是所有输入
        assert cost.shuffle_bytes == 100 * 1024 * 1024

    def test_estimate_sort_cost(self):
        """测试 Sort 算子的代价估算"""
        cost = CostEstimator.estimate_sort_cost(
            num_rows=1_000_000,
            size_bytes=100 * 1024 * 1024,
        )

        assert cost.cpu_cost > 0
        # Sort 内存应该是输入的 2 倍
        expected_memory = 100 * 1024 * 1024 * 2
        assert cost.peak_memory_bytes == expected_memory
        # Shuffle 成本
        assert cost.shuffle_bytes == 100 * 1024 * 1024

    def test_estimate_zero_rows(self):
        """测试零行数情况"""
        cost = CostEstimator.estimate_read_cost(
            num_rows=0,
            size_bytes=0,
        )

        assert cost.cpu_cost == 0.0
        assert cost.io_read_bytes == 0


class TestPipelineProperties:
    """PipelineProperties 的单元测试"""

    def test_long_pipeline_detection(self):
        """测试长管线识别"""
        props = PipelineProperties(num_stages=6)
        assert props.is_long_pipeline

        props2 = PipelineProperties(num_stages=5)
        assert not props2.is_long_pipeline

    def test_short_pipeline_detection(self):
        """测试短管线识别"""
        props = PipelineProperties(num_stages=2)
        assert props.is_short_pipeline

        props2 = PipelineProperties(num_stages=3)
        assert not props2.is_short_pipeline


class TestReservationRatioDeriver:
    """ReservationRatioDeriver 的单元测试"""

    def test_uniform_demand(self):
        """
        场景：所有算子需求相同

        预期：R 值应该接近基础值 0.3
        （因为没有算子竞争压力）
        """
        costs = {
            'op1': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
            'op2': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
            'op3': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
        }
        props = PipelineProperties(num_stages=3)

        r = ReservationRatioDeriver.derive(costs, props)

        # 不均衡系数 = 1.0，R 应该接近 0.3
        assert 0.25 < r < 0.35

    def test_imbalanced_demand(self):
        """
        场景：某个算子需求是其他的 10 倍

        预期：R 值应该更高（保证大需求算子不被饿死）
        """
        costs = {
            'op1': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
            'op2': OperatorCost(peak_memory_bytes=1000 * 1024 * 1024),  # 10 倍
            'op3': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
        }
        props = PipelineProperties(num_stages=3)

        r = ReservationRatioDeriver.derive(costs, props)

        # 应该比均衡情况高
        uniform_r = ReservationRatioDeriver.derive({
            'op1': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
            'op2': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
            'op3': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
        }, props)
        assert r > uniform_r

    def test_long_pipeline_with_gpu(self):
        """
        场景：长管线 + GPU 算子

        预期：R 值应该较高（长管线竞争 + GPU 计算密集）
        """
        costs = {
            f'op{i}': OperatorCost(peak_memory_bytes=100 * 1024 * 1024)
            for i in range(8)  # 8 算子
        }
        props = PipelineProperties(
            num_stages=8,
            has_gpu_ops=True,
        )

        r = ReservationRatioDeriver.derive(costs, props)

        # 应该相对较高
        assert r > 0.5

    def test_short_pipeline_no_gpu(self):
        """
        场景：短管线，无 GPU

        预期：R 值应该较低（竞争少）
        """
        costs = {
            'op1': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
            'op2': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
        }
        props = PipelineProperties(
            num_stages=2,
            has_gpu_ops=False,
        )

        r = ReservationRatioDeriver.derive(costs, props)

        # 应该较低
        assert 0.2 < r < 0.4

    def test_alltoall_pipeline(self):
        """
        场景：包含 AllToAll（Sort/Shuffle）

        预期：R 值应该较低
        （AllToAll 会解除内存限制）
        """
        costs = {
            'op1': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
            'sort': OperatorCost(peak_memory_bytes=500 * 1024 * 1024),
            'op3': OperatorCost(peak_memory_bytes=100 * 1024 * 1024),
        }
        props = PipelineProperties(
            num_stages=3,
            has_alltoall_ops=True,
        )

        r = ReservationRatioDeriver.derive(costs, props)

        # 应该较低
        assert r < 0.4

    def test_empty_costs(self):
        """测试空代价情况"""
        r = ReservationRatioDeriver.derive({}, PipelineProperties())

        # 应该返回默认值
        assert r == 0.5

    def test_boundary_constraints(self):
        """测试边界约束"""
        # 极端不均衡
        costs = {
            'op1': OperatorCost(peak_memory_bytes=1 * 1024 * 1024),
            'op2': OperatorCost(peak_memory_bytes=10000 * 1024 * 1024),
        }
        props = PipelineProperties(num_stages=2)

        r = ReservationRatioDeriver.derive(costs, props)

        # 应该被限制在 [0.1, 0.9] 范围内
        assert 0.1 <= r <= 0.9


class TestIntegration:
    """集成测试：模拟完整的代价估算和预留比例推导"""

    def test_standard_pipeline(self):
        """
        测试标准管线：Read → Filter → Map → Write
        """
        # 估算代价
        read_cost = CostEstimator.estimate_read_cost(
            num_rows=10_000_000,
            size_bytes=1000 * 1024 * 1024,  # 1GB
        )

        filter_cost = CostEstimator.estimate_filter_cost(
            num_rows=10_000_000,
            selectivity=0.5,
            input_size_bytes=1000 * 1024 * 1024,
        )

        map_cost = CostEstimator.estimate_map_cost(
            num_rows=5_000_000,
            amplification_ratio=1.5,
            input_size_bytes=500 * 1024 * 1024,
            has_gpu=False,
        )

        costs = {
            'read': read_cost,
            'filter': filter_cost,
            'map': map_cost,
        }

        props = PipelineProperties(num_stages=3)

        r = ReservationRatioDeriver.derive(costs, props)

        # R 应该在合理范围内
        assert 0.1 <= r <= 0.9

    def test_gpu_pipeline(self):
        """
        测试 GPU 管线：Read → Filter → GPU Map
        """
        costs = {
            'read': CostEstimator.estimate_read_cost(
                num_rows=1_000_000,
                size_bytes=100 * 1024 * 1024,
            ),
            'filter': CostEstimator.estimate_filter_cost(
                num_rows=1_000_000,
                selectivity=0.8,
                input_size_bytes=100 * 1024 * 1024,
            ),
            'gpu_map': CostEstimator.estimate_map_cost(
                num_rows=800_000,
                amplification_ratio=1.0,
                input_size_bytes=80 * 1024 * 1024,
                has_gpu=True,
            ),
        }

        props = PipelineProperties(
            num_stages=3,
            has_gpu_ops=True,
        )

        r = ReservationRatioDeriver.derive(costs, props)

        # GPU 管线应该推导出相对较高的 R
        assert r > 0.4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
