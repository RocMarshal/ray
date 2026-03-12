"""
Ray Data CBO Phase 2: 代价模型与预留比例推导

源码位置：python/ray/data/_internal/stats/cost_model.py

功能：
1. OperatorCost 代价向量定义
2. CostEstimator 代价估算引擎
3. ReservationRatioDeriver 预留比例推导算法
4. 代价权重管理
"""

from dataclasses import dataclass
from typing import Optional, Dict
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class OperatorCost:
    """
    单算子的代价向量

    设计说明：
    - 使用向量代价而非标量代价（支持异构资源）
    - 每个维度都是可选的（某些算子可能不消耗 GPU）
    - 支持加权总代价计算（用于决策）
    """

    # === 计算资源 ===
    cpu_cost: float = 0.0          # CPU 核 × 秒
    gpu_cost: float = 0.0          # GPU 核 × 秒

    # === 存储和传输 ===
    io_read_bytes: float = 0.0     # 读取字节数
    io_write_bytes: float = 0.0    # 写入字节数
    peak_memory_bytes: float = 0.0  # 峰值内存占用
    object_store_bytes: float = 0.0  # Object Store 占用

    # === 网络传输（Shuffle） ===
    shuffle_bytes: float = 0.0     # 跨节点传输字节数

    def total_weighted_cost(self, weights: Optional['CostWeights'] = None) -> float:
        """
        计算加权总代价（用于决策）

        权重反映了不同资源的相对重要性：
        - GPU 比 CPU 贵 (10 倍)
        - 网络传输最贵 (5 倍)
        - IO 和内存成本较低
        """
        w = weights or CostWeights()

        return (
            self.cpu_cost * w.cpu +
            self.gpu_cost * w.gpu +
            self.io_read_bytes * w.io_read +
            self.io_write_bytes * w.io_write +
            self.peak_memory_bytes * w.memory +
            self.object_store_bytes * w.object_store +
            self.shuffle_bytes * w.shuffle
        )

    def add(self, other: 'OperatorCost') -> 'OperatorCost':
        """
        合并两个代价（用于管线总代价）

        说明：
        - CPU/GPU/IO: 求和（算子并发执行）
        - peak_memory: 取 max（算子轮流执行，峰值是最高的）
        """
        return OperatorCost(
            cpu_cost=self.cpu_cost + other.cpu_cost,
            gpu_cost=self.gpu_cost + other.gpu_cost,
            io_read_bytes=self.io_read_bytes + other.io_read_bytes,
            io_write_bytes=self.io_write_bytes + other.io_write_bytes,
            peak_memory_bytes=max(self.peak_memory_bytes, other.peak_memory_bytes),
            object_store_bytes=self.object_store_bytes + other.object_store_bytes,
            shuffle_bytes=self.shuffle_bytes + other.shuffle_bytes,
        )

    def to_dict(self) -> Dict[str, float]:
        """转换为字典（用于日志）"""
        return {
            'cpu_cost': round(self.cpu_cost, 2),
            'gpu_cost': round(self.gpu_cost, 2),
            'io_read_bytes': self.io_read_bytes,
            'io_write_bytes': self.io_write_bytes,
            'peak_memory_gb': round(self.peak_memory_bytes / (1024**3), 2),
            'object_store_gb': round(self.object_store_bytes / (1024**3), 2),
            'shuffle_gb': round(self.shuffle_bytes / (1024**3), 2),
        }

    def to_log_string(self) -> str:
        """格式化为日志字符串"""
        parts = []

        if self.cpu_cost > 0:
            parts.append(f"CPU={self.cpu_cost:.1f}")

        if self.gpu_cost > 0:
            parts.append(f"GPU={self.gpu_cost:.1f}")

        if self.peak_memory_bytes > 0:
            memory_gb = self.peak_memory_bytes / (1024**3)
            parts.append(f"Memory={memory_gb:.2f}GB")

        if self.shuffle_bytes > 0:
            shuffle_gb = self.shuffle_bytes / (1024**3)
            parts.append(f"Shuffle={shuffle_gb:.2f}GB")

        return "[" + ", ".join(parts) + "]"


@dataclass
class CostWeights:
    """
    代价权重，可根据集群特征调整

    权重定义了不同资源的相对成本。
    可以根据实际集群配置调整这些权重。
    """

    cpu: float = 1.0           # CPU 成本系数
    gpu: float = 10.0          # GPU 比 CPU 贵 10 倍
    io_read: float = 0.001     # 读取开销
    io_write: float = 0.002    # 写入开销（比读高）
    memory: float = 0.0001     # 内存开销（最便宜）
    object_store: float = 0.0002  # Object Store 开销
    shuffle: float = 0.005     # 网络传输最贵（跨节点）

    @classmethod
    def cpu_intensive(cls) -> 'CostWeights':
        """CPU 密集型集群的权重"""
        return CostWeights(
            cpu=1.0,
            gpu=0.0,  # 没有 GPU
            shuffle=0.01,  # 网络相对贵
        )

    @classmethod
    def gpu_intensive(cls) -> 'CostWeights':
        """GPU 密集型集群的权重"""
        return CostWeights(
            cpu=0.1,
            gpu=20.0,  # GPU 非常贵
            memory=0.00001,  # 内存便宜
        )

    @classmethod
    def bandwidth_constrained(cls) -> 'CostWeights':
        """网络受限集群的权重"""
        return CostWeights(
            shuffle=0.02,  # 网络非常贵
            io_read=0.005,
            io_write=0.01,
        )


class CostEstimator:
    """
    代价估算引擎

    基于 OperatorStatistics 计算每个算子的代价向量。
    使用经验公式和启发式方法进行估算。

    设计说明：
    - 代价系数基于经验值，可在 Phase 3 通过反馈调整
    - 某些算子（如 Map）需要用户提示（amplification_ratio）
    - 无法精确估算时使用保守的默认值
    """

    # === 代价系数（可调参数）===
    # Read 的 CPU 成本（单位：核-秒 / 1M 行）
    READ_CPU_PER_M_ROWS = 0.01

    # Filter 的 CPU 成本（单位：核-秒 / 1M 行）
    FILTER_CPU_PER_M_ROWS = 0.005

    # Map 的 CPU 成本（单位：核-秒 / 1M 行），假设普通 UDF
    MAP_CPU_PER_M_ROWS = 0.1

    # GPU Map 的 GPU 成本（单位：核-秒 / 1M 行）
    GPU_MAP_GPU_PER_M_ROWS = 0.05

    # Join 的 CPU 成本（单位：核-秒 / 1M 行）
    JOIN_CPU_PER_M_ROWS = 0.1

    # Join 的内存（单位：倍数）
    # 参考：HashJoin 需要 shuffle_memory + join_memory(2x PyArrow) + output
    JOIN_MEMORY_MULTIPLIER = 3.0

    # Sort 的 CPU 成本（单位：核-秒 / 1M 行）
    SORT_CPU_PER_M_ROWS = 0.05

    # Sort 的内存（单位：倍数）
    SORT_MEMORY_MULTIPLIER = 2.0

    @staticmethod
    def estimate_read_cost(
        num_rows: Optional[int],
        size_bytes: Optional[int],
    ) -> OperatorCost:
        """
        估算 Read 算子的代价

        CPU 成本来自：
        - 文件读取解析
        - 反序列化
        - Schema 推断

        Object Store 成本是输出大小（数据需要保存在 Object Store）
        """
        if num_rows is None or num_rows == 0:
            return OperatorCost()

        cpu_cost = (num_rows / 1_000_000) * CostEstimator.READ_CPU_PER_M_ROWS
        object_store_bytes = size_bytes or 0

        return OperatorCost(
            cpu_cost=cpu_cost,
            io_read_bytes=size_bytes or 0,
            object_store_bytes=object_store_bytes,
        )

    @staticmethod
    def estimate_filter_cost(
        num_rows: Optional[int],
        selectivity: Optional[float],
        input_size_bytes: Optional[int],
    ) -> OperatorCost:
        """
        估算 Filter 算子的代价

        CPU 成本：谓词评估
        Object Store：输出大小（input × selectivity）
        """
        if num_rows is None or num_rows == 0:
            return OperatorCost()

        selectivity = selectivity or 0.5
        output_bytes = int((input_size_bytes or 0) * selectivity)

        cpu_cost = (num_rows / 1_000_000) * CostEstimator.FILTER_CPU_PER_M_ROWS

        return OperatorCost(
            cpu_cost=cpu_cost,
            object_store_bytes=output_bytes,
        )

    @staticmethod
    def estimate_map_cost(
        num_rows: Optional[int],
        amplification_ratio: Optional[float],
        input_size_bytes: Optional[int],
        has_gpu: bool = False,
    ) -> OperatorCost:
        """
        估算 Map 算子的代价

        CPU/GPU 成本：UDF 执行
        Object Store：输出大小（input × amplification）

        参数：
            has_gpu: 是否是 GPU 加速的 Map（如 TorchMap）
        """
        if num_rows is None or num_rows == 0:
            return OperatorCost()

        amplification_ratio = amplification_ratio or 1.0
        output_rows = int(num_rows * amplification_ratio)
        output_bytes = int((input_size_bytes or 0) * amplification_ratio)

        if has_gpu:
            gpu_cost = (output_rows / 1_000_000) * CostEstimator.GPU_MAP_GPU_PER_M_ROWS
            return OperatorCost(
                gpu_cost=gpu_cost,
                object_store_bytes=output_bytes,
            )
        else:
            cpu_cost = (output_rows / 1_000_000) * CostEstimator.MAP_CPU_PER_M_ROWS
            return OperatorCost(
                cpu_cost=cpu_cost,
                object_store_bytes=output_bytes,
            )

    @staticmethod
    def estimate_join_cost(
        left_rows: Optional[int],
        right_rows: Optional[int],
        left_size_bytes: Optional[int],
        right_size_bytes: Optional[int],
        output_size_bytes: Optional[int],
    ) -> OperatorCost:
        """
        估算 Join 算子的代价

        CPU 成本：哈希计算、探测
        内存峰值：
          - Shuffle 内存
          - Join 内存（2x PyArrow 缓冲）
          - 输出内存
          总计约 3x 的输入大小

        Shuffle 成本：跨节点传输（如果有 Shuffle）
        """
        if (left_rows is None or right_rows is None or
            left_rows == 0 or right_rows == 0):
            return OperatorCost()

        total_input_rows = left_rows + right_rows
        total_input_bytes = (left_size_bytes or 0) + (right_size_bytes or 0)
        output_size = output_size_bytes or 0

        cpu_cost = (total_input_rows / 1_000_000) * CostEstimator.JOIN_CPU_PER_M_ROWS

        # Join 内存峰值
        peak_memory = total_input_bytes * CostEstimator.JOIN_MEMORY_MULTIPLIER

        # Join 通常涉及 Shuffle（跨节点传输）
        shuffle_bytes = total_input_bytes

        return OperatorCost(
            cpu_cost=cpu_cost,
            peak_memory_bytes=peak_memory,
            object_store_bytes=output_size,
            shuffle_bytes=shuffle_bytes,
        )

    @staticmethod
    def estimate_sort_cost(
        num_rows: Optional[int],
        size_bytes: Optional[int],
    ) -> OperatorCost:
        """
        估算 Sort 算子的代价

        CPU 成本：排序算法（O(n log n)）
        内存峰值：需要额外的内存来存储中间结果
        Shuffle：全局排序需要跨节点传输
        """
        if num_rows is None or num_rows == 0:
            return OperatorCost()

        # CPU 成本：O(n log n)
        cpu_cost = (num_rows / 1_000_000) * math.log2(max(num_rows, 2)) * 0.00001

        peak_memory = (size_bytes or 0) * CostEstimator.SORT_MEMORY_MULTIPLIER
        shuffle_bytes = size_bytes or 0

        return OperatorCost(
            cpu_cost=cpu_cost,
            peak_memory_bytes=peak_memory,
            object_store_bytes=size_bytes or 0,
            shuffle_bytes=shuffle_bytes,
        )


@dataclass
class PipelineProperties:
    """
    管线的特征描述

    用于预留比例推导的特性分析。
    """

    has_gpu_ops: bool = False       # 是否有 GPU 算子
    has_alltoall_ops: bool = False  # 是否有 AllToAll（Sort/Shuffle）
    has_shuffle_ops: bool = False   # 是否有 Shuffle
    num_stages: int = 0             # 算子数量
    is_spot_environment: bool = False  # 是否在 Spot 实例上

    @property
    def is_long_pipeline(self) -> bool:
        """是否是长管线（> 5 算子）"""
        return self.num_stages > 5

    @property
    def is_short_pipeline(self) -> bool:
        """是否是短管线（≤ 2 算子）"""
        return self.num_stages <= 2


class ReservationRatioDeriver:
    """
    预留比例自动推导

    核心算法：
    1. 分析算子间的资源竞争（不均衡系数）
    2. 检测管线特征（长度、GPU、AllToAll）
    3. 推导最优的 R 值

    设计说明：
    - 不均衡系数越高，R 值越高（保证没有算子被饿死）
    - 长管线需要更高的 R（更多竞争）
    - GPU 算子需要更高的 R（计算密集）
    - AllToAll 需要更低的 R（会解除内存限制）
    """

    @staticmethod
    def derive(
        op_costs: Dict[str, OperatorCost],
        pipeline_props: PipelineProperties,
    ) -> float:
        """
        推导最优预留比例

        参数：
            op_costs: 每个算子的代价向量
            pipeline_props: 管线特征

        返回：
            推导的 R 值（0.1 ~ 0.9）

        算法步骤：
        1. 提取内存需求
        2. 计算不均衡系数
        3. 基础值调整
        4. 管线特征修正
        5. 边界约束
        """
        if not op_costs or pipeline_props.num_stages == 0:
            return 0.5

        # Step 1: 提取每个算子的内存需求
        memory_demands = {}
        for op_name, cost in op_costs.items():
            # 保守估计：取 peak_memory 和 object_store 均摊的较大值
            peak = cost.peak_memory_bytes
            object_store_share = cost.object_store_bytes / max(1, pipeline_props.num_stages)
            memory_demands[op_name] = max(peak, object_store_share)

        total_demand = sum(memory_demands.values())
        if total_demand == 0:
            return 0.5

        # Step 2: 计算不均衡系数
        avg_demand = total_demand / pipeline_props.num_stages
        max_demand = max(memory_demands.values())

        if avg_demand > 0:
            imbalance_ratio = max_demand / avg_demand
        else:
            imbalance_ratio = 1.0

        logger.debug(
            f"ReservationRatio derivation: "
            f"avg_demand={avg_demand / (1024**3):.2f}GB, "
            f"max_demand={max_demand / (1024**3):.2f}GB, "
            f"imbalance_ratio={imbalance_ratio:.2f}"
        )

        # Step 3: 基础值 + 不均衡调整
        # imbalance = 1.0（完全均衡）→ r_imbalance = 0.0
        # imbalance = 5.0（严重不均衡）→ r_imbalance = 0.4
        r_base = 0.3
        r_imbalance = min(0.5, (imbalance_ratio - 1.0) * 0.1)

        # Step 4: 管线特征修正
        r_pipeline = 0.0

        # 长管线加码：更多算子竞争
        if pipeline_props.is_long_pipeline:
            r_pipeline += min(0.2, (pipeline_props.num_stages - 5) * 0.02)

        # GPU 算子加码：计算密集，需要更多内存预留
        if pipeline_props.has_gpu_ops:
            r_pipeline += 0.1

        # AllToAll 减码：会解除内存限制，无需高 R
        if pipeline_props.has_alltoall_ops:
            r_pipeline -= 0.1

        # 单算子主导减码：其他算子预留无意义
        if max_demand > total_demand * 0.7:
            r_pipeline -= 0.15

        # Spot 实例减码：资源不可靠，不能太激进
        if pipeline_props.is_spot_environment:
            r_pipeline -= 0.1

        # Step 5: 边界约束和最终计算
        r_final = max(0.1, min(0.9, r_base + r_imbalance + r_pipeline))

        logger.info(
            f"Derived reservation_ratio={r_final:.2f} "
            f"(base={r_base:.2f}, imbalance={r_imbalance:.2f}, "
            f"pipeline={r_pipeline:.2f}, "
            f"stages={pipeline_props.num_stages}, "
            f"gpu={pipeline_props.has_gpu_ops}, "
            f"alltoall={pipeline_props.has_alltoall_ops})"
        )

        return round(r_final, 2)


if __name__ == '__main__':
    # 快速测试
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # 测试代价估算
    read_cost = CostEstimator.estimate_read_cost(
        num_rows=1000000,
        size_bytes=1024 * 1024 * 100,  # 100MB
    )
    print(f"Read Cost: {read_cost.to_log_string()}")

    filter_cost = CostEstimator.estimate_filter_cost(
        num_rows=1000000,
        selectivity=0.5,
        input_size_bytes=1024 * 1024 * 100,
    )
    print(f"Filter Cost: {filter_cost.to_log_string()}")

    # 测试预留比例推导
    op_costs = {
        'read': read_cost,
        'filter': filter_cost,
    }
    props = PipelineProperties(
        num_stages=2,
        has_gpu_ops=False,
        has_alltoall_ops=False,
    )
    r = ReservationRatioDeriver.derive(op_costs, props)
    print(f"Derived R={r}")
