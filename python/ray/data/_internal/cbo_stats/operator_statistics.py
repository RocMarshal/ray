"""
Ray Data CBO Phase 1: OperatorStatistics 核心实现

源码位置：python/ray/data/_internal/stats/operator_statistics.py

功能：
1. OperatorStatistics 数据结构
2. ColumnStatistics 数据结构
3. 统计信息的基础操作
4. 零成本的 Parquet 列统计提取
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """统计信息的置信度等级"""
    EXACT = 1.0        # 精确值（来自元数据）
    HIGH = 0.95        # 高置信（Parquet footer 统计）
    MEDIUM = 0.7       # 中置信（采样或聚合）
    LOW = 0.4          # 低置信（启发式估算）
    UNKNOWN = 0.0      # 未知（不可用）


@dataclass
class ColumnStatistics:
    """
    单列的统计信息

    设计说明：
    - 来自 Parquet footer 的信息：min_value, max_value, null_count, distinct_count
    - 这些信息是"零成本"的（已在 footer 中，无需额外 IO）
    - 缺失的信息会导致某些优化失效，但系统仍然可以工作
    """

    name: str

    # === 从 Parquet footer 直接获取（零成本）===
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    null_count: Optional[int] = None
    distinct_count: Optional[int] = None  # NDV (Number of Distinct Values)

    # === 计算得出的信息 ===
    avg_size_bytes: Optional[float] = None  # 平均每个值占用的字节数

    # === Parquet 编码信息 ===
    encoding: Optional[str] = None       # 如 "PLAIN", "RLE", "DICT"
    compression: Optional[str] = None    # 如 "SNAPPY", "GZIP"

    # === 统计信息质量指标 ===
    confidence: float = ConfidenceLevel.EXACT.value  # 置信度（0.0 ~ 1.0）

    def __post_init__(self):
        """验证数据完整性"""
        if self.null_count is not None and self.null_count < 0:
            logger.warning(
                f"Column {self.name}: null_count={self.null_count} 是负数，设置为 0"
            )
            self.null_count = 0

        if self.distinct_count is not None and self.distinct_count <= 0:
            logger.debug(
                f"Column {self.name}: distinct_count={self.distinct_count} 无效"
            )
            self.distinct_count = None

    def is_available(self) -> bool:
        """检查是否有有效的统计信息"""
        return (
            self.min_value is not None or
            self.max_value is not None or
            self.null_count is not None or
            self.distinct_count is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'name': self.name,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'null_count': self.null_count,
            'distinct_count': self.distinct_count,
            'avg_size_bytes': self.avg_size_bytes,
            'encoding': self.encoding,
            'compression': self.compression,
            'confidence': self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnStatistics':
        """从字典反序列化"""
        return cls(**data)


@dataclass
class OperatorStatistics:
    """
    算子级的统计信息，沿 DAG 传播

    设计说明：
    1. 所有字段都是 Optional，因为统计信息可能不完整或不可用
    2. 置信度反映了信息的可靠程度
    3. scale() 方法用于在谓词下推后更新统计信息
    4. 这个类被所有优化规则使用，是 CBO 的基础
    """

    # === 基础统计量 ===
    num_rows: Optional[int] = None          # 行数
    size_bytes: Optional[int] = None        # 数据大小（字节）
    num_blocks: Optional[int] = None        # 块数（用于 Shuffle）
    avg_row_bytes: Optional[float] = None   # 平均行大小

    # === 列级统计量 ===
    column_stats: Dict[str, ColumnStatistics] = field(default_factory=dict)

    # === 算子特定的统计量 ===
    selectivity: Optional[float] = None          # Filter 选择率 (0.0 ~ 1.0)
    amplification_ratio: Optional[float] = None  # Map 膨胀比 (output_rows / input_rows)
    skew_factor: Optional[float] = None          # Shuffle 倾斜度 (max / avg)

    # === 统计信息质量 ===
    confidence: float = ConfidenceLevel.EXACT.value  # 置信度 (0.0 ~ 1.0)

    def __post_init__(self):
        """验证数据的一致性"""
        # 检查 num_rows 和 size_bytes 的一致性
        if self.num_rows is not None and self.num_rows < 0:
            logger.warning(f"num_rows={self.num_rows} 是负数，设置为 0")
            self.num_rows = 0

        if self.size_bytes is not None and self.size_bytes < 0:
            logger.warning(f"size_bytes={self.size_bytes} 是负数，设置为 0")
            self.size_bytes = 0

        # 检查 avg_row_bytes 的合理性
        if (self.avg_row_bytes is not None and self.avg_row_bytes < 0):
            logger.warning(f"avg_row_bytes={self.avg_row_bytes} 是负数，设置为 None")
            self.avg_row_bytes = None

        # 如果同时有 num_rows 和 size_bytes，可以计算 avg_row_bytes
        if (self.num_rows and self.size_bytes and not self.avg_row_bytes):
            self.avg_row_bytes = self.size_bytes / self.num_rows

        # 检查 selectivity 范围
        if self.selectivity is not None:
            if not (0.0 <= self.selectivity <= 1.0):
                logger.warning(
                    f"selectivity={self.selectivity} 超出范围 [0, 1]，"
                    f"设置为 {max(0.0, min(1.0, self.selectivity))}"
                )
                self.selectivity = max(0.0, min(1.0, self.selectivity))

    def scale(self, ratio: float) -> 'OperatorStatistics':
        """
        按比例缩放统计信息（用于谓词下推后的传播）

        参数：
            ratio: 缩放比例，通常是 selectivity 或 amplification_ratio

        返回：
            新的 OperatorStatistics 对象

        说明：
            - num_rows 和 size_bytes 乘以 ratio
            - 置信度降低（传播会引入更多不确定性）
            - 列级统计保持不变（因为 Filter 不改变列的范围）
        """
        if ratio < 0.0 or ratio > 10.0:
            logger.warning(
                f"scale ratio={ratio} 超出合理范围 [0, 10]，"
                f"这可能表示数据膨胀或其他异常情况"
            )

        new_stats = OperatorStatistics(
            num_rows=(int(self.num_rows * ratio) if self.num_rows else None),
            size_bytes=(int(self.size_bytes * ratio) if self.size_bytes else None),
            num_blocks=self.num_blocks,  # Shuffle 后块数不变
            avg_row_bytes=self.avg_row_bytes,
            column_stats=self.column_stats,  # 列统计保持不变
            selectivity=ratio,
            amplification_ratio=self.amplification_ratio,
            skew_factor=self.skew_factor,
            confidence=max(0.0, self.confidence * 0.9),  # 传播降低置信度
        )
        return new_stats

    def is_available(self) -> bool:
        """检查是否有可用的统计信息"""
        return (
            self.num_rows is not None or
            self.size_bytes is not None or
            bool(self.column_stats)
        )

    def get_column_stat(self, column_name: str) -> Optional[ColumnStatistics]:
        """获取指定列的统计信息"""
        return self.column_stats.get(column_name)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化和日志）"""
        return {
            'num_rows': self.num_rows,
            'size_bytes': self.size_bytes,
            'size_gb': self.size_bytes / (1024**3) if self.size_bytes else None,
            'num_blocks': self.num_blocks,
            'avg_row_bytes': self.avg_row_bytes,
            'selectivity': self.selectivity,
            'amplification_ratio': self.amplification_ratio,
            'skew_factor': self.skew_factor,
            'confidence': round(self.confidence, 2),
            'num_columns': len(self.column_stats),
        }

    def to_log_string(self) -> str:
        """格式化为日志字符串"""
        parts = []

        if self.num_rows is not None:
            parts.append(f"rows={self.num_rows}")

        if self.size_bytes is not None:
            size_gb = self.size_bytes / (1024**3)
            if size_gb > 1:
                parts.append(f"size={size_gb:.2f}GB")
            else:
                size_mb = self.size_bytes / (1024**2)
                parts.append(f"size={size_mb:.2f}MB")

        if self.avg_row_bytes is not None:
            parts.append(f"avg_row={self.avg_row_bytes:.0f}B")

        if self.selectivity is not None:
            parts.append(f"selectivity={self.selectivity:.2%}")

        if self.amplification_ratio is not None:
            parts.append(f"amplification={self.amplification_ratio:.2f}x")

        if self.skew_factor is not None:
            parts.append(f"skew={self.skew_factor:.2f}")

        if self.confidence < 1.0:
            parts.append(f"confidence={self.confidence:.1%}")

        return "[" + ", ".join(parts) + "]"

    @classmethod
    def zero(cls) -> 'OperatorStatistics':
        """创建一个代表零数据的统计对象"""
        return cls(num_rows=0, size_bytes=0, confidence=ConfidenceLevel.EXACT.value)

    @classmethod
    def unknown(cls) -> 'OperatorStatistics':
        """创建一个代表未知统计的对象（所有字段为 None）"""
        return cls(confidence=ConfidenceLevel.UNKNOWN.value)

    def merge(self, other: 'OperatorStatistics') -> 'OperatorStatistics':
        """
        合并两个统计对象（用于 Union 操作）

        说明：
            - num_rows: 求和
            - size_bytes: 求和
            - column_stats: 两个都有时保持，只有一个有时使用有的那个
            - confidence: 取较小值（保守估计）
        """
        if other is None:
            return self

        # 合并行数
        merged_rows = None
        if self.num_rows is not None and other.num_rows is not None:
            merged_rows = self.num_rows + other.num_rows
        elif self.num_rows is not None:
            merged_rows = self.num_rows
        elif other.num_rows is not None:
            merged_rows = other.num_rows

        # 合并字节数
        merged_bytes = None
        if self.size_bytes is not None and other.size_bytes is not None:
            merged_bytes = self.size_bytes + other.size_bytes
        elif self.size_bytes is not None:
            merged_bytes = self.size_bytes
        elif other.size_bytes is not None:
            merged_bytes = other.size_bytes

        # 合并列统计
        merged_col_stats = {}
        merged_col_stats.update(self.column_stats)
        merged_col_stats.update(other.column_stats)

        # 合并置信度（保守估计）
        merged_confidence = min(self.confidence, other.confidence)

        return OperatorStatistics(
            num_rows=merged_rows,
            size_bytes=merged_bytes,
            num_blocks=(self.num_blocks or other.num_blocks),
            avg_row_bytes=(self.avg_row_bytes or other.avg_row_bytes),
            column_stats=merged_col_stats,
            confidence=merged_confidence,
        )


# === 工具函数 ===

def estimate_selectivity_from_column_stats(
    column_name: str,
    operator: str,  # 如 'EQ', 'GT', 'LT', 'IN'
    value: Any,
    column_stat: Optional[ColumnStatistics],
) -> Optional[float]:
    """
    基于列统计估算选择率

    参数：
        column_name: 列名
        operator: 操作符（'EQ', 'GT', 'LT', 'IN', 'BETWEEN'）
        value: 比较值
        column_stat: 列统计信息

    返回：
        选择率 (0.0 ~ 1.0)，或 None 表示无法估算

    设计说明：
        这是一个启发式的实现，采用经典数据库的估算方法：
        - 等值谓词 (=)：1 / NDV
        - 范围谓词 (>, <)：(max - val) / (max - min) 或 (val - min) / (max - min)
        - IN 谓词：len(values) / NDV
        - 无法估算的情况：返回保守值 0.5
    """
    try:
        if column_stat is None or not column_stat.is_available():
            # 无法估算，返回保守默认值
            return 0.5

        if operator == 'EQ':
            # 等值选择率 = 1 / NDV
            if column_stat.distinct_count and column_stat.distinct_count > 0:
                return 1.0 / column_stat.distinct_count
            else:
                return 0.01  # 保守默认值

        elif operator in ('GT', 'GTE'):
            # 范围选择率
            if (column_stat.min_value is not None and
                column_stat.max_value is not None):
                # 假设数据均匀分布
                try:
                    range_size = float(column_stat.max_value - column_stat.min_value)
                    if range_size == 0:
                        return 0.5  # 所有值相同，选择率 50%

                    value_val = float(value)
                    if operator == 'GT':
                        return (column_stat.max_value - value_val) / range_size
                    else:  # GTE
                        return (column_stat.max_value - value_val + 1) / range_size
                except (TypeError, ValueError):
                    return 0.33  # 无法比较，使用启发式值
            else:
                return 0.33

        elif operator in ('LT', 'LTE'):
            # 范围选择率
            if (column_stat.min_value is not None and
                column_stat.max_value is not None):
                try:
                    range_size = float(column_stat.max_value - column_stat.min_value)
                    if range_size == 0:
                        return 0.5

                    value_val = float(value)
                    if operator == 'LT':
                        return (value_val - column_stat.min_value) / range_size
                    else:  # LTE
                        return (value_val - column_stat.min_value + 1) / range_size
                except (TypeError, ValueError):
                    return 0.33
            else:
                return 0.33

        elif operator == 'IN':
            # IN 选择率 = len(values) / NDV
            if (column_stat.distinct_count and
                column_stat.distinct_count > 0):
                values_len = len(value) if isinstance(value, (list, tuple)) else 1
                return min(1.0, values_len / column_stat.distinct_count)
            else:
                return 0.1

        else:
            # 未知操作符，返回保守值
            logger.debug(f"Unknown operator '{operator}' for selectivity estimation")
            return 0.5

    except Exception as e:
        logger.debug(
            f"Failed to estimate selectivity for {column_name} {operator}: {e}"
        )
        return 0.5


if __name__ == '__main__':
    # 快速测试
    logging.basicConfig(level=logging.DEBUG)

    # 测试 ColumnStatistics
    col = ColumnStatistics(
        name='age',
        min_value=0,
        max_value=100,
        null_count=10,
        distinct_count=50,
        confidence=ConfidenceLevel.HIGH.value,
    )
    print(f"Column: {col.to_dict()}")

    # 测试 OperatorStatistics
    stats = OperatorStatistics(
        num_rows=1000,
        size_bytes=1024 * 1024,
        avg_row_bytes=1024,
        selectivity=0.5,
        confidence=ConfidenceLevel.HIGH.value,
        column_stats={'age': col},
    )
    print(f"Stats: {stats.to_log_string()}")

    # 测试 scale
    scaled = stats.scale(0.5)
    print(f"Scaled (0.5): {scaled.to_log_string()}")

    # 测试选择率估算
    sel = estimate_selectivity_from_column_stats('age', 'GT', 50, col)
    print(f"Selectivity (age > 50): {sel:.2%}")
