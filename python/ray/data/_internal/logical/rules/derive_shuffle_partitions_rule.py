"""
Ray Data CBO Phase 3: Shuffle分区数推导规则

源码位置：python/ray/data/_internal/logical/rules/derive_shuffle_partitions_rule.py

功能：
1. DeriveShufflePartitionsRule 规则实现
2. 自动推导Join/Sort/Shuffle的分区数
3. 基于数据量自动计算最优分区数
4. 支持用户显式设置的参数不被覆盖
"""

from typing import List, Type, Any, Optional
import logging
import math

from ray.data._internal.logical.interfaces.optimizer import Rule
from ray.data._internal.logical.interfaces.plan import Plan

logger = logging.getLogger(__name__)


class DeriveShufflePartitionsRule(Rule):
    """
    CBO 规则：自动推导Shuffle分区数

    目标分区大小：512MB（可配置）
    分区数范围：[10, 10000]

    功能：
    - 在物理优化阶段执行
    - 分析Join/Sort/Shuffle算子的输入数据量
    - 自动计算最优的num_partitions
    - 仅在用户未显式设置时覆盖
    - 输出详细的日志便于调试

    注册位置：
    在 _PHYSICAL_RULESET 中注册，依赖 DeriveReservationRatioRule

    执行流程：
    1. 遍历物理计划中的所有算子
    2. 识别 Shuffle 类算子（Join/Sort/Repartition/Shuffle）
    3. 获取输入数据大小
    4. 计算最优的分区数
    5. 如果用户未设置，应用推导的值
    """

    # 配置参数
    TARGET_PARTITION_SIZE_BYTES = 512 * 1024 * 1024  # 512MB
    MIN_PARTITIONS = 10
    MAX_PARTITIONS = 10000

    @classmethod
    def dependencies(cls) -> List[Type["Rule"]]:
        """
        该规则依赖 DeriveReservationRatioRule

        原因：预留比例推导会影响资源可用性，
        从而影响分区数的选择。
        """
        from ray.data._internal.logical.rules.derive_reservation_ratio_rule import (
            DeriveReservationRatioRule,
        )

        return [DeriveReservationRatioRule]

    def apply(self, plan: Plan) -> Plan:
        """
        应用规则到物理计划

        参数：
            plan: 物理执行计划

        返回：
            修改后的计划（分区数可能已更新）
        """
        try:
            context = getattr(plan, 'context', None)
            if context is not None:
                if not getattr(context, 'enable_cost_based_optimization', True):
                    logger.debug("CBO disabled, skipping shuffle partition derivation")
                    return plan

            for op in plan.dag.post_order_iter():
                if self._is_shuffle_operator(op):
                    derived_partitions = self._derive_partitions(op, plan)

                    if derived_partitions is not None:
                        self._apply_derived_partitions(op, derived_partitions)

            return plan

        except Exception as e:
            logger.warning(f"Error deriving shuffle partitions: {e}, skipping")
            return plan

    def _is_shuffle_operator(self, op: Any) -> bool:
        """
        检查操作符是否为 Shuffle 类操作

        支持的操作符：
        - Join
        - Sort
        - Shuffle
        - Repartition
        """
        op_type = type(op).__name__

        shuffle_types = ['Join', 'Sort', 'Shuffle', 'Repartition']
        return any(t in op_type for t in shuffle_types)

    def _derive_partitions(self, op: Any, plan: Any) -> Optional[int]:
        """
        推导最优的分区数

        参数：
            op: Shuffle 算子
            plan: 执行计划

        返回：
            推导的分区数，若无法推导则返回 None
        """
        # 获取输入数据大小
        input_size_bytes = self._get_input_size_bytes(op)

        if input_size_bytes is None or input_size_bytes == 0:
            logger.debug(f"Cannot determine input size for {op.name}, skipping")
            return None

        # 计算分区数
        # num_partitions = ceil(total_bytes / target_partition_size)
        num_partitions = max(
            self.MIN_PARTITIONS,
            min(
                self.MAX_PARTITIONS,
                math.ceil(input_size_bytes / self.TARGET_PARTITION_SIZE_BYTES)
            )
        )

        logger.debug(
            f"Derived num_partitions for {op.name}: {num_partitions} "
            f"(input_size={input_size_bytes / (1024 ** 3):.2f}GB, "
            f"target_partition_size={self.TARGET_PARTITION_SIZE_BYTES / (1024 ** 2):.0f}MB)"
        )

        return num_partitions

    def _get_input_size_bytes(self, op: Any) -> Optional[int]:
        """
        获取算子的输入数据大小

        对于 Join：左表 + 右表
        对于其他：使用第一个输入的大小
        """
        try:
            op_type = type(op).__name__

            if 'Join' in op_type:
                # Join 算子有左右两个输入
                left_size = self._get_stats_size(op, 'left')
                right_size = self._get_stats_size(op, 'right')

                if left_size is not None and right_size is not None:
                    return left_size + right_size
                elif left_size is not None:
                    return left_size
                elif right_size is not None:
                    return right_size
            else:
                # 其他算子（Sort/Shuffle/Repartition）
                # 使用第一个输入的统计信息
                if hasattr(op, '_upstream') and op._upstream:
                    upstream_op = op._upstream[0]
                    size = self._get_op_output_size(upstream_op)
                    if size is not None:
                        return size

        except Exception as e:
            logger.debug(f"Error getting input size for {op.name}: {e}")

        return None

    def _get_stats_size(self, op: Any, side: str) -> Optional[int]:
        """获取Join左侧或右侧的统计大小"""
        try:
            if side == 'left' and hasattr(op, '_left'):
                return self._get_op_output_size(op._left)
            elif side == 'right' and hasattr(op, '_right'):
                return self._get_op_output_size(op._right)
        except Exception:
            pass
        return None

    def _get_op_output_size(self, op: Any) -> Optional[int]:
        """获取算子输出的字节大小"""
        try:
            if hasattr(op, '_logical_operators') and op._logical_operators:
                logical_op = op._logical_operators[0]
                if hasattr(logical_op, 'infer_statistics'):
                    stats = logical_op.infer_statistics()
                    if stats and hasattr(stats, 'size_bytes'):
                        return stats.size_bytes
        except Exception:
            pass
        return None

    def _apply_derived_partitions(self, op: Any, num_partitions: int) -> None:
        """
        应用推导的分区数到算子

        参数：
            op: Shuffle 算子
            num_partitions: 推导的分区数
        """
        try:
            # 检查用户是否显式设置了 num_partitions
            user_set = False

            if hasattr(op, '_user_config'):
                user_set = op._user_config.get('num_partitions') is not None

            if user_set:
                logger.info(
                    f"User explicitly set num_partitions={op.num_partitions} "
                    f"for {op.name}, not overriding with CBO derived value {num_partitions}"
                )
                return

            # 应用推导的值
            if hasattr(op, 'num_partitions'):
                old_partitions = op.num_partitions
                op.num_partitions = num_partitions

                logger.info(
                    f"CBO derived num_partitions for {op.name}: "
                    f"{old_partitions} → {num_partitions}"
                )
            elif hasattr(op, '_num_partitions'):
                old_partitions = op._num_partitions
                op._num_partitions = num_partitions

                logger.info(
                    f"CBO derived num_partitions for {op.name}: "
                    f"{old_partitions} → {num_partitions}"
                )

        except Exception as e:
            logger.warning(f"Error applying derived partitions to {op.name}: {e}")

    @staticmethod
    def calculate_num_partitions(
        total_bytes: Optional[int],
        target_partition_size_bytes: Optional[int] = None,
        min_partitions: Optional[int] = None,
        max_partitions: Optional[int] = None,
    ) -> int:
        """
        静态方法：计算最优分区数

        参数：
            total_bytes: 总数据大小（字节）
            target_partition_size_bytes: 目标分区大小（默认512MB）
            min_partitions: 最小分区数（默认10）
            max_partitions: 最大分区数（默认10000）

        返回：
            推导的分区数

        示例：
            >>> calculate_num_partitions(10 * 1024**3)  # 10GB
            20
        """
        if total_bytes is None or total_bytes == 0:
            return min_partitions or DeriveShufflePartitionsRule.MIN_PARTITIONS

        target_size = target_partition_size_bytes or \
            DeriveShufflePartitionsRule.TARGET_PARTITION_SIZE_BYTES
        min_p = min_partitions or DeriveShufflePartitionsRule.MIN_PARTITIONS
        max_p = max_partitions or DeriveShufflePartitionsRule.MAX_PARTITIONS

        num_partitions = max(
            min_p,
            min(
                max_p,
                math.ceil(total_bytes / target_size)
            )
        )

        return num_partitions


# === 规则注册点 ===
"""
在 python/ray/data/_internal/logical/optimizers.py 中，
将规则注册到 _PHYSICAL_RULESET：

_PHYSICAL_RULESET = Ruleset([
    InheritTargetMaxBlockSizeRule,
    SetReadParallelismRule,
    FuseOperators,
    ConfigureMapTaskMemoryUsingOutputSize,
    DeriveReservationRatioRule,
    DeriveShufflePartitionsRule,      # ← 新增
])
"""
