"""
Ray Data CBO Phase 3: Join重排序规则

源码位置：python/ray/data/_internal/logical/rules/join_reorder_rule.py

功能：
1. JoinReorderRule 规则实现
2. INNER JOIN自动左右表交换优化
3. 小表作为build侧，减少内存占用
4. 仅处理INNER JOIN，其他类型不修改
"""

from typing import List, Type, Any, Optional
import logging

from ray.data._internal.logical.interfaces.optimizer import Rule
from ray.data._internal.logical.interfaces.plan import Plan

logger = logging.getLogger(__name__)


class JoinReorderRule(Rule):
    """
    CBO 规则：Join左右表自动排序

    优化原理：
    - 在 Hash Join 中，小表应该作为 build 侧（构建哈希表）
    - 大表作为 probe 侧（扫描）
    - 这样可以降低内存占用，提升 L1/L2 缓存命中率

    支持的Join类型：
    - INNER JOIN（完全支持）
    - 其他类型（LEFT/RIGHT/OUTER）：保持原样，不修改

    功能：
    - 在物理优化阶段执行
    - 对每个 Join 算子，比较左右表大小
    - 如果左表 > 右表，交换
    - 仅在用户未显式指定 join_side 时修改
    - 输出详细的日志便于调试

    注册位置：
    在 _PHYSICAL_RULESET 中注册，依赖 DeriveShufflePartitionsRule

    执行流程：
    1. 遍历物理计划中的所有Join算子
    2. 检查Join类型（仅处理 INNER）
    3. 获取左右表的统计信息
    4. 比较数据大小
    5. 如果左表 > 右表，交换
    """

    @classmethod
    def dependencies(cls) -> List[Type["Rule"]]:
        """
        该规则依赖 DeriveShufflePartitionsRule

        原因：分区数可能影响内存计算，
        应该先推导分区数。
        """
        from ray.data._internal.logical.rules.derive_shuffle_partitions_rule import (
            DeriveShufflePartitionsRule,
        )

        return [DeriveShufflePartitionsRule]

    def apply(self, plan: Plan) -> Plan:
        """
        应用规则到物理计划

        参数：
            plan: 物理执行计划

        返回：
            修改后的计划（Join可能已重排序）
        """
        try:
            context = getattr(plan, 'context', None)
            if context is not None:
                if not getattr(context, 'enable_cost_based_optimization', True):
                    logger.debug("CBO disabled, skipping join reorder")
                    return plan

            for op in plan.dag.post_order_iter():
                if self._is_join_operator(op):
                    self._try_reorder_join(op)

            return plan

        except Exception as e:
            logger.warning(f"Error reordering join: {e}, skipping")
            return plan

    def _is_join_operator(self, op: Any) -> bool:
        """检查操作符是否为Join操作"""
        return 'Join' in type(op).__name__

    def _try_reorder_join(self, op: Any) -> None:
        """
        尝试重排序Join的左右表

        参数：
            op: Join 算子
        """
        try:
            # 步骤1：检查Join类型
            join_type = self._get_join_type(op)

            if join_type != 'INNER':
                logger.debug(
                    f"Join {op.name} is {join_type}, not reordering "
                    f"(only INNER JOIN is reordered)"
                )
                return

            # 步骤2：检查用户是否显式指定了join_side
            if self._is_user_specified(op):
                logger.info(
                    f"Join {op.name} has user-specified join_side, not reordering"
                )
                return

            # 步骤3：获取左右表大小
            left_size = self._get_side_size(op, 'left')
            right_size = self._get_side_size(op, 'right')

            if left_size is None or right_size is None:
                logger.debug(
                    f"Cannot determine table sizes for Join {op.name}, skipping"
                )
                return

            # 步骤4：如果左表更大，交换
            if left_size > right_size:
                self._swap_join_sides(op)

                logger.info(
                    f"CBO reordered Join {op.name}: "
                    f"left ({left_size / (1024**3):.2f}GB) > "
                    f"right ({right_size / (1024**3):.2f}GB), swapped"
                )
            else:
                logger.debug(
                    f"Join {op.name} is already well-ordered: "
                    f"left ({left_size / (1024**3):.2f}GB) <= "
                    f"right ({right_size / (1024**3):.2f}GB)"
                )

        except Exception as e:
            logger.warning(f"Error reordering Join {op.name}: {e}")

    def _get_join_type(self, op: Any) -> str:
        """
        获取Join的类型

        返回值：
            'INNER', 'LEFT', 'RIGHT', 'OUTER', 'CROSS', 或 'UNKNOWN'
        """
        try:
            # 优先从 join_type 属性获取
            if hasattr(op, 'join_type'):
                return op.join_type.upper()

            # 尝试从 _join_type 属性获取
            if hasattr(op, '_join_type'):
                jtype = op._join_type
                if isinstance(jtype, str):
                    return jtype.upper()
                # 如果是枚举类型，尝试获取 name
                if hasattr(jtype, 'name'):
                    return jtype.name.upper()
                if hasattr(jtype, 'value'):
                    return str(jtype.value).upper()

        except Exception:
            pass

        return 'UNKNOWN'

    def _is_user_specified(self, op: Any) -> bool:
        """
        检查用户是否显式指定了join_side

        参数：
            op: Join 算子

        返回：
            True 如果用户指定了，False 否则
        """
        try:
            # 检查 _user_config
            if hasattr(op, '_user_config'):
                if op._user_config.get('join_side') is not None:
                    return True

            # 检查各种命名约定
            if hasattr(op, '_user_specified_side'):
                return op._user_specified_side

            if hasattr(op, '_user_join_side'):
                return True

        except Exception:
            pass

        return False

    def _get_side_size(self, op: Any, side: str) -> Optional[int]:
        """
        获取Join一侧的数据大小（字节）

        参数：
            op: Join 算子
            side: 'left' 或 'right'

        返回：
            数据大小（字节），若无法获取则返回 None
        """
        try:
            if side == 'left':
                side_op = self._get_left_input(op)
            elif side == 'right':
                side_op = self._get_right_input(op)
            else:
                return None

            if side_op is None:
                return None

            return self._get_op_output_size(side_op)

        except Exception as e:
            logger.debug(f"Error getting {side} side size: {e}")
            return None

    def _get_left_input(self, op: Any) -> Any:
        """获取Join的左输入操作符"""
        if hasattr(op, '_left'):
            return op._left
        if hasattr(op, '_inputs') and len(op._inputs) > 0:
            return op._inputs[0]
        if hasattr(op, 'left_input'):
            return op.left_input
        return None

    def _get_right_input(self, op: Any) -> Any:
        """获取Join的右输入操作符"""
        if hasattr(op, '_right'):
            return op._right
        if hasattr(op, '_inputs') and len(op._inputs) > 1:
            return op._inputs[1]
        if hasattr(op, 'right_input'):
            return op.right_input
        return None

    def _get_op_output_size(self, op: Any) -> Optional[int]:
        """获取算子输出的字节大小"""
        try:
            # 尝试从逻辑算子的统计信息获取
            if hasattr(op, '_logical_operators') and op._logical_operators:
                logical_op = op._logical_operators[0]
                if hasattr(logical_op, 'infer_statistics'):
                    stats = logical_op.infer_statistics()
                    if stats and hasattr(stats, 'size_bytes'):
                        return stats.size_bytes

            # 尝试从直接属性获取
            if hasattr(op, 'size_bytes'):
                return op.size_bytes

            if hasattr(op, '_size_bytes'):
                return op._size_bytes

        except Exception as e:
            logger.debug(f"Error getting output size: {e}")

        return None

    def _swap_join_sides(self, op: Any) -> None:
        """
        交换Join的左右输入

        参数：
            op: Join 算子
        """
        try:
            left = self._get_left_input(op)
            right = self._get_right_input(op)

            if left is None or right is None:
                return

            # 尝试直接交换 _left 和 _right
            if hasattr(op, '_left') and hasattr(op, '_right'):
                op._left, op._right = right, left
                return

            # 尝试交换 _inputs 列表
            if hasattr(op, '_inputs') and len(op._inputs) >= 2:
                op._inputs[0], op._inputs[1] = op._inputs[1], op._inputs[0]
                return

            # 尝试交换命名的属性
            if hasattr(op, 'left_input') and hasattr(op, 'right_input'):
                op.left_input, op.right_input = op.right_input, op.left_input
                return

            logger.warning(f"Could not find way to swap inputs for {op.name}")

        except Exception as e:
            logger.warning(f"Error swapping join sides: {e}")


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
    DeriveShufflePartitionsRule,
    JoinReorderRule,                  # ← 新增
])
"""
