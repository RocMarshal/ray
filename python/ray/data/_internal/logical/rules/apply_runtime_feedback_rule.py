"""
Ray Data CBO Phase 4: 反馈应用规则

源码位置：python/ray/data/_internal/logical/rules/apply_runtime_feedback_rule.py

功能：
1. ApplyRuntimeFeedbackRule - 在物理优化阶段应用运行时反馈
2. 修正统计信息以改进后续优化
3. 日志记录反馈应用的效果
"""

from typing import List, Type, Any, Optional
import logging

from ray.data._internal.logical.interfaces.optimizer import Rule
from ray.data._internal.logical.interfaces.plan import Plan

logger = logging.getLogger(__name__)


class ApplyRuntimeFeedbackRule(Rule):
    """
    CBO 规则：应用运行时反馈修正统计信息

    功能：
    - 在物理优化阶段执行
    - 从缓存加载前次执行的反馈数据
    - 用指数移动平均融合估算和实际值
    - 修正当前执行计划的统计信息
    - 输出详细的日志便于审计

    注册位置：
    在 _PHYSICAL_RULESET 中注册，依赖 JoinReorderRule（最后一个优化规则）

    执行流程：
    1. 尝试加载前次执行的反馈
    2. 遍历物理计划中的所有算子
    3. 查找对应的运行时指标
    4. 使用 EMA 融合估算和实际值
    5. 更新统计信息的置信度
    6. 输出反馈应用的统计信息
    """

    @classmethod
    def dependencies(cls) -> List[Type["Rule"]]:
        """
        该规则依赖 JoinReorderRule

        原因：这是最后一个优化规则，应该在所有其他优化完成后
        才应用反馈，确保反馈基于最终的计划结构。
        """
        from ray.data._internal.logical.rules.join_reorder_rule import JoinReorderRule

        return [JoinReorderRule]

    def apply(self, plan: Plan) -> Plan:
        """
        应用规则到物理计划

        参数：
            plan: 物理执行计划

        返回：
            修改后的计划（统计信息可能已更新）
        """
        try:
            context = getattr(plan, 'context', None)
            if context is not None:
                if not getattr(context, 'enable_cost_based_optimization', True):
                    logger.debug("CBO disabled, skipping runtime feedback")
                    return plan

            # 步骤1：尝试加载反馈
            from ray.data._internal.cbo_stats.runtime_feedback_collector import (
                get_feedback_collector,
            )

            feedback_collector = get_feedback_collector()
            op_metrics = feedback_collector.load_feedback(plan)

            if not op_metrics:
                logger.debug("No runtime feedback available for this plan")
                return plan

            # 步骤2-5：遍历并修正统计信息
            num_corrected = 0
            for op in plan.dag.post_order_iter():
                if op.name not in op_metrics:
                    continue

                metrics = op_metrics[op.name]
                num_corrected += self._apply_feedback_to_operator(
                    op, metrics, feedback_collector
                )

            if num_corrected > 0:
                logger.info(
                    f"Applied runtime feedback to {num_corrected} operators "
                    f"(plan: {self._get_plan_hash(plan)})"
                )

            return plan

        except Exception as e:
            logger.warning(f"Error applying runtime feedback: {e}, skipping")
            return plan

    def _apply_feedback_to_operator(
        self,
        op: Any,
        op_metrics: Any,  # OpRuntimeMetrics
        feedback_collector: Any,
    ) -> int:
        """
        应用反馈到单个算子

        参数：
            op: 物理算子
            op_metrics: 该算子的运行时指标
            feedback_collector: 反馈收集器

        返回：
            修正的统计数量（0或1）
        """
        try:
            # 尝试获取算子的统计信息
            stats = self._get_operator_statistics(op)
            if not stats:
                return 0

            # 应用反馈
            original_rows = stats.num_rows
            original_bytes = stats.size_bytes

            feedback_collector.apply_feedback_to_statistics(stats, op_metrics)

            # 记录修正
            if original_rows != stats.num_rows or original_bytes != stats.size_bytes:
                logger.debug(
                    f"Corrected statistics for {op.name}: "
                    f"rows {original_rows} → {stats.num_rows}, "
                    f"bytes {original_bytes} → {stats.size_bytes}, "
                    f"confidence {stats.confidence:.2f}"
                )
                return 1

        except Exception as e:
            logger.debug(f"Error applying feedback to operator {op.name}: {e}")

        return 0

    def _get_operator_statistics(self, op: Any) -> Optional[Any]:
        """
        获取算子的统计信息

        参数：
            op: 物理算子

        返回：
            OperatorStatistics 对象，若无法获取则返回 None
        """
        try:
            # 尝试从逻辑算子获取
            if hasattr(op, '_logical_operators') and op._logical_operators:
                logical_op = op._logical_operators[0]
                if hasattr(logical_op, 'infer_statistics'):
                    stats = logical_op.infer_statistics()
                    if stats:
                        return stats

            # 尝试从直接属性获取
            if hasattr(op, 'statistics'):
                return op.statistics

            if hasattr(op, '_statistics'):
                return op._statistics

        except Exception:
            pass

        return None

    def _get_plan_hash(self, plan: Any) -> str:
        """获取计划的哈希值（用于日志）"""
        try:
            from ray.data._internal.cbo_stats.runtime_feedback_collector import (
                RuntimeFeedbackCollector,
            )
            collector = RuntimeFeedbackCollector()
            return collector._compute_plan_hash(plan)[:8]
        except Exception:
            return "unknown"


# === 规则注册点 ===
"""
在 python/ray/data/_internal/logical/optimizers.py 中，
将规则注册到 _PHYSICAL_RULESET：

_PHYSICAL_RULESET = Ruleset([
    InheritTargetMaxBlockSizeRule,
    SetReadParallelismRule,
    FuseOperators,
    ConfigureMapTaskMemoryUsingOutputSize,
    DeriveReservationRatioRule,      # Phase 2
    DeriveShufflePartitionsRule,      # Phase 3
    JoinReorderRule,                  # Phase 3
    ApplyRuntimeFeedbackRule,         # Phase 4 ← 新增
])
"""
