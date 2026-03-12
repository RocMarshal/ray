"""
Ray Data CBO Phase 2: 物理优化规则

源码位置：python/ray/data/_internal/logical/rules/derive_reservation_ratio_rule.py

功能：
1. DeriveReservationRatioRule 规则实现
2. 自动推导预留比例并应用
3. 日志输出和可观测性支持
"""

from typing import List, Type, Dict, Any
import logging

from ray.data._internal.logical.interfaces.optimizer import Rule
from ray.data._internal.logical.interfaces.plan import Plan

logger = logging.getLogger(__name__)


class DeriveReservationRatioRule(Rule):
    """
    CBO 规则：自动推导预留比例

    功能：
    - 在物理优化阶段执行
    - 基于统计信息推导最优的 reservation_ratio
    - 只在用户未显式设置时覆盖
    - 输出详细的日志便于调试

    注册位置：
    在 _PHYSICAL_RULESET 中注册，依赖 FuseOperators

    执行流程：
    1. 遍历物理计划中的所有算子
    2. 收集每个算子的统计信息
    3. 估算每个算子的代价
    4. 提取管线特征
    5. 推导最优的 R 值
    6. 如果用户未设置，应用推导的 R 值
    """

    @classmethod
    def dependencies(cls) -> List[Type["Rule"]]:
        """
        该规则依赖 FuseOperators

        原因：算子融合会改变算子数量和资源需求，
        必须融合完成后再推导预留比例。
        """
        from ray.data._internal.logical.rules.operator_fusion import FuseOperators

        return [FuseOperators]

    def apply(self, plan: Plan) -> Plan:
        """
        应用规则到物理计划

        参数：
            plan: 物理执行计划

        返回：
            修改后的计划（预留比例可能已更新）
        """
        try:
            # 1. 收集统计信息
            statistics_cache = self._collect_statistics(plan)
            if not statistics_cache:
                logger.debug("No statistics available, skipping R derivation")
                return plan

            # 2. 检查 CBO 是否启用
            context = getattr(plan, 'context', None)
            if context is not None:
                if not getattr(context, 'enable_cost_based_optimization', True):
                    logger.debug("CBO disabled, skipping R derivation")
                    return plan

            # 3. 估算代价
            op_costs = self._estimate_costs(plan, statistics_cache)

            # 4. 提取管线特征
            pipeline_props = self._extract_pipeline_properties(plan)

            # 5. 推导预留比例
            from ray.data._internal.cbo_stats.cost_model import ReservationRatioDeriver
            derived_ratio = ReservationRatioDeriver.derive(op_costs, pipeline_props)

            # 6. 应用推导结果
            return self._apply_derived_ratio(plan, derived_ratio)

        except Exception as e:
            logger.warning(f"Error deriving reservation ratio: {e}, skipping")
            return plan

    def _collect_statistics(self, plan: Any) -> Dict[str, Any]:
        """
        收集物理计划中所有算子的统计信息

        返回：
            {op_id -> OperatorStatistics}
        """
        statistics_cache = {}

        try:
            # 遍历 DAG 中的所有操作符
            for op in plan.dag.post_order_iter():
                if hasattr(op, '_logical_operators') and op._logical_operators:
                    # 获取逻辑算子的统计信息
                    logical_op = op._logical_operators[0]
                    if hasattr(logical_op, 'infer_statistics'):
                        stats = logical_op.infer_statistics()
                        if stats:
                            statistics_cache[op.name] = stats

        except Exception as e:
            logger.debug(f"Error collecting statistics: {e}")

        return statistics_cache

    def _estimate_costs(
        self,
        plan: Any,
        statistics_cache: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        估算每个算子的代价

        参数：
            plan: 物理执行计划
            statistics_cache: 收集的统计信息

        返回：
            {op_name -> OperatorCost}
        """
        from ray.data._internal.cbo_stats.cost_model import (
            CostEstimator,
            OperatorCost,
        )

        op_costs = {}

        try:
            for op in plan.dag.post_order_iter():
                op_name = op.name
                stats = statistics_cache.get(op_name)

                if not stats or not stats.is_available():
                    # 统计不可用，用零代价
                    op_costs[op_name] = OperatorCost()
                    continue

                # 根据算子类型估算代价
                op_cost = self._estimate_operator_cost(op, stats, CostEstimator)
                op_costs[op_name] = op_cost

                logger.debug(
                    f"Estimated cost for {op_name}: {op_cost.to_log_string()}"
                )

        except Exception as e:
            logger.debug(f"Error estimating costs: {e}")

        return op_costs

    def _estimate_operator_cost(
        self,
        op: Any,
        stats: Any,
        cost_estimator,
    ) -> Any:
        """
        估算单个算子的代价

        参数：
            op: 物理操作符
            stats: 操作符的统计信息
            cost_estimator: 代价估算引擎

        返回：
            OperatorCost 对象
        """
        from ray.data._internal.cbo_stats.cost_model import OperatorCost

        op_type = type(op).__name__

        # 根据操作符类型选择相应的代价估算方法
        if 'Read' in op_type:
            return cost_estimator.estimate_read_cost(
                stats.num_rows,
                stats.size_bytes,
            )
        elif 'Filter' in op_type:
            return cost_estimator.estimate_filter_cost(
                stats.num_rows,
                stats.selectivity,
                stats.size_bytes,
            )
        elif 'Map' in op_type:
            has_gpu = hasattr(op, '_is_gpu_op') and op._is_gpu_op
            return cost_estimator.estimate_map_cost(
                stats.num_rows,
                stats.amplification_ratio,
                stats.size_bytes,
                has_gpu,
            )
        elif 'Join' in op_type:
            # 简化实现：假设左右等大
            return cost_estimator.estimate_join_cost(
                stats.num_rows,
                stats.num_rows,
                stats.size_bytes,
                stats.size_bytes,
                stats.size_bytes,
            )
        elif 'Sort' in op_type:
            return cost_estimator.estimate_sort_cost(
                stats.num_rows,
                stats.size_bytes,
            )
        else:
            # 未知操作符，用默认代价
            return OperatorCost()

    def _extract_pipeline_properties(self, plan: Any) -> Any:
        """
        提取管线的特征

        分析：
        - 有多少个算子
        - 是否有 GPU 算子
        - 是否有 AllToAll（Sort/Shuffle）
        """
        from ray.data._internal.cbo_stats.cost_model import PipelineProperties

        num_stages = 0
        has_gpu_ops = False
        has_alltoall_ops = False

        try:
            for op in plan.dag.post_order_iter():
                num_stages += 1

                op_type = type(op).__name__

                if 'GPU' in op_type or (hasattr(op, '_is_gpu_op') and op._is_gpu_op):
                    has_gpu_ops = True

                if any(x in op_type for x in ['Sort', 'Shuffle', 'Repartition']):
                    has_alltoall_ops = True

        except Exception as e:
            logger.debug(f"Error extracting pipeline properties: {e}")

        props = PipelineProperties(
            num_stages=num_stages,
            has_gpu_ops=has_gpu_ops,
            has_alltoall_ops=has_alltoall_ops,
        )

        logger.debug(
            f"Pipeline properties: stages={num_stages}, "
            f"gpu={has_gpu_ops}, alltoall={has_alltoall_ops}"
        )

        return props

    def _apply_derived_ratio(self, plan: Any, derived_ratio: float) -> Any:
        """
        应用推导的预留比例到计划

        参数：
            plan: 物理执行计划
            derived_ratio: 推导的 R 值

        返回：
            修改后的计划
        """
        try:
            # 检查用户是否显式设置了 reservation_ratio
            context = plan.context
            user_set = getattr(context, '_user_set_reservation_ratio', False)

            if user_set:
                logger.info(
                    f"User explicitly set reservation_ratio="
                    f"{context.op_resource_reservation_ratio:.2f}, "
                    f"not overriding with CBO derived value {derived_ratio:.2f}"
                )
                return plan

            # 应用推导的值
            context.op_resource_reservation_ratio = derived_ratio

            logger.info(
                f"CBO derived reservation_ratio={derived_ratio:.2f}"
            )

        except Exception as e:
            logger.warning(f"Error applying derived ratio: {e}")

        return plan


# === 规则注册点 ===
"""
在 python/ray/data/_internal/logical/optimizers.py 中，
将规则注册到 _PHYSICAL_RULESET：

_PHYSICAL_RULESET = Ruleset([
    InheritTargetMaxBlockSizeRule,
    SetReadParallelismRule,
    FuseOperators,
    ConfigureMapTaskMemoryUsingOutputSize,
    DeriveReservationRatioRule,      # ← 新增
])
"""
