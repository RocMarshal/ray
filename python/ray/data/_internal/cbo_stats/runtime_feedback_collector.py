"""
Ray Data CBO Phase 4: 运行时反馈循环系统

源码位置：python/ray/data/_internal/stats/runtime_feedback_collector.py

功能：
1. RuntimeFeedbackCollector - 收集运行时统计信息
2. 反馈缓存存储和管理
3. LRU 淘汰策略
4. 统计信息更新逻辑（指数移动平均）
"""

import json
import logging
import os
import hashlib
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class OpRuntimeMetrics:
    """算子运行时指标"""
    op_name: str
    actual_rows: Optional[int] = None
    actual_bytes: Optional[int] = None
    actual_avg_bytes_per_row: Optional[float] = None
    actual_wall_time_seconds: Optional[float] = None
    execution_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpRuntimeMetrics":
        """从字典创建"""
        return cls(**data)


@dataclass
class FeedbackCacheEntry:
    """缓存条目"""
    plan_hash: str
    op_metrics: Dict[str, OpRuntimeMetrics]
    created_timestamp: float
    last_updated_timestamp: float
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'plan_hash': self.plan_hash,
            'op_metrics': {
                name: metrics.to_dict()
                for name, metrics in self.op_metrics.items()
            },
            'created_timestamp': self.created_timestamp,
            'last_updated_timestamp': self.last_updated_timestamp,
            'access_count': self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackCacheEntry":
        """从字典创建"""
        op_metrics = {
            name: OpRuntimeMetrics.from_dict(metrics)
            for name, metrics in data['op_metrics'].items()
        }
        return cls(
            plan_hash=data['plan_hash'],
            op_metrics=op_metrics,
            created_timestamp=data['created_timestamp'],
            last_updated_timestamp=data['last_updated_timestamp'],
            access_count=data.get('access_count', 0),
        )


class RuntimeFeedbackCollector:
    """运行时反馈收集和管理系统"""

    # 配置参数
    CACHE_DIR = Path.home() / '.ray' / 'data' / 'stats_cache'
    MAX_CACHE_SIZE = 1000  # 最多缓存 1000 个计划
    CACHE_RETENTION_DAYS = 30  # 缓存保留 30 天
    EMA_ALPHA = 0.7  # 指数移动平均的新值权重

    def __init__(self):
        """初始化反馈收集器"""
        self._ensure_cache_dir()
        self._memory_cache = OrderedDict()  # 内存缓存用于快速访问

    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在"""
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create cache directory: {e}")

    def collect_runtime_metrics(
        self,
        physical_plan: Any,
    ) -> Optional[Dict[str, OpRuntimeMetrics]]:
        """
        收集物理执行计划的运行时指标

        参数：
            physical_plan: 执行完成的物理计划

        返回：
            {op_name -> OpRuntimeMetrics} 字典，若无法收集则返回 None
        """
        try:
            metrics = {}

            for op in physical_plan.dag.post_order_iter():
                if not hasattr(op, 'metrics'):
                    continue

                op_metrics = op.metrics

                # 尝试提取关键指标
                actual_rows = self._get_metric_value(
                    op_metrics, 'rows_task_outputs_generated'
                )
                actual_bytes = self._get_metric_value(
                    op_metrics, 'bytes_task_outputs_generated'
                )
                wall_time = self._get_metric_value(
                    op_metrics, 'task_completion_time_total_s'
                )

                # 计算平均行大小
                avg_bytes_per_row = None
                if actual_rows and actual_bytes and actual_rows > 0:
                    avg_bytes_per_row = actual_bytes / actual_rows

                metrics[op.name] = OpRuntimeMetrics(
                    op_name=op.name,
                    actual_rows=actual_rows,
                    actual_bytes=actual_bytes,
                    actual_avg_bytes_per_row=avg_bytes_per_row,
                    actual_wall_time_seconds=wall_time,
                    execution_timestamp=time.time(),
                )

                logger.debug(
                    f"Collected metrics for {op.name}: "
                    f"rows={actual_rows}, bytes={actual_bytes}"
                )

            return metrics if metrics else None

        except Exception as e:
            logger.debug(f"Error collecting runtime metrics: {e}")
            return None

    def _get_metric_value(self, metrics: Any, attr_name: str) -> Any:
        """安全地获取指标值"""
        try:
            if hasattr(metrics, attr_name):
                return getattr(metrics, attr_name)
        except Exception:
            pass
        return None

    def save_feedback(
        self,
        physical_plan: Any,
        op_metrics: Dict[str, OpRuntimeMetrics],
    ) -> None:
        """
        保存反馈到缓存

        参数：
            physical_plan: 执行完成的物理计划
            op_metrics: 运行时指标字典
        """
        try:
            plan_hash = self._compute_plan_hash(physical_plan)

            entry = FeedbackCacheEntry(
                plan_hash=plan_hash,
                op_metrics=op_metrics,
                created_timestamp=time.time(),
                last_updated_timestamp=time.time(),
                access_count=1,
            )

            # 保存到内存缓存
            self._memory_cache[plan_hash] = entry

            # 保存到磁盘
            self._save_to_disk(plan_hash, entry)

            # 执行 LRU 淘汰
            self._evict_lru()

            logger.info(f"Saved feedback for plan {plan_hash}")

        except Exception as e:
            logger.warning(f"Error saving feedback: {e}")

    def load_feedback(
        self,
        physical_plan: Any,
    ) -> Optional[Dict[str, OpRuntimeMetrics]]:
        """
        加载反馈

        参数：
            physical_plan: 物理执行计划

        返回：
            {op_name -> OpRuntimeMetrics} 字典，若无缓存则返回 None
        """
        try:
            plan_hash = self._compute_plan_hash(physical_plan)

            # 先查内存缓存
            if plan_hash in self._memory_cache:
                entry = self._memory_cache[plan_hash]
                entry.access_count += 1
                return entry.op_metrics

            # 再查磁盘缓存
            entry = self._load_from_disk(plan_hash)
            if entry:
                entry.access_count += 1
                self._memory_cache[plan_hash] = entry
                return entry.op_metrics

            return None

        except Exception as e:
            logger.debug(f"Error loading feedback: {e}")
            return None

    def apply_feedback_to_statistics(
        self,
        estimated_stats: Any,  # OperatorStatistics
        op_metrics: OpRuntimeMetrics,
    ) -> Any:
        """
        使用运行时反馈修正统计信息

        采用指数移动平均(EMA)融合估算值和实际值：
        new_value = alpha * actual + (1-alpha) * estimated

        参数：
            estimated_stats: 估算的统计信息
            op_metrics: 运行时指标

        返回：
            修正后的统计信息
        """
        if not op_metrics:
            return estimated_stats

        try:
            # 修正行数
            if (op_metrics.actual_rows is not None and
                    estimated_stats.num_rows is not None):
                alpha = self.EMA_ALPHA
                new_rows = int(
                    alpha * op_metrics.actual_rows +
                    (1 - alpha) * estimated_stats.num_rows
                )
                estimated_stats.num_rows = new_rows
                logger.debug(
                    f"Updated num_rows: "
                    f"{op_metrics.actual_rows} (actual) + "
                    f"{estimated_stats.num_rows} (estimated) "
                    f"→ {new_rows} (EMA)"
                )

            # 修正字节数
            if (op_metrics.actual_bytes is not None and
                    estimated_stats.size_bytes is not None):
                alpha = self.EMA_ALPHA
                new_bytes = int(
                    alpha * op_metrics.actual_bytes +
                    (1 - alpha) * estimated_stats.size_bytes
                )
                estimated_stats.size_bytes = new_bytes

            # 提升置信度
            if hasattr(estimated_stats, 'confidence'):
                estimated_stats.confidence = min(
                    estimated_stats.confidence + 0.2,
                    1.0
                )

        except Exception as e:
            logger.debug(f"Error applying feedback to statistics: {e}")

        return estimated_stats

    def _compute_plan_hash(self, physical_plan: Any) -> str:
        """
        计算执行计划的哈希值

        用于缓存键，基于计划的 DAG 结构（不包括参数值）

        参数：
            physical_plan: 物理执行计划

        返回：
            哈希字符串
        """
        try:
            # 使用计划的 DAG 字符串表示
            if hasattr(physical_plan, 'dag'):
                dag_str = physical_plan.dag.dag_str
                # 只取算子类型和连接关系，忽略参数
                hash_input = self._extract_dag_structure(dag_str)
                return hashlib.sha256(hash_input.encode()).hexdigest()
        except Exception:
            pass

        # 降级方案：使用计划对象的字符串表示
        try:
            hash_input = str(physical_plan)
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(id(physical_plan)).encode()).hexdigest()

    def _extract_dag_structure(self, dag_str: str) -> str:
        """提取 DAG 结构（忽略参数细节）"""
        # 简单实现：保留算子名称和连接，移除参数
        # 例如：Read(path=/tmp) → Read
        return re.sub(r'\([^)]*\)', '', dag_str)

    def _save_to_disk(
        self,
        plan_hash: str,
        entry: FeedbackCacheEntry,
    ) -> None:
        """保存缓存条目到磁盘"""
        try:
            cache_file = self.CACHE_DIR / f"{plan_hash}.json"
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Error saving cache to disk: {e}")

    def _load_from_disk(self, plan_hash: str) -> Optional[FeedbackCacheEntry]:
        """从磁盘加载缓存条目"""
        try:
            cache_file = self.CACHE_DIR / f"{plan_hash}.json"

            if not cache_file.exists():
                return None

            # 检查过期
            file_mtime = os.path.getmtime(cache_file)
            age_seconds = time.time() - file_mtime
            if age_seconds > self.CACHE_RETENTION_DAYS * 86400:
                logger.debug(f"Cache expired for {plan_hash}")
                cache_file.unlink()
                return None

            with open(cache_file, 'r') as f:
                data = json.load(f)
                return FeedbackCacheEntry.from_dict(data)

        except Exception as e:
            logger.debug(f"Error loading cache from disk: {e}")
            return None

    def _evict_lru(self) -> None:
        """执行 LRU 淘汰"""
        try:
            # 如果内存缓存超过大小限制
            while len(self._memory_cache) > self.MAX_CACHE_SIZE:
                # 移除最久未访问的条目
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
                logger.debug(f"Evicted cache entry {oldest_key}")

            # 清理过期的磁盘缓存
            self._cleanup_expired_disk_cache()

        except Exception as e:
            logger.debug(f"Error during LRU eviction: {e}")

    def _cleanup_expired_disk_cache(self) -> None:
        """清理过期的磁盘缓存"""
        try:
            if not self.CACHE_DIR.exists():
                return

            current_time = time.time()
            for cache_file in self.CACHE_DIR.glob('*.json'):
                try:
                    file_mtime = os.path.getmtime(cache_file)
                    age_seconds = current_time - file_mtime
                    if age_seconds > self.CACHE_RETENTION_DAYS * 86400:
                        cache_file.unlink()
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Error cleaning up disk cache: {e}")

    def clear_cache(self) -> None:
        """清空所有缓存（用于测试）"""
        try:
            self._memory_cache.clear()
            if self.CACHE_DIR.exists():
                for cache_file in self.CACHE_DIR.glob('*.json'):
                    cache_file.unlink()
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    @staticmethod
    def get_instance() -> "RuntimeFeedbackCollector":
        """获取全局单例"""
        global _feedback_collector
        if _feedback_collector is None:
            _feedback_collector = RuntimeFeedbackCollector()
        return _feedback_collector


# === 全局单例访问 ===
_feedback_collector: Optional[RuntimeFeedbackCollector] = None


def get_feedback_collector() -> RuntimeFeedbackCollector:
    """获取反馈收集器单例"""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = RuntimeFeedbackCollector.get_instance()
    return _feedback_collector
