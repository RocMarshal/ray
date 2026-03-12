"""
Ray Data CBO Phase 1: Parquet 列统计提取

源码位置：python/ray/data/_internal/datasource/parquet_stat_extractor.py

功能：
从 Parquet footer 零成本地提取列级统计信息
"""

from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


def extract_column_stats_from_parquet_metadata(
    parquet_metadata: Any,
    schema: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    从 Parquet FileMetaData 中提取列级统计信息

    参数：
        parquet_metadata: PyArrow 的 parquet.FileMetaData 对象
        schema: PyArrow Schema 对象

    返回：
        Dict[列名 -> 统计信息字典]

    说明：
        这个函数实现了零成本的列统计提取：
        - Parquet footer 已经被读取（在 _fetch_parquet_file_info 中）
        - 我们只是遍历 footer 并提取其中的统计信息
        - 无额外的数据读取或网络 IO

        Parquet 中每列的统计包括：
        - min/max: 列的最小和最大值
        - null_count: NULL 值的个数
        - distinct_count: 去重后的值个数（如果 writer 支持）
        - 编码和压缩方式
    """
    column_stats = {}

    try:
        num_columns = parquet_metadata.num_columns
        num_row_groups = parquet_metadata.num_row_groups

        for col_idx in range(num_columns):
            try:
                # 获取列名
                col_name = schema.field(col_idx).name

                # 跨所有 RowGroup 聚合统计信息
                total_null_count = 0
                global_min = None
                global_max = None
                global_distinct_count = None
                encoding = None
                compression = None

                for rg_idx in range(num_row_groups):
                    try:
                        col_metadata = parquet_metadata.row_group(rg_idx).column(col_idx)

                        # 获取编码和压缩方式（只需获取一次）
                        if encoding is None and col_metadata.encoding:
                            encoding = str(col_metadata.encoding)
                        if compression is None and col_metadata.compression:
                            compression = str(col_metadata.compression)

                        # 聚合 NULL 计数
                        if col_metadata.is_stats_set:
                            stats = col_metadata.statistics
                            if stats and stats.null_count is not None:
                                total_null_count += stats.null_count

                            # 聚合 min/max（跨所有 row groups）
                            if stats and stats.has_min_max:
                                try:
                                    col_min = stats.min
                                    col_max = stats.max

                                    if global_min is None or col_min < global_min:
                                        global_min = col_min
                                    if global_max is None or col_max > global_max:
                                        global_max = col_max
                                except (TypeError, AttributeError):
                                    pass

                            # 获取 distinct count（通常只有一个有效）
                            if stats and stats.distinct_count is not None:
                                global_distinct_count = stats.distinct_count

                    except Exception as e:
                        logger.debug(f"Failed to extract stats from row group {rg_idx}, col {col_idx}: {e}")
                        continue

                # 保存该列的统计
                column_stats[col_name] = {
                    'min': global_min,
                    'max': global_max,
                    'null_count': total_null_count,
                    'distinct_count': global_distinct_count,
                    'encoding': encoding,
                    'compression': compression,
                }

            except Exception as e:
                logger.debug(f"Failed to extract stats for column {col_idx}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to extract column statistics from Parquet metadata: {e}")

    return column_stats


def estimate_column_average_size(
    column_name: str,
    column_type: str,
    parquet_metadata: Any,
    schema: Any,
) -> Optional[float]:
    """
    估算列的平均行大小（字节）

    参数：
        column_name: 列名
        column_type: 列的数据类型
        parquet_metadata: Parquet FileMetaData
        schema: PyArrow Schema

    返回：
        平均行大小（字节），或 None 表示无法估算

    说明：
        这个函数基于 Parquet footer 中的大小信息估算平均行大小。
        对于固定长度的类型（int, float）可以精确计算。
        对于变长类型（string）需要采用启发式方法。
    """
    try:
        num_row_groups = parquet_metadata.num_row_groups
        total_size = 0
        num_rows = 0

        for rg_idx in range(num_row_groups):
            rg = parquet_metadata.row_group(rg_idx)

            # 查找该列在 row group 中的位置
            col_idx = None
            for i in range(schema.num_fields):
                if schema.field(i).name == column_name:
                    col_idx = i
                    break

            if col_idx is None:
                continue

            col_metadata = rg.column(col_idx)

            # 获取该列在该 row group 中的压缩大小
            if col_metadata.total_compressed_size > 0:
                total_size += col_metadata.total_compressed_size

            # 获取行数
            num_rows = max(num_rows, rg.num_rows)

        if num_rows > 0 and total_size > 0:
            # 注意：这是压缩后的大小，实际内存大小可能更大
            # 估算比率：通常压缩率在 2-5 倍之间
            avg_compressed = total_size / num_rows
            # 保守估算：假设解压缩后是压缩大小的 3 倍
            avg_uncompressed = avg_compressed * 3
            return avg_uncompressed

    except Exception as e:
        logger.debug(f"Failed to estimate average size for column {column_name}: {e}")

    return None


def extract_total_bytes_from_parquet(parquet_metadata: Any) -> Optional[int]:
    """
    从 Parquet metadata 提取总数据大小

    参数：
        parquet_metadata: PyArrow parquet.FileMetaData

    返回：
        总字节数（解压缩后的估算），或 None
    """
    try:
        total_bytes = 0
        for rg_idx in range(parquet_metadata.num_row_groups):
            rg = parquet_metadata.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                # 使用未压缩的大小（如果可用）
                if col.total_uncompressed_size > 0:
                    total_bytes += col.total_uncompressed_size
                else:
                    # 回退到压缩大小（会低估）
                    total_bytes += col.total_compressed_size

        if total_bytes > 0:
            return total_bytes
    except Exception as e:
        logger.debug(f"Failed to extract total bytes from Parquet metadata: {e}")

    return None


def extract_row_count_from_parquet(parquet_metadata: Any) -> Optional[int]:
    """
    从 Parquet metadata 提取行数

    这个信息通常总是可用的（Parquet 必须记录行数）。
    """
    try:
        return parquet_metadata.num_rows
    except Exception as e:
        logger.debug(f"Failed to extract row count from Parquet metadata: {e}")
        return None


# ============================================================================
# 集成点：如何在 Read 算子中使用这些函数
# ============================================================================

"""
集成到 ReadOperator 中的方式：

class ParquetReadOperator(ReadOperator):
    def infer_statistics(self):
        from ray.data._internal.cbo_stats.operator_statistics import (
            OperatorStatistics,
            ColumnStatistics,
            ConfidenceLevel,
        )

        # 获取已有的 Parquet 文件信息
        file_infos = self._file_infos  # 来自 _fetch_parquet_file_info
        if not file_infos:
            return None

        total_rows = 0
        total_size = 0
        all_column_stats = {}

        # 遍历所有 Parquet 文件
        for file_info in file_infos:
            # 提取列统计
            col_stats_dict = extract_column_stats_from_parquet_metadata(
                file_info.metadata,
                file_info.schema,
            )

            # 转换为 ColumnStatistics 对象
            for col_name, col_stat_dict in col_stats_dict.items():
                if col_name not in all_column_stats:
                    all_column_stats[col_name] = ColumnStatistics(
                        name=col_name,
                        min_value=col_stat_dict.get('min'),
                        max_value=col_stat_dict.get('max'),
                        null_count=col_stat_dict.get('null_count'),
                        distinct_count=col_stat_dict.get('distinct_count'),
                        encoding=col_stat_dict.get('encoding'),
                        compression=col_stat_dict.get('compression'),
                        confidence=ConfidenceLevel.HIGH.value,
                    )

            # 累加行数和大小
            total_rows += extract_row_count_from_parquet(file_info.metadata) or 0
            total_bytes = extract_total_bytes_from_parquet(file_info.metadata)
            if total_bytes:
                total_size += total_bytes

        # 计算平均行大小
        avg_row_bytes = None
        if total_rows > 0 and total_size > 0:
            avg_row_bytes = total_size / total_rows

        return OperatorStatistics(
            num_rows=total_rows,
            size_bytes=total_size,
            avg_row_bytes=avg_row_bytes,
            column_stats=all_column_stats,
            confidence=ConfidenceLevel.HIGH.value,
        )
"""
