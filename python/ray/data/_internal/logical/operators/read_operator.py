import copy
import functools
import math
from typing import Any, Dict, Optional, Union

from ray.data._internal.compute import ComputeStrategy
from ray.data._internal.logical.interfaces import (
    LogicalOperatorSupportsPredicatePushdown,
    LogicalOperatorSupportsProjectionPushdown,
    SourceOperator,
)
from ray.data._internal.logical.operators.map_operator import AbstractMap
from ray.data.block import (
    BlockMetadata,
    BlockMetadataWithSchema,
)
from ray.data.context import DataContext
from ray.data.datasource.datasource import Datasource, Reader
from ray.data.expressions import Expr

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "Read",
]


class Read(
    AbstractMap,
    SourceOperator,
    LogicalOperatorSupportsProjectionPushdown,
    LogicalOperatorSupportsPredicatePushdown,
):
    """Logical operator for read."""

    # TODO: make this a frozen dataclass. https://github.com/ray-project/ray/issues/55747
    def __init__(
        self,
        datasource: Datasource,
        datasource_or_legacy_reader: Union[Datasource, Reader],
        parallelism: int,
        num_outputs: Optional[int] = None,
        ray_remote_args: Optional[Dict[str, Any]] = None,
        compute: Optional[ComputeStrategy] = None,
    ):
        super().__init__(
            name=f"Read{datasource.get_name()}",
            input_op=None,
            can_modify_num_rows=True,
            num_outputs=num_outputs,
            ray_remote_args=ray_remote_args,
            compute=compute,
        )
        self.datasource = datasource
        self.datasource_or_legacy_reader = datasource_or_legacy_reader
        self.parallelism = parallelism
        self.detected_parallelism = None

    def output_data(self):
        return None

    def set_detected_parallelism(self, parallelism: int):
        """
        Set the true parallelism that should be used during execution. This
        should be specified by the user or detected by the optimizer.
        """
        self.detected_parallelism = parallelism

    def get_detected_parallelism(self) -> int:
        """
        Get the true parallelism that should be used during execution.
        """
        return self.detected_parallelism

    def estimated_num_outputs(self) -> Optional[int]:
        return self.num_outputs or self._estimate_num_outputs()

    def infer_metadata(self) -> BlockMetadata:
        """A ``BlockMetadata`` that represents the aggregate metadata of the outputs.

        This method gets metadata from the read tasks. It doesn't trigger any actual
        execution.
        """
        return self._cached_output_metadata.metadata

    def infer_statistics(self):
        """Infer output statistics from datasource metadata.

        For Parquet sources this also extracts column-level min/max/null_count
        from the footer at zero additional I/O cost.
        """
        from ray.data._internal.cbo_stats.operator_statistics import (
            ColumnStatistics,
            ConfidenceLevel,
            OperatorStatistics,
        )

        metadata = self.infer_metadata()
        if metadata.num_rows is None and metadata.size_bytes is None:
            return None

        column_stats = {}
        # Try to extract Parquet column-level statistics
        try:
            column_stats = self._extract_parquet_column_stats()
        except Exception as e:
            logger.debug("CBO: failed to extract Parquet column stats: %s", e)

        return OperatorStatistics(
            num_rows=metadata.num_rows,
            size_bytes=metadata.size_bytes,
            column_stats=column_stats,
            confidence=ConfidenceLevel.HIGH.value,
        )

    def _extract_parquet_column_stats(self):
        """Best-effort extraction of column statistics from Parquet metadata."""
        from ray.data._internal.cbo_stats.operator_statistics import (
            ColumnStatistics,
            ConfidenceLevel,
        )
        from ray.data._internal.datasource.parquet_stat_extractor import (
            extract_column_stats_from_parquet_metadata,
        )

        ds = self.datasource
        # ParquetDatasource stores sampled _ParquetFileInfo objects
        file_infos = getattr(ds, "_sampled_file_infos", None)
        if not file_infos:
            return {}

        all_column_stats = {}
        for fi in file_infos:
            pq_meta = getattr(fi, "metadata", None)
            if pq_meta is None:
                continue
            try:
                schema = pq_meta.schema.to_arrow_schema()
            except Exception:
                continue
            col_dict = extract_column_stats_from_parquet_metadata(pq_meta, schema)
            for col_name, stat in col_dict.items():
                if col_name not in all_column_stats:
                    all_column_stats[col_name] = ColumnStatistics(
                        name=col_name,
                        min_value=stat.get("min"),
                        max_value=stat.get("max"),
                        null_count=stat.get("null_count"),
                        distinct_count=stat.get("distinct_count"),
                        encoding=stat.get("encoding"),
                        compression=stat.get("compression"),
                        confidence=ConfidenceLevel.HIGH.value,
                    )
                else:
                    # Merge: widen min/max, accumulate null_count
                    existing = all_column_stats[col_name]
                    try:
                        new_min = stat.get("min")
                        new_max = stat.get("max")
                        if new_min is not None and (
                            existing.min_value is None
                            or new_min < existing.min_value
                        ):
                            existing.min_value = new_min
                        if new_max is not None and (
                            existing.max_value is None
                            or new_max > existing.max_value
                        ):
                            existing.max_value = new_max
                    except TypeError:
                        pass
                    nc = stat.get("null_count")
                    if nc is not None and existing.null_count is not None:
                        existing.null_count += nc
        return all_column_stats

    def infer_schema(self):
        return self._cached_output_metadata.schema

    def _estimate_num_outputs(self) -> Optional[int]:
        metadata = self._cached_output_metadata.metadata

        target_max_block_size = DataContext.get_current().target_max_block_size

        # In either case of
        #   - Total byte-size estimate not available
        #   - Target max-block-size not being configured
        #
        # We fallback to estimating number of outputs to be equivalent to the
        # number of input files being read (if any)
        if metadata.size_bytes is None or target_max_block_size is None:
            # NOTE: If there's no input files specified, return the count (could be 0)
            return (
                len(metadata.input_files) if metadata.input_files is not None else None
            )

        # Otherwise, estimate total number of blocks from estimated total
        # byte size
        return math.ceil(metadata.size_bytes / target_max_block_size)

    @functools.cached_property
    def _cached_output_metadata(self) -> "BlockMetadataWithSchema":
        # Legacy datasources might not implement `get_read_tasks`.
        if self.datasource.should_create_reader:
            empty_meta = BlockMetadata(None, None, None, None)
            return BlockMetadataWithSchema(metadata=empty_meta, schema=None)

        # HACK: Try to get a single read task to get the metadata.
        read_tasks = self.datasource.get_read_tasks(1)
        if len(read_tasks) == 0:
            # If there are no read tasks, the dataset is probably empty.
            empty_meta = BlockMetadata(None, None, None, None)
            return BlockMetadataWithSchema(metadata=empty_meta, schema=None)

        # `get_read_tasks` isn't guaranteed to return exactly one read task.
        metadata = [read_task.metadata for read_task in read_tasks]

        if all(meta.num_rows is not None for meta in metadata):
            num_rows = sum(meta.num_rows for meta in metadata)
            original_num_rows = num_rows
            # Apply per-block limit if set
            if self.per_block_limit is not None:
                num_rows = min(num_rows, self.per_block_limit)
        else:
            num_rows = None
            original_num_rows = None

        if all(meta.size_bytes is not None for meta in metadata):
            size_bytes = sum(meta.size_bytes for meta in metadata)
            # Pro-rate the byte size if we applied a row limit
            if (
                self.per_block_limit is not None
                and original_num_rows is not None
                and original_num_rows > 0
            ):
                size_bytes = int(size_bytes * (num_rows / original_num_rows))
        else:
            size_bytes = None

        input_files = []
        for meta in metadata:
            if meta.input_files is not None:
                input_files.extend(meta.input_files)

        meta = BlockMetadata(
            num_rows=num_rows,
            size_bytes=size_bytes,
            input_files=input_files,
            exec_stats=None,
        )
        schemas = [
            read_task.schema for read_task in read_tasks if read_task.schema is not None
        ]
        from ray.data._internal.util import unify_schemas_with_validation

        schema = None
        if schemas:
            schema = unify_schemas_with_validation(schemas)
        return BlockMetadataWithSchema(metadata=meta, schema=schema)

    def supports_projection_pushdown(self) -> bool:
        return self.datasource.supports_projection_pushdown()

    def get_projection_map(self) -> Optional[Dict[str, str]]:
        return self.datasource.get_projection_map()

    def apply_projection(
        self,
        projection_map: Optional[Dict[str, str]],
    ) -> "Read":
        clone = copy.copy(self)

        projected_datasource = self.datasource.apply_projection(projection_map)
        clone.datasource = projected_datasource
        clone.datasource_or_legacy_reader = projected_datasource

        return clone

    def get_column_renames(self) -> Optional[Dict[str, str]]:
        return self.datasource.get_column_renames()

    def supports_predicate_pushdown(self) -> bool:
        return self.datasource.supports_predicate_pushdown()

    def get_current_predicate(self) -> Optional[Expr]:
        return self.datasource.get_current_predicate()

    def apply_predicate(self, predicate_expr: Expr) -> "Read":
        predicated_datasource = self.datasource.apply_predicate(predicate_expr)

        clone = copy.copy(self)
        clone.datasource = predicated_datasource
        clone.datasource_or_legacy_reader = predicated_datasource

        return clone
