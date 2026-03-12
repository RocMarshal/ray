from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

GIB = 1024**3


@dataclass
class RunMetrics:
    """All metrics collected for a single benchmark run."""

    workload_id: str = ""
    plan_description: str = ""
    mode: str = ""
    enable_cbo: bool = False
    duration_s: float = 0.0
    reservation_ratio: float = 0.5
    operator_count: int = 0
    operators: List[Dict[str, Any]] = field(default_factory=list)
    global_bytes_spilled: int = 0
    dataset_bytes_spilled: int = 0
    streaming_schedule_s: float = 0.0
    total_input_rows: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    data_scale_bytes: Optional[int] = None
    peak_object_store_bytes: Optional[int] = None
    peak_heap_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleBenchResult:
    """Result of a single per-rule benchmark."""

    rule_name: str
    rule_category: str
    optimization_goal: str
    pipeline_desc: str
    baseline_label: str
    optimized_label: str
    baseline_time_s: float
    optimized_time_s: float
    improvement_pct: float
    sweep_detail: Optional[List[Dict[str, Any]]] = None


RuleBenchmarkRunner = Callable[[int, int], RuleBenchResult]
WorkloadRunner = Callable[[int, bool], RunMetrics]


@dataclass(frozen=True)
class RuleBenchmarkCase:
    """Registry entry for a per-rule benchmark."""

    name: str
    category: str
    optimization_goal: str
    runner: RuleBenchmarkRunner


@dataclass(frozen=True)
class EndToEndWorkloadCase:
    """Logical workload definition used to build end-to-end execution plans."""

    workload_id: str
    mode: str
    num_rows: int
    runner_key: str
    description: str


@dataclass(frozen=True)
class WorkloadExecutionCase:
    """Concrete run produced from a workload definition and CBO toggle."""

    run_id: str
    workload_id: str
    mode: str
    num_rows: int
    cbo_enabled: bool
    repetition: int
    runner_key: str


@dataclass(frozen=True)
class ParameterizedWorkloadCase:
    """Parameterized same-chain workload used by matrix/profile experiments."""

    workload_id: str
    num_rows: int
    transform_ops: int
    payload_bytes_per_row: int
    description: str


@dataclass(frozen=True)
class ParameterizedExecutionCase:
    """Concrete execution of a parameterized workload."""

    run_id: str
    workload_id: str
    num_rows: int
    transform_ops: int
    payload_bytes_per_row: int
    cbo_enabled: bool
    repetition: int


@dataclass
class RValueMatrixResult:
    """Result cell for the reservation-ratio matrix experiment."""

    workload_id: str
    num_rows: int
    transform_ops: int
    derived_ratio: float
    derived_duration_s: float
    best_ratio: float
    best_duration_s: float
    default_ratio: float
    default_duration_s: float
    r_error: float
    operator_count: int
    data_scale_bytes: Optional[int] = None
    peak_object_store_bytes: Optional[int] = None
    peak_heap_bytes: Optional[int] = None
    spill_bytes: int = 0
    pipeline_desc: str = ""
    sweep_detail: Optional[List[Dict[str, Any]]] = None
    subtask_index: Optional[int] = None
    subtask_count: Optional[int] = None


@dataclass(frozen=True)
class SubtaskAssignment:
    """Execution shard definition for matrix/profile workloads."""

    experiment_type: str
    subtask_index: int
    subtask_count: int
    item_ids: Tuple[str, ...]


def build_default_end_to_end_workloads(
    batch_rows: int,
    streaming_rows: int,
) -> List[EndToEndWorkloadCase]:
    return [
        EndToEndWorkloadCase(
            workload_id="batch_primary",
            mode="batch",
            num_rows=batch_rows,
            runner_key="batch",
            description="range -> filter -> map_batches -> repartition -> sort",
        ),
        EndToEndWorkloadCase(
            workload_id="streaming_primary",
            mode="streaming",
            num_rows=streaming_rows,
            runner_key="streaming",
            description="range -> random_shuffle -> map_batches -> repartition",
        ),
    ]


def build_end_to_end_execution_plan(
    batch_rows: int,
    streaming_rows: int,
    repetitions: int,
) -> List[WorkloadExecutionCase]:
    plan: List[WorkloadExecutionCase] = []
    for rep in range(1, repetitions + 1):
        for workload in build_default_end_to_end_workloads(batch_rows, streaming_rows):
            for cbo_enabled in (False, True):
                cbo_label = "on" if cbo_enabled else "off"
                plan.append(
                    WorkloadExecutionCase(
                        run_id=(
                            f"{workload.workload_id}-rep{rep}-cbo-{cbo_label}"
                        ),
                        workload_id=workload.workload_id,
                        mode=workload.mode,
                        num_rows=workload.num_rows,
                        cbo_enabled=cbo_enabled,
                        repetition=rep,
                        runner_key=workload.runner_key,
                    )
                )
    return plan


def build_parameterized_workloads(
    row_counts: Sequence[int],
    operator_counts: Sequence[int],
    payload_bytes_per_row: int = 0,
) -> List[ParameterizedWorkloadCase]:
    workloads: List[ParameterizedWorkloadCase] = []
    for num_rows in row_counts:
        for transform_ops in operator_counts:
            workloads.append(
                ParameterizedWorkloadCase(
                    workload_id=f"chain_rows{num_rows}_ops{transform_ops}",
                    num_rows=num_rows,
                    transform_ops=transform_ops,
                    payload_bytes_per_row=payload_bytes_per_row,
                    description=(
                        "range -> filter -> map_batches*xN -> repartition -> sort"
                    ),
                )
            )
    return workloads


def derive_payload_bytes_per_row(
    target_total_bytes: int,
    target_rows: int,
    base_row_bytes: int = 8,
) -> int:
    if target_total_bytes <= 0:
        raise ValueError("target_total_bytes must be positive")
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    avg_bytes_per_row = (target_total_bytes + target_rows - 1) // target_rows
    return max(int(avg_bytes_per_row - base_row_bytes), 0)


def resolve_memory_profile_preset(
    preset_name: str,
    operator_counts: Sequence[int],
) -> Dict[str, Any]:
    normalized = preset_name.strip().lower()
    if normalized != "10gb_10m":
        raise ValueError(
            f"Unknown memory-profile preset: {preset_name}. "
            "Available presets: 10gb_10m"
        )

    target_total_bytes = 10 * GIB
    target_rows = 10_000_000
    payload_bytes_per_row = derive_payload_bytes_per_row(
        target_total_bytes=target_total_bytes,
        target_rows=target_rows,
    )
    return {
        "preset_name": "10gb_10m",
        "row_counts": [target_rows],
        "operator_counts": list(operator_counts),
        "payload_bytes_per_row": payload_bytes_per_row,
        "target_total_bytes": target_total_bytes,
        "target_rows": target_rows,
        "description": "10 GiB synthetic input at 10 million rows",
    }


def build_memory_profile_plan(
    row_counts: Sequence[int],
    operator_counts: Sequence[int],
    repetitions: int,
    payload_bytes_per_row: int,
) -> List[ParameterizedExecutionCase]:
    plan: List[ParameterizedExecutionCase] = []
    for rep in range(1, repetitions + 1):
        for workload in build_parameterized_workloads(
            row_counts=row_counts,
            operator_counts=operator_counts,
            payload_bytes_per_row=payload_bytes_per_row,
        ):
            for cbo_enabled in (False, True):
                cbo_label = "on" if cbo_enabled else "off"
                plan.append(
                    ParameterizedExecutionCase(
                        run_id=f"{workload.workload_id}-rep{rep}-cbo-{cbo_label}",
                        workload_id=workload.workload_id,
                        num_rows=workload.num_rows,
                        transform_ops=workload.transform_ops,
                        payload_bytes_per_row=workload.payload_bytes_per_row,
                        cbo_enabled=cbo_enabled,
                        repetition=rep,
                    )
                )
    return plan


def _validate_subtask_selection(
    subtask_index: int,
    subtask_count: int,
) -> None:
    if subtask_count <= 0:
        raise ValueError("subtask_count must be positive")
    if subtask_index < 0 or subtask_index >= subtask_count:
        raise ValueError(
            f"subtask_index must be within [0, {subtask_count - 1}], got {subtask_index}"
        )


def _split_evenly(items: Sequence[Any], num_chunks: int) -> List[List[Any]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    chunks: List[List[Any]] = [[] for _ in range(num_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % num_chunks].append(item)
    return chunks


def shard_parameterized_workloads(
    workloads: Sequence[ParameterizedWorkloadCase],
    subtask_index: int,
    subtask_count: int,
) -> List[ParameterizedWorkloadCase]:
    _validate_subtask_selection(subtask_index, subtask_count)
    return list(_split_evenly(list(workloads), subtask_count)[subtask_index])


def build_parameterized_subtask_manifest(
    workloads: Sequence[ParameterizedWorkloadCase],
    subtask_count: int,
    experiment_type: str,
) -> List[SubtaskAssignment]:
    chunks = _split_evenly(list(workloads), subtask_count)
    manifest: List[SubtaskAssignment] = []
    for subtask_index, chunk in enumerate(chunks):
        manifest.append(
            SubtaskAssignment(
                experiment_type=experiment_type,
                subtask_index=subtask_index,
                subtask_count=subtask_count,
                item_ids=tuple(workload.workload_id for workload in chunk),
            )
        )
    return manifest


def shard_memory_profile_plan(
    plan: Sequence[ParameterizedExecutionCase],
    subtask_index: int,
    subtask_count: int,
) -> List[ParameterizedExecutionCase]:
    _validate_subtask_selection(subtask_index, subtask_count)
    grouped_cases: Dict[Tuple[str, int], List[ParameterizedExecutionCase]] = {}
    group_order: List[Tuple[str, int]] = []
    for case in plan:
        key = (case.workload_id, case.repetition)
        if key not in grouped_cases:
            grouped_cases[key] = []
            group_order.append(key)
        grouped_cases[key].append(case)

    grouped_plan = [grouped_cases[key] for key in group_order]
    selected_groups = _split_evenly(grouped_plan, subtask_count)[subtask_index]
    shard: List[ParameterizedExecutionCase] = []
    for group in selected_groups:
        shard.extend(group)
    return shard


def build_memory_profile_subtask_manifest(
    plan: Sequence[ParameterizedExecutionCase],
    subtask_count: int,
) -> List[SubtaskAssignment]:
    grouped_cases: Dict[Tuple[str, int], List[ParameterizedExecutionCase]] = {}
    group_order: List[Tuple[str, int]] = []
    for case in plan:
        key = (case.workload_id, case.repetition)
        if key not in grouped_cases:
            grouped_cases[key] = []
            group_order.append(key)
        grouped_cases[key].append(case)

    grouped_plan = [grouped_cases[key] for key in group_order]
    chunks = _split_evenly(grouped_plan, subtask_count)
    manifest: List[SubtaskAssignment] = []
    for subtask_index, groups in enumerate(chunks):
        item_ids: List[str] = []
        for group in groups:
            item_ids.append(f"{group[0].workload_id}:rep{group[0].repetition}")
        manifest.append(
            SubtaskAssignment(
                experiment_type="memory_profile",
                subtask_index=subtask_index,
                subtask_count=subtask_count,
                item_ids=tuple(item_ids),
            )
        )
    return manifest


def describe_rule_catalog(cases: Sequence[RuleBenchmarkCase]) -> str:
    lines = ["Per-rule benchmark catalog:"]
    for case in cases:
        lines.append(
            f"  - {case.name} [{case.category}/{case.optimization_goal}]"
        )
    return "\n".join(lines)


def describe_end_to_end_plan(plan: Sequence[WorkloadExecutionCase]) -> str:
    lines = ["End-to-end execution plan:"]
    for case in plan:
        cbo_label = "ON" if case.cbo_enabled else "OFF"
        lines.append(
            f"  - {case.run_id}: mode={case.mode}, rows={case.num_rows}, "
            f"CBO={cbo_label}"
        )
    return "\n".join(lines)


def describe_parameterized_workloads(
    workloads: Sequence[ParameterizedWorkloadCase],
    title: str,
) -> str:
    lines = [title]
    for workload in workloads:
        lines.append(
            f"  - {workload.workload_id}: rows={workload.num_rows}, "
            f"transform_ops={workload.transform_ops}, "
            f"payload_bytes_per_row={workload.payload_bytes_per_row}"
        )
    return "\n".join(lines)


def describe_memory_profile_plan(
    plan: Sequence[ParameterizedExecutionCase],
) -> str:
    lines = ["Memory-profile execution plan:"]
    for case in plan:
        cbo_label = "ON" if case.cbo_enabled else "OFF"
        lines.append(
            f"  - {case.run_id}: rows={case.num_rows}, "
            f"transform_ops={case.transform_ops}, "
            f"payload_bytes_per_row={case.payload_bytes_per_row}, "
            f"CBO={cbo_label}"
        )
    return "\n".join(lines)


def describe_subtask_manifest(
    manifest: Sequence[SubtaskAssignment],
    title: str,
) -> str:
    lines = [title]
    for assignment in manifest:
        lines.append(
            f"  - shard {assignment.subtask_index}/{assignment.subtask_count}: "
            f"{len(assignment.item_ids)} item(s) -> {', '.join(assignment.item_ids)}"
        )
    return "\n".join(lines)


def select_rule_benchmark_cases(
    all_cases: Sequence[RuleBenchmarkCase],
    rules: Optional[Sequence[str]] = None,
) -> List[RuleBenchmarkCase]:
    if not rules:
        return list(all_cases)

    mapping = {case.name.lower(): case for case in all_cases}
    selected: List[RuleBenchmarkCase] = []
    unknown: List[str] = []
    seen: set[str] = set()

    for rule in rules:
        normalized = rule.strip().lower()
        if not normalized:
            continue
        case = mapping.get(normalized)
        if case is None:
            unknown.append(rule.strip())
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(case)

    if unknown:
        available = ", ".join(case.name for case in all_cases)
        raise ValueError(
            f"Unknown benchmark rule(s): {', '.join(unknown)}. "
            f"Available rules: {available}"
        )

    return selected


def run_rule_benchmark_suite(
    cases: Sequence[RuleBenchmarkCase],
    num_rows: int,
    repetitions: int,
    log_fn: Callable[[str], None] = print,
) -> List[RuleBenchResult]:
    results: List[RuleBenchResult] = []
    for case in cases:
        log_fn(f"  [{case.name}] Running ...")
        result = case.runner(num_rows, repetitions)
        results.append(result)
        log_fn(
            f"  [{case.name}] baseline={result.baseline_time_s:.3f}s  "
            f"optimized={result.optimized_time_s:.3f}s  "
            f"delta={result.improvement_pct:+.1f}%"
        )
    return results


def execute_end_to_end_plan(
    plan: Sequence[WorkloadExecutionCase],
    runner_registry: Dict[str, WorkloadRunner],
    repetitions: int,
    log_fn: Callable[[str], None] = print,
) -> List[RunMetrics]:
    results: List[RunMetrics] = []
    for case in plan:
        runner = runner_registry.get(case.runner_key)
        if runner is None:
            raise KeyError(
                f"No workload runner registered for key '{case.runner_key}'"
            )
        cbo_label = "ON" if case.cbo_enabled else "OFF"
        log_fn(
            f"[rep {case.repetition}/{repetitions}] "
            f"{case.workload_id} mode={case.mode} CBO={cbo_label}"
        )
        results.append(runner(case.num_rows, case.cbo_enabled))
    return results


def aggregate_runs(runs: Sequence[RunMetrics]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[RunMetrics]] = {}
    for run in runs:
        key = f"{run.mode}-cbo-{run.enable_cbo}"
        grouped.setdefault(key, []).append(run)

    aggregates: Dict[str, Dict[str, Any]] = {}
    for key, values in grouped.items():
        durations = [value.duration_s for value in values]
        aggregate: Dict[str, Any] = {
            "runs": len(values),
            "duration_s_mean": round(statistics.mean(durations), 4),
            "duration_s_stdev": round(statistics.pstdev(durations), 4),
            "reservation_ratio_mean": round(
                statistics.mean(value.reservation_ratio for value in values), 4
            ),
            "operator_count": values[0].operator_count,
            "global_bytes_spilled_mean": int(
                statistics.mean(value.global_bytes_spilled for value in values)
            ),
            "dataset_bytes_spilled_mean": int(
                statistics.mean(value.dataset_bytes_spilled for value in values)
            ),
        }
        if values[0].mode == "streaming":
            aggregate["latency_p50_ms_mean"] = round(
                statistics.mean(value.latency_p50_ms for value in values), 3
            )
            aggregate["latency_p95_ms_mean"] = round(
                statistics.mean(value.latency_p95_ms for value in values), 3
            )
            aggregate["latency_p99_ms_mean"] = round(
                statistics.mean(value.latency_p99_ms for value in values), 3
            )
        aggregates[key] = aggregate

    return aggregates


def format_end_to_end_comparison(aggregates: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  Ray Data CBO Benchmark Results")
    lines.append("=" * 72)

    for mode in ("batch", "streaming"):
        off_key = f"{mode}-cbo-False"
        on_key = f"{mode}-cbo-True"
        off = aggregates.get(off_key, {})
        on = aggregates.get(on_key, {})
        if not off or not on:
            continue

        lines.append(f"\n  [{mode.upper()}]")
        lines.append(f"  {'Metric':<35} {'CBO OFF':>12} {'CBO ON':>12} {'Delta':>10}")
        lines.append(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

        duration_off = off["duration_s_mean"]
        duration_on = on["duration_s_mean"]
        delta_pct = (
            (duration_on - duration_off) / duration_off * 100
            if duration_off > 0
            else 0
        )
        lines.append(
            f"  {'Duration (s)':<35} {duration_off:>12.3f} {duration_on:>12.3f} "
            f"{delta_pct:>+9.1f}%"
        )
        lines.append(
            f"  {'Reservation Ratio':<35} "
            f"{off['reservation_ratio_mean']:>12.3f} "
            f"{on['reservation_ratio_mean']:>12.3f}"
        )

        spill_off = off["global_bytes_spilled_mean"]
        spill_on = on["global_bytes_spilled_mean"]
        lines.append(
            f"  {'Global Spill (MB)':<35} "
            f"{spill_off / (1024**2):>12.1f} "
            f"{spill_on / (1024**2):>12.1f}"
        )

        if mode == "streaming":
            for pct in ("p50", "p95", "p99"):
                key_name = f"latency_{pct}_ms_mean"
                value_off = off.get(key_name, 0)
                value_on = on.get(key_name, 0)
                lines.append(
                    f"  {'Latency ' + pct + ' (ms)':<35} "
                    f"{value_off:>12.3f} {value_on:>12.3f}"
                )

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


def _fmt_sweep(
    entries: List[Dict[str, Any]],
    key: str,
    label: str,
) -> List[str]:
    lines: List[str] = []
    default_t = next(entry["time_s"] for entry in entries if entry["is_default"])
    best_t = min(entry["time_s"] for entry in entries)
    lines.append(f"    {label:<12} {'Time (s)':>10}  {'vs Default':>12}")
    lines.append(f"    {'---' * 13}")
    for entry in entries:
        value = entry[key]
        delta = (
            (entry["time_s"] - default_t) / default_t * 100
            if default_t > 0
            else 0
        )
        tag = "  (default)" if entry["is_default"] else ""
        if entry["time_s"] == best_t and not entry["is_default"]:
            tag = "  <- best"
        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
        lines.append(
            f"    {formatted_value:<12} {entry['time_s']:>10.3f} "
            f" {delta:>+11.1f}%{tag}"
        )
    return lines


def format_rule_results_table(
    results: Sequence[RuleBenchResult],
    num_rows: int,
    repetitions: int,
) -> str:
    width = 88
    separator = "=" * width
    lines: List[str] = [
        separator,
        "  Ray Data Optimization Rules - Per-Rule Benchmark Results",
        separator,
        f"  Config: {num_rows:,} rows x {repetitions} rep(s)\n",
    ]
    header = (
        f"  {'Rule':<26} {'Type':>4}  {'Goal':<11}"
        f"{'Baseline':>10} {'Optimized':>10} {'Improv.':>9}"
    )
    lines.append(header)
    lines.append(f"  {'-' * (width - 4)}")
    for result in results:
        arrow = "+" if result.improvement_pct >= 0 else "-"
        lines.append(
            f"  {result.rule_name:<26} {result.rule_category:>4}  "
            f"{result.optimization_goal:<11}"
            f"{result.baseline_time_s:>9.3f}s {result.optimized_time_s:>9.3f}s "
            f"{arrow}{abs(result.improvement_pct):>7.1f}%"
        )
    lines.append(f"  {'-' * (width - 4)}")
    lines.append("\n  Legend:")
    for result in results:
        lines.append(f"    {result.rule_name:<26} {result.pipeline_desc}")
        lines.append(
            f"    {'':<26} baseline: {result.baseline_label}"
            f" | optimized: {result.optimized_label}"
        )
    for result in results:
        if not result.sweep_detail:
            continue
        lines.append(f"\n  {result.rule_name} - Parameter Sweep:")
        if "ratio" in result.sweep_detail[0]:
            lines.extend(_fmt_sweep(result.sweep_detail, "ratio", "Ratio"))
        elif "num_partitions" in result.sweep_detail[0]:
            lines.extend(_fmt_sweep(result.sweep_detail, "num_partitions", "N parts"))
    lines.append(f"\n{separator}")
    return "\n".join(lines)


def format_r_value_matrix_table(
    results: Sequence[RValueMatrixResult],
) -> str:
    width = 108
    separator = "=" * width
    lines: List[str] = [
        separator,
        "  Ray Data CBO Reservation Ratio Matrix",
        separator,
        (
            f"  {'Workload':<28} {'Rows':>10} {'Ops':>6} {'Derived R':>10} "
            f"{'Best R':>10} {'R Error':>10} {'Derived (s)':>12} {'Best (s)':>10}"
        ),
        f"  {'-' * (width - 4)}",
    ]
    for result in results:
        lines.append(
            f"  {result.workload_id:<28} {result.num_rows:>10,} "
            f"{result.transform_ops:>6} {result.derived_ratio:>10.3f} "
            f"{result.best_ratio:>10.3f} {result.r_error:>10.3f} "
            f"{result.derived_duration_s:>12.3f} {result.best_duration_s:>10.3f}"
        )
    lines.append(f"  {'-' * (width - 4)}")
    lines.append("\n  Sweep details:")
    for result in results:
        lines.append(
            f"    {result.workload_id}: default={result.default_ratio:.2f} "
            f"({result.default_duration_s:.3f}s), derived={result.derived_ratio:.2f}, "
            f"best={result.best_ratio:.2f}"
        )
    lines.append(f"\n{separator}")
    return "\n".join(lines)


def format_memory_profile_table(
    runs: Sequence[RunMetrics],
) -> str:
    width = 120
    separator = "=" * width
    lines: List[str] = [
        separator,
        "  Ray Data CBO Memory Profile",
        separator,
        (
            f"  {'Workload':<28} {'CBO':>5} {'Rows':>10} {'Ops':>6} "
            f"{'Peak Heap MB':>14} {'Peak Obj MB':>14} {'Spill MB':>12} "
            f"{'R':>6} {'Time (s)':>10}"
        ),
        f"  {'-' * (width - 4)}",
    ]
    for run in runs:
        transform_ops = run.metadata.get("transform_ops")
        spill_bytes = max(run.global_bytes_spilled, run.dataset_bytes_spilled)
        peak_heap_mb = (
            (run.peak_heap_bytes or 0) / (1024**2)
            if run.peak_heap_bytes is not None
            else 0.0
        )
        peak_obj_mb = (
            (run.peak_object_store_bytes or 0) / (1024**2)
            if run.peak_object_store_bytes is not None
            else 0.0
        )
        lines.append(
            f"  {run.workload_id:<28} "
            f"{('ON' if run.enable_cbo else 'OFF'):>5} "
            f"{run.total_input_rows:>10,} "
            f"{str(transform_ops):>6} "
            f"{peak_heap_mb:>14.1f} "
            f"{peak_obj_mb:>14.1f} "
            f"{spill_bytes / (1024**2):>12.1f} "
            f"{run.reservation_ratio:>6.2f} "
            f"{run.duration_s:>10.3f}"
        )
    lines.append(f"\n{separator}")
    return "\n".join(lines)
