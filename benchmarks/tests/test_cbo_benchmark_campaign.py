import importlib.util
import json
import sys
import time
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schema = _load_module(
    "cbo_benchmark_schema",
    "python/ray/data/benchmarks/cbo_benchmark_schema.py",
)
harness = _load_module(
    "cbo_benchmark_harness",
    "python/ray/data/benchmarks/cbo_benchmark_harness.py",
)
campaign = _load_module(
    "cbo_benchmark_campaign",
    "python/ray/data/benchmarks/cbo_benchmark_campaign.py",
)


def test_build_campaign_plan_for_matrix_shards_materializes_support_files(tmp_path):
    spec = campaign.build_campaign_spec_template()
    spec["output_dir"] = str(tmp_path / "campaign_out")
    spec["experiments"]["memory_profile"]["enabled"] = False
    spec["experiments"]["per_rule"]["enabled"] = False
    spec["experiments"]["end_to_end"]["enabled"] = False
    spec["experiments"]["r_value_matrix"] = {
        "enabled": True,
        "row_scales": [1000, 2000],
        "operator_counts": [2],
        "sweep_ratios": [0.3, 0.5],
        "repetitions": 1,
        "subtask_count": 2,
    }

    plan = campaign.build_campaign_plan(spec)
    task_ids = [task["task_id"] for task in plan["tasks"]]

    assert task_ids == [
        "r_value_matrix.run.0",
        "r_value_matrix.run.1",
        "r_value_matrix.finalize",
    ]
    campaign.write_campaign_plan(str(tmp_path / "campaign_plan.json"), plan)

    manifest_path = tmp_path / "campaign_out" / "r_value_matrix" / "manifest.json"
    input_results_path = (
        tmp_path / "campaign_out" / "r_value_matrix" / "input_results.json"
    )
    assert manifest_path.exists()
    assert input_results_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_results = json.loads(input_results_path.read_text(encoding="utf-8"))

    assert len(manifest["r_value_matrix"]) == 2
    assert len(input_results) == 2
    assert input_results[0].endswith("shard_000_of_2.json")
    assert "r_value_matrix.finalize" in campaign.render_campaign_plan_summary(plan)


def test_execute_campaign_plan_finalizes_existing_matrix_results_into_bundle(tmp_path):
    payload0 = schema.build_r_value_matrix_payload(
        [
            {
                "workload_id": "chain_rows1000_ops2",
                "num_rows": 1000,
                "transform_ops": 2,
                "derived_ratio": 0.50,
                "derived_duration_s": 1.4,
                "best_ratio": 0.30,
                "best_duration_s": 1.0,
                "default_ratio": 0.50,
                "default_duration_s": 1.4,
                "r_error": 0.20,
                "operator_count": 4,
                "subtask_index": 0,
                "subtask_count": 2,
            }
        ],
        {
            "row_scales": [1000, 2000],
            "operator_counts": [2],
            "sweep_ratios": [0.3, 0.5],
            "repetitions": 1,
            "subtask_index": 0,
            "subtask_count": 2,
            "selected_workload_ids": ["chain_rows1000_ops2"],
        },
    )
    payload1 = schema.build_r_value_matrix_payload(
        [
            {
                "workload_id": "chain_rows2000_ops2",
                "num_rows": 2000,
                "transform_ops": 2,
                "derived_ratio": 0.34,
                "derived_duration_s": 1.8,
                "best_ratio": 0.30,
                "best_duration_s": 1.6,
                "default_ratio": 0.50,
                "default_duration_s": 2.2,
                "r_error": 0.04,
                "operator_count": 4,
                "subtask_index": 1,
                "subtask_count": 2,
            }
        ],
        {
            "row_scales": [1000, 2000],
            "operator_counts": [2],
            "sweep_ratios": [0.3, 0.5],
            "repetitions": 1,
            "subtask_index": 1,
            "subtask_count": 2,
            "selected_workload_ids": ["chain_rows2000_ops2"],
        },
    )
    shard0_path = tmp_path / "existing" / "shard0.json"
    shard1_path = tmp_path / "existing" / "shard1.json"
    manifest_path = tmp_path / "existing" / "manifest.json"
    shard0_path.parent.mkdir(parents=True, exist_ok=True)
    shard0_path.write_text(json.dumps(payload0), encoding="utf-8")
    shard1_path.write_text(json.dumps(payload1), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "r_value_matrix": [
                    {
                        "experiment_type": "r_value_matrix",
                        "subtask_index": 0,
                        "subtask_count": 2,
                        "item_ids": ["chain_rows1000_ops2"],
                    },
                    {
                        "experiment_type": "r_value_matrix",
                        "subtask_index": 1,
                        "subtask_count": 2,
                        "item_ids": ["chain_rows2000_ops2"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    spec = campaign.build_campaign_spec_template()
    spec["output_dir"] = str(tmp_path / "campaign_out")
    spec["experiments"]["memory_profile"]["enabled"] = False
    spec["experiments"]["per_rule"]["enabled"] = False
    spec["experiments"]["end_to_end"]["enabled"] = False
    spec["experiments"]["r_value_matrix"] = {
        "enabled": True,
        "input_results": [str(shard0_path), str(shard1_path)],
        "manifest_path": str(manifest_path),
    }

    plan = campaign.build_campaign_plan(spec)
    assert [task["task_id"] for task in plan["tasks"]] == ["r_value_matrix.finalize"]

    result = campaign.execute_campaign_plan(plan)
    assert result["task_results"][0]["status"] == "completed"

    bundle_manifest_path = (
        tmp_path
        / "campaign_out"
        / "r_value_matrix"
        / "bundle"
        / "bundle_manifest.json"
    )
    community_qa_path = (
        tmp_path
        / "campaign_out"
        / "r_value_matrix"
        / "bundle"
        / "cbo_benchmark_community_qa.md"
    )
    assert bundle_manifest_path.exists()
    assert community_qa_path.exists()

    rerun = campaign.execute_campaign_plan(plan)
    assert rerun["task_results"][0]["status"] == "skipped_existing"


def test_execute_campaign_plan_runs_independent_shards_in_parallel(tmp_path):
    helper_script = tmp_path / "write_after_sleep.py"
    helper_script.write_text(
        "\n".join(
            [
                "import sys",
                "import time",
                "from pathlib import Path",
                "time.sleep(float(sys.argv[2]))",
                "Path(sys.argv[1]).write_text('done', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    output0 = tmp_path / "parallel" / "out0.txt"
    output1 = tmp_path / "parallel" / "out1.txt"
    final_output = tmp_path / "parallel" / "final.txt"
    plan = {
        "campaign_name": "parallel-smoke",
        "output_dir": str(tmp_path / "parallel"),
        "executor": {
            "max_parallelism": 2,
            "local_max_parallelism": 2,
            "allow_local_fallback_for_remote_stages": True,
            "targets": [],
            "state_path": str(tmp_path / "parallel" / "execution_state.json"),
        },
        "tasks": [
            {
                "task_id": "run.0",
                "experiment_type": "dummy",
                "stage": "run_shard",
                "description": "parallel task 0",
                "command": [sys.executable, str(helper_script), str(output0), "0.3"],
                "output_path": str(output0),
                "log_path": str(tmp_path / "parallel" / "logs" / "run0.log"),
                "dependency_ids": [],
                "cwd": str(tmp_path),
                "resume_skip_if_output_exists": True,
            },
            {
                "task_id": "run.1",
                "experiment_type": "dummy",
                "stage": "run_shard",
                "description": "parallel task 1",
                "command": [sys.executable, str(helper_script), str(output1), "0.3"],
                "output_path": str(output1),
                "log_path": str(tmp_path / "parallel" / "logs" / "run1.log"),
                "dependency_ids": [],
                "cwd": str(tmp_path),
                "resume_skip_if_output_exists": True,
            },
            {
                "task_id": "finalize",
                "experiment_type": "dummy",
                "stage": "finalize",
                "description": "finalize task",
                "command": [sys.executable, str(helper_script), str(final_output), "0.0"],
                "output_path": str(final_output),
                "log_path": str(tmp_path / "parallel" / "logs" / "finalize.log"),
                "dependency_ids": ["run.0", "run.1"],
                "cwd": str(tmp_path),
                "resume_skip_if_output_exists": True,
            },
        ],
        "support_files": [],
    }

    start = time.perf_counter()
    result = campaign.execute_campaign_plan(plan)
    elapsed = time.perf_counter() - start

    assert len(result["task_results"]) == 3
    assert elapsed < 0.75
    state = json.loads(
        (tmp_path / "parallel" / "execution_state.json").read_text(encoding="utf-8")
    )
    assert state["tasks"]["run.0"]["status"] == "completed"
    assert state["tasks"]["finalize"]["status"] == "completed"


def test_execute_campaign_plan_dispatches_run_shard_to_remote_wrapper(tmp_path):
    helper_script = tmp_path / "remote_helper.py"
    helper_script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                "Path(sys.argv[1]).write_text(json.dumps({'ok': True}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "remote" / "out.json"
    plan = {
        "campaign_name": "remote-smoke",
        "output_dir": str(tmp_path / "remote"),
        "executor": {
            "max_parallelism": 1,
            "local_max_parallelism": 1,
            "allow_local_fallback_for_remote_stages": False,
            "targets": [
                {
                    "name": "loopback",
                    "capacity": 1,
                    "stages": ["run_shard"],
                    "backend": {
                        "type": "template",
                        "command_template": ["sh", "-lc", "{command}"],
                    },
                }
            ],
            "state_path": str(tmp_path / "remote" / "execution_state.json"),
        },
        "tasks": [
            {
                "task_id": "run.0",
                "experiment_type": "dummy",
                "stage": "run_shard",
                "description": "remote wrapped task",
                "command": [sys.executable, str(helper_script), str(output_path)],
                "output_path": str(output_path),
                "log_path": str(tmp_path / "remote" / "logs" / "run0.log"),
                "dependency_ids": [],
                "cwd": str(tmp_path),
                "resume_skip_if_output_exists": True,
            }
        ],
        "support_files": [],
    }

    result = campaign.execute_campaign_plan(plan)
    assert result["task_results"][0]["assigned_target"] == "loopback"
    state = json.loads(
        (tmp_path / "remote" / "execution_state.json").read_text(encoding="utf-8")
    )
    assert state["tasks"]["run.0"]["assigned_target"] == "loopback"
    assert output_path.exists()


def test_build_backend_command_renders_standard_backends(tmp_path):
    task = {
        "task_id": "r_value_matrix.run.0",
        "command": ["python3", "cbo_benchmark.py", "--r-value-matrix"],
        "cwd": "/workspace/ray",
        "output_path": str(tmp_path / "out.json"),
        "log_path": str(tmp_path / "run.log"),
    }

    ssh_command, _ = campaign.build_backend_command(
        task,
        {
            "name": "ssh-a",
            "backend": {
                "type": "ssh",
                "host": "worker-a",
                "ssh_binary": "ssh",
                "ssh_args": ["-p", "2222"],
                "remote_working_dir": "/remote/ray",
            },
        },
    )
    ray_job_command, _ = campaign.build_backend_command(
        task,
        {
            "name": "ray-job-a",
            "backend": {
                "type": "ray_job",
                "address": "http://127.0.0.1:8265",
                "submission_id_prefix": "phase62",
                "working_dir": "/workspace/ray",
            },
        },
    )
    k8s_command, _ = campaign.build_backend_command(
        task,
        {
            "name": "k8s-a",
            "backend": {
                "type": "k8s_job",
                "image": "python:3.12",
                "namespace": "bench",
                "job_name_prefix": "phase62",
            },
        },
    )

    assert ssh_command[0] == "ssh"
    assert "worker-a" in ssh_command
    assert "cd /remote/ray &&" in ssh_command[-1]
    assert ray_job_command[:4] == ["ray", "job", "submit", "--address"]
    assert "phase62-r-value-matrix-run-0" in " ".join(ray_job_command)
    assert k8s_command[:2] == ["sh", "-lc"]
    assert "kubectl create job" in k8s_command[-1]
