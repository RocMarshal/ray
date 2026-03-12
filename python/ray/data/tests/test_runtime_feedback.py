"""
Ray Data CBO Phase 4: Runtime Feedback Loop System Complete Tests

Source Location: python/ray/data/tests/test_runtime_feedback.py

Test Coverage:
1. RuntimeFeedbackCollector core functionality
2. Runtime metrics collection and persistence
3. Feedback loading and application
4. LRU cache eviction
5. Exponential moving average calculation
"""

import pytest
import tempfile
import time
from pathlib import Path

from ray.data._internal.cbo_stats.runtime_feedback_collector import (
    RuntimeFeedbackCollector,
    OpRuntimeMetrics,
    FeedbackCacheEntry,
)


class TestOpRuntimeMetrics:
    """Unit tests for OpRuntimeMetrics"""

    def test_create_metrics(self):
        """Test creating runtime metrics"""
        metrics = OpRuntimeMetrics(
            op_name="test_op",
            actual_rows=1000,
            actual_bytes=1024 * 1024,
            actual_avg_bytes_per_row=1024.0,
            actual_wall_time_seconds=5.0,
        )

        assert metrics.op_name == "test_op"
        assert metrics.actual_rows == 1000
        assert metrics.actual_bytes == 1024 * 1024
        assert metrics.actual_avg_bytes_per_row == 1024.0

    def test_metrics_to_dict(self):
        """Test metrics serialization"""
        metrics = OpRuntimeMetrics(
            op_name="test_op",
            actual_rows=1000,
            actual_bytes=1024 * 1024,
        )

        data = metrics.to_dict()
        assert data['op_name'] == "test_op"
        assert data['actual_rows'] == 1000

    def test_metrics_from_dict(self):
        """Test metrics deserialization"""
        data = {
            'op_name': 'test_op',
            'actual_rows': 1000,
            'actual_bytes': 1024 * 1024,
            'actual_avg_bytes_per_row': None,
            'actual_wall_time_seconds': None,
            'execution_timestamp': None,
        }

        metrics = OpRuntimeMetrics.from_dict(data)
        assert metrics.op_name == 'test_op'
        assert metrics.actual_rows == 1000


class TestFeedbackCacheEntry:
    """Unit tests for FeedbackCacheEntry"""

    def test_create_entry(self):
        """Test creating cache entry"""
        metrics = {
            'op1': OpRuntimeMetrics(op_name='op1', actual_rows=1000),
            'op2': OpRuntimeMetrics(op_name='op2', actual_rows=2000),
        }

        entry = FeedbackCacheEntry(
            plan_hash='abc123',
            op_metrics=metrics,
            created_timestamp=time.time(),
            last_updated_timestamp=time.time(),
        )

        assert entry.plan_hash == 'abc123'
        assert len(entry.op_metrics) == 2

    def test_entry_serialization(self):
        """Test cache entry serialization"""
        metrics = {
            'op1': OpRuntimeMetrics(op_name='op1', actual_rows=1000),
        }

        entry = FeedbackCacheEntry(
            plan_hash='abc123',
            op_metrics=metrics,
            created_timestamp=time.time(),
            last_updated_timestamp=time.time(),
        )

        data = entry.to_dict()
        assert data['plan_hash'] == 'abc123'
        assert 'op1' in data['op_metrics']

        # Deserialization
        new_entry = FeedbackCacheEntry.from_dict(data)
        assert new_entry.plan_hash == 'abc123'


class TestRuntimeFeedbackCollector:
    """Unit tests for RuntimeFeedbackCollector"""

    def test_collector_initialization(self):
        """Test collector initialization"""
        collector = RuntimeFeedbackCollector()
        assert collector.CACHE_DIR == Path.home() / '.ray' / 'data' / 'stats_cache'

    def test_compute_plan_hash(self):
        """Test plan hash computation"""
        collector = RuntimeFeedbackCollector()

        class MockPlan:
            class MockDAG:
                dag_str = "Read->Filter->Map"

            dag = MockDAG()

        plan = MockPlan()
        hash_val = collector._compute_plan_hash(plan)

        assert len(hash_val) == 64  # SHA256 hash length
        assert hash_val == collector._compute_plan_hash(plan)  # Idempotency

    def test_extract_dag_structure(self):
        """Test DAG structure extraction"""
        collector = RuntimeFeedbackCollector()

        dag_str = "Read(path=/tmp)->Filter(condition=x>0)->Map(func=f)"
        structure = collector._extract_dag_structure(dag_str)

        assert "(" not in structure  # Parameters removed
        assert "Read" in structure
        assert "Filter" in structure

    def test_save_and_load_metrics(self):
        """Test metrics saving and loading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = RuntimeFeedbackCollector()
            collector.CACHE_DIR = Path(tmpdir)

            # Save
            metrics = {
                'op1': OpRuntimeMetrics(op_name='op1', actual_rows=1000),
            }
            entry = FeedbackCacheEntry(
                plan_hash='abc123',
                op_metrics=metrics,
                created_timestamp=time.time(),
                last_updated_timestamp=time.time(),
            )

            collector._save_to_disk('abc123', entry)

            # Load
            loaded_entry = collector._load_from_disk('abc123')
            assert loaded_entry is not None
            assert loaded_entry.plan_hash == 'abc123'

    def test_cache_expiration(self):
        """Test cache expiration mechanism"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = RuntimeFeedbackCollector()
            collector.CACHE_DIR = Path(tmpdir)
            collector.CACHE_RETENTION_DAYS = 0  # Expire immediately

            # Save
            metrics = {
                'op1': OpRuntimeMetrics(op_name='op1', actual_rows=1000),
            }
            entry = FeedbackCacheEntry(
                plan_hash='abc123',
                op_metrics=metrics,
                created_timestamp=time.time(),
                last_updated_timestamp=time.time(),
            )
            collector._save_to_disk('abc123', entry)

            # Wait to ensure timestamp differs
            time.sleep(0.01)

            # Loading should return None (expired)
            loaded_entry = collector._load_from_disk('abc123')
            assert loaded_entry is None

    def test_lru_eviction(self):
        """Test LRU eviction"""
        collector = RuntimeFeedbackCollector()
        collector.MAX_CACHE_SIZE = 2

        # Add 3 entries
        for i in range(3):
            entry = FeedbackCacheEntry(
                plan_hash=f'hash{i}',
                op_metrics={},
                created_timestamp=time.time(),
                last_updated_timestamp=time.time(),
            )
            collector._memory_cache[f'hash{i}'] = entry

        # Perform eviction
        collector._evict_lru()

        # Memory cache size should be <= MAX_CACHE_SIZE
        assert len(collector._memory_cache) <= collector.MAX_CACHE_SIZE

    def test_apply_feedback_with_ema(self):
        """Test exponential moving average feedback application"""
        collector = RuntimeFeedbackCollector()
        collector.EMA_ALPHA = 0.7

        class MockStats:
            def __init__(self):
                self.num_rows = 1000
                self.size_bytes = 1024 * 1024
                self.confidence = 0.5

        stats = MockStats()
        metrics = OpRuntimeMetrics(
            op_name='op1',
            actual_rows=1500,
            actual_bytes=1536 * 1024,
        )

        collector.apply_feedback_to_statistics(stats, metrics)

        # Verify EMA calculation
        expected_rows = int(0.7 * 1500 + 0.3 * 1000)
        assert stats.num_rows == expected_rows

        # Confidence should improve
        assert stats.confidence > 0.5

    def test_feedback_singleton(self):
        """Test global singleton access"""
        from ray.data._internal.cbo_stats.runtime_feedback_collector import (
            get_feedback_collector,
        )

        collector1 = get_feedback_collector()
        collector2 = get_feedback_collector()

        assert collector1 is collector2


class TestPhase4Integration:
    """Phase 4 integration tests"""

    def test_feedback_cycle(self):
        """Test complete feedback cycle: collect → save → load → apply"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = RuntimeFeedbackCollector()
            collector.CACHE_DIR = Path(tmpdir)

            # Step 1: Create and save feedback
            metrics = {
                'op1': OpRuntimeMetrics(
                    op_name='op1',
                    actual_rows=1000,
                    actual_bytes=1024 * 1024,
                ),
            }

            class MockPlan:
                class MockDAG:
                    dag_str = "Read->Filter"

                dag = MockDAG()

            plan = MockPlan()
            collector.save_feedback(plan, metrics)

            # Step 2: Load feedback
            loaded_metrics = collector.load_feedback(plan)
            assert loaded_metrics is not None
            assert 'op1' in loaded_metrics

            # Step 3: Apply feedback
            class MockStats:
                def __init__(self):
                    self.num_rows = 900
                    self.size_bytes = 900 * 1024
                    self.confidence = 0.3

            stats = MockStats()
            collector.apply_feedback_to_statistics(
                stats, loaded_metrics['op1']
            )

            # Verify: statistics should be updated
            assert stats.num_rows != 900
            assert stats.confidence > 0.3

    def test_cache_hit_improves_accuracy(self):
        """Test that cache hit improves accuracy"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = RuntimeFeedbackCollector()
            collector.CACHE_DIR = Path(tmpdir)

            # Simulate multiple feedback cycles
            class MockStats:
                def __init__(self):
                    self.num_rows = 1000
                    self.size_bytes = 1024 * 1024
                    self.confidence = 0.5

            actual_rows = 1500

            # First feedback
            stats = MockStats()
            metrics1 = OpRuntimeMetrics(
                op_name='op1',
                actual_rows=actual_rows,
                actual_bytes=1536 * 1024,
            )
            collector.apply_feedback_to_statistics(stats, metrics1)
            first_estimate = stats.num_rows
            first_confidence = stats.confidence

            # Second feedback
            stats = MockStats()
            stats.num_rows = first_estimate  # Use previous estimate as initial
            metrics2 = OpRuntimeMetrics(
                op_name='op1',
                actual_rows=actual_rows,
                actual_bytes=1536 * 1024,
            )
            collector.apply_feedback_to_statistics(stats, metrics2)
            second_estimate = stats.num_rows
            second_confidence = stats.confidence

            # Second should be closer to actual
            error1 = abs(first_estimate - actual_rows)
            error2 = abs(second_estimate - actual_rows)
            # Verify: second error should be smaller (or same)
            assert error2 <= error1 + 1  # Allow rounding error
            assert second_confidence > first_confidence

    def test_memory_cache_access_pattern(self):
        """Test memory cache access pattern"""
        collector = RuntimeFeedbackCollector()

        entry = FeedbackCacheEntry(
            plan_hash='hash1',
            op_metrics={},
            created_timestamp=time.time(),
            last_updated_timestamp=time.time(),
            access_count=0,
        )

        collector._memory_cache['hash1'] = entry

        # First access
        assert collector._memory_cache['hash1'].access_count == 0
        collector._memory_cache['hash1'].access_count += 1
        assert collector._memory_cache['hash1'].access_count == 1

        # Second access
        collector._memory_cache['hash1'].access_count += 1
        assert collector._memory_cache['hash1'].access_count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
