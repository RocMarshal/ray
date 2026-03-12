"""
Ray Data CBO: Shuffle Partition Derivation Tests

Source Location: python/ray/data/tests/test_shuffle_partition_derivation.py

Test Coverage:
1. DeriveShufflePartitionsRule - partition number calculation
2. Edge cases and boundary conditions
3. Integration tests for shuffle optimization
"""

import pytest
from ray.data._internal.logical.rules.derive_shuffle_partitions_rule import (
    DeriveShufflePartitionsRule,
)


class TestDeriveShufflePartitionsRule:
    """Unit tests for DeriveShufflePartitionsRule"""

    def test_calculate_num_partitions_basic(self):
        """Test basic partition number calculation"""
        # 10GB data, target 512MB partition -> 20 partitions
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=10 * 1024 * 1024 * 1024,  # 10GB
        )
        assert num_partitions == 20

    def test_calculate_num_partitions_small_data(self):
        """Test partition calculation with small data"""
        # 100MB data -> minimum partition count 10
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=100 * 1024 * 1024,  # 100MB
        )
        assert num_partitions == DeriveShufflePartitionsRule.MIN_PARTITIONS

    def test_calculate_num_partitions_large_data(self):
        """Test partition calculation with large data"""
        # 5TB data, target 512MB partition -> 10000 partitions (max limit)
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=5 * 1024 * 1024 * 1024 * 1024,  # 5TB
        )
        assert num_partitions == DeriveShufflePartitionsRule.MAX_PARTITIONS

    def test_calculate_num_partitions_zero_bytes(self):
        """Test partition calculation with zero bytes"""
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=0,
        )
        assert num_partitions == DeriveShufflePartitionsRule.MIN_PARTITIONS

    def test_calculate_num_partitions_none_bytes(self):
        """Test partition calculation with None input"""
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=None,
        )
        assert num_partitions == DeriveShufflePartitionsRule.MIN_PARTITIONS

    def test_calculate_num_partitions_custom_target_size(self):
        """Test partition calculation with custom target size"""
        # 1GB data, target 256MB partition -> 4 partitions
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=1 * 1024 * 1024 * 1024,  # 1GB
            target_partition_size_bytes=256 * 1024 * 1024,  # 256MB
        )
        assert num_partitions == 4

    def test_calculate_num_partitions_custom_bounds(self):
        """Test partition calculation with custom min/max bounds"""
        # 100MB data, target 512MB, but min 5, max 15
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=100 * 1024 * 1024,  # 100MB
            min_partitions=5,
            max_partitions=15,
        )
        assert num_partitions == 5

    def test_calculate_num_partitions_exact_target(self):
        """Test partition calculation with exact target size match"""
        # 5 × 512MB = 2.56GB, should get 5 partitions
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=5 * 512 * 1024 * 1024,  # 5 × 512MB
        )
        assert num_partitions == 5

    def test_calculate_num_partitions_rounding(self):
        """Test partition count rounding up"""
        # 2.5GB, target 512MB -> ceil(2.5 / 0.5) = 5
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=int(2.5 * 1024 * 1024 * 1024),
        )
        assert num_partitions == 5

    def test_partition_size_calculation(self):
        """Test actual partition size calculation"""
        # 10GB, derived 20 partitions -> avg 512MB/partition
        total_bytes = 10 * 1024 * 1024 * 1024
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=total_bytes,
        )

        avg_partition_size = total_bytes / num_partitions
        target_size = DeriveShufflePartitionsRule.TARGET_PARTITION_SIZE_BYTES

        # Actual partition size should be close to target (error < 1%)
        assert abs(avg_partition_size - target_size) / target_size < 0.01


class TestShufflePartitionIntegration:
    """Integration tests for shuffle partition derivation"""

    def test_shuffle_partitions_standard_case(self):
        """Test standard shuffle partition derivation scenario"""
        # 10GB join operation
        total_bytes = 10 * 1024 * 1024 * 1024
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=total_bytes,
        )

        # Should derive 20 partitions
        assert num_partitions == 20

        # Each partition avg 512MB
        avg_partition_size = total_bytes / num_partitions
        expected_size = 512 * 1024 * 1024
        assert abs(avg_partition_size - expected_size) < 1024  # Error < 1KB

    def test_shuffle_partitions_small_data(self):
        """Test shuffle partition derivation with small data"""
        # 50MB data should use minimum partition count
        total_bytes = 50 * 1024 * 1024
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=total_bytes,
        )

        assert num_partitions == 10

    def test_shuffle_partitions_large_data(self):
        """Test shuffle partition derivation with large data"""
        # 10TB data should be limited to max partition count
        total_bytes = 10 * 1024 * 1024 * 1024 * 1024
        num_partitions = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=total_bytes,
        )

        assert num_partitions == 10000

    def test_partition_range_enforcement(self):
        """Test partition count range enforcement"""
        # Extremely small data -> minimum partition count
        small = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=1024,  # 1KB
        )
        assert small == DeriveShufflePartitionsRule.MIN_PARTITIONS

        # Extremely large data -> maximum partition count
        large = DeriveShufflePartitionsRule.calculate_num_partitions(
            total_bytes=1024 ** 5,  # 1 PB
        )
        assert large == DeriveShufflePartitionsRule.MAX_PARTITIONS

    def test_partition_calculation_edge_cases(self):
        """Test partition calculation edge cases"""
        cases = [
            # (input_bytes, expected_partitions)
            (0, 10),  # Zero bytes -> minimum partition
            (1024, 10),  # 1KB -> minimum partition
            (512 * 1024 * 1024, 1),  # 512MB -> 1 partition, limited by min -> 10
            (10 * 1024 * 1024 * 1024, 20),  # 10GB -> 20 partitions
            (5 * 1024 * 1024 * 1024 * 1024, 10000),  # 5TB -> 10000 (max)
        ]

        for input_bytes, expected in cases:
            result = DeriveShufflePartitionsRule.calculate_num_partitions(
                total_bytes=input_bytes,
            )
            assert result == expected, \
                f"Failed for {input_bytes} bytes: got {result}, expected {expected}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
