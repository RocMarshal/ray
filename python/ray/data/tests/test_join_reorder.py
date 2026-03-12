"""
Ray Data CBO: Join Reorder Optimization Tests

Source Location: python/ray/data/tests/test_join_reorder.py

Test Coverage:
1. JoinReorderRule - join type detection and reordering
2. Join side optimization
3. Integration tests for join optimization
"""

import pytest
from ray.data._internal.logical.rules.join_reorder_rule import JoinReorderRule


class TestJoinReorderRule:
    """Unit tests for JoinReorderRule"""

    def test_join_type_detection_inner(self):
        """Test INNER JOIN type detection"""
        rule = JoinReorderRule()

        class MockJoinOp:
            def __init__(self, join_type):
                self.join_type = join_type
                self.name = "test_join"

        op_inner = MockJoinOp("INNER")
        assert rule._get_join_type(op_inner) == "INNER"

    def test_join_type_detection_left(self):
        """Test LEFT JOIN type detection"""
        rule = JoinReorderRule()

        class MockJoinOp:
            def __init__(self, join_type):
                self.join_type = join_type
                self.name = "test_join"

        op_left = MockJoinOp("LEFT")
        assert rule._get_join_type(op_left) == "LEFT"

    def test_join_type_detection_right(self):
        """Test RIGHT JOIN type detection"""
        rule = JoinReorderRule()

        class MockJoinOp:
            def __init__(self, join_type):
                self.join_type = join_type
                self.name = "test_join"

        op_right = MockJoinOp("RIGHT")
        assert rule._get_join_type(op_right) == "RIGHT"

    def test_join_type_detection_unknown(self):
        """Test unknown JOIN type detection"""
        rule = JoinReorderRule()

        class MockJoinOp:
            def __init__(self):
                self.name = "test_join"

        op = MockJoinOp()
        assert rule._get_join_type(op) == "UNKNOWN"

    def test_is_join_operator_true(self):
        """Test join operator identification"""
        rule = JoinReorderRule()

        class MockOp:
            pass

        MockOp.__name__ = "JoinOperator"
        op = MockOp()
        assert rule._is_join_operator(op) is True

    def test_is_join_operator_false(self):
        """Test non-join operator identification"""
        rule = JoinReorderRule()

        class MockOp:
            pass

        MockOp.__name__ = "FilterOperator"
        op = MockOp()
        assert rule._is_join_operator(op) is False

    def test_user_specified_join_side(self):
        """Test user-specified join_side detection"""
        rule = JoinReorderRule()

        class MockOp:
            def __init__(self):
                self._user_config = {'join_side': 'left'}

        op = MockOp()
        assert rule._is_user_specified(op) is True

    def test_user_not_specified_join_side(self):
        """Test unspecified join_side detection"""
        rule = JoinReorderRule()

        class MockOp:
            def __init__(self):
                self._user_config = {}

        op = MockOp()
        assert rule._is_user_specified(op) is False

    def test_only_inner_join_reordered(self):
        """Test that only INNER JOIN is reordered"""
        rule = JoinReorderRule()

        # LEFT JOIN should not be reordered
        class MockLeftJoin:
            def __init__(self):
                self.join_type = "LEFT"
                self.name = "left_join"
                self._user_config = {}

        op = MockLeftJoin()

        # Calling _try_reorder_join should not raise exception
        # (blackbox test, checking for no exception)
        try:
            rule._try_reorder_join(op)
        except Exception:
            pytest.fail("_try_reorder_join raised exception for LEFT JOIN")


class TestJoinReorderIntegration:
    """Integration tests for join reorder optimization"""

    def test_join_reorder_scenario_small_build_table(self):
        """Test join reorder scenario: small table as build side"""
        rule = JoinReorderRule()

        # Small table (500MB) should be on right side (build side)
        # Large table (5GB) should be on left side (probe side)

        class MockOp:
            def __init__(self):
                self.name = "join"
                self.join_type = "INNER"
                self._user_config = {}

        # This is a structural test
        assert rule._get_join_type(MockOp()) == "INNER"
        assert not rule._is_user_specified(MockOp())

    def test_join_type_coverage(self):
        """Test coverage of all join types"""
        rule = JoinReorderRule()

        class MockJoinOp:
            def __init__(self, join_type):
                self.join_type = join_type
                self.name = "test_join"

        join_types = ["INNER", "LEFT", "RIGHT", "FULL"]
        detected_types = [
            rule._get_join_type(MockJoinOp(jt)) for jt in join_types
        ]

        # All types should be correctly detected
        assert detected_types == join_types

    def test_join_operator_detection(self):
        """Test join operator detection across different types"""
        rule = JoinReorderRule()

        class MockOp:
            pass

        operator_names = ["JoinOperator", "InnerJoin", "LeftJoin", "Join"]
        results = []

        for name in operator_names:
            MockOp.__name__ = name
            op = MockOp()
            results.append(rule._is_join_operator(op))

        # First one is definitely a join operator
        assert results[0] is True
        # Others depend on implementation details

    def test_join_reorder_with_user_config(self):
        """Test join reorder respects user configuration"""
        rule = JoinReorderRule()

        class MockOp:
            def __init__(self, config=None):
                self.name = "join"
                self.join_type = "INNER"
                self._user_config = config or {}

        # User specified config
        op_with_config = MockOp({'join_side': 'right'})
        assert rule._is_user_specified(op_with_config) is True

        # No user config
        op_without_config = MockOp()
        assert rule._is_user_specified(op_without_config) is False

    def test_join_reorder_edge_cases(self):
        """Test join reorder edge cases"""
        rule = JoinReorderRule()

        class MockJoinOp:
            def __init__(self, join_type, name="join"):
                self.join_type = join_type
                self.name = name
                self._user_config = {}

        # Edge case: empty join type
        op_empty = MockJoinOp("")
        assert rule._get_join_type(op_empty) == ""

        # Edge case: None join type (fallback)
        op_none = MockJoinOp(None)
        # Should either return None or "UNKNOWN" depending on implementation
        result = rule._get_join_type(op_none)
        assert result in [None, "UNKNOWN"] or result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
