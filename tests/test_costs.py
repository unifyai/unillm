"""Tests for the cost computation module."""

import pytest

from unillm.costs import compute_cost, compute_cost_from_response


class TestComputeCost:
    """Tests for the compute_cost function."""

    def test_compute_cost_gpt4o(self):
        """Test cost computation for gpt-4o model."""
        # gpt-4o: input=$2.50/M, output=$10.00/M
        cost = compute_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)

        # Expected: (1000 * 2.5e-6) + (500 * 1e-5) = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 1e-9

    def test_compute_cost_gpt4o_mini(self):
        """Test cost computation for gpt-4o-mini model."""
        # gpt-4o-mini: input=$0.15/M, output=$0.60/M
        cost = compute_cost("gpt-4o-mini", prompt_tokens=10000, completion_tokens=5000)

        # Expected: (10000 * 1.5e-7) + (5000 * 6e-7) = 0.0015 + 0.003 = 0.0045
        assert abs(cost - 0.0045) < 1e-9

    def test_compute_cost_claude_sonnet(self):
        """Test cost computation for Claude model."""
        # claude-3-5-sonnet: input=$3.00/M, output=$15.00/M
        cost = compute_cost(
            "claude-3-5-sonnet-20241022",
            prompt_tokens=2000,
            completion_tokens=1000,
        )

        # Expected: (2000 * 3e-6) + (1000 * 1.5e-5) = 0.006 + 0.015 = 0.021
        assert abs(cost - 0.021) < 1e-9

    def test_compute_cost_zero_tokens(self):
        """Test cost computation with zero tokens."""
        cost = compute_cost("gpt-4o", prompt_tokens=0, completion_tokens=0)
        assert cost == 0

    def test_compute_cost_unknown_model(self):
        """Test that unknown models raise ValueError."""
        with pytest.raises(ValueError, match="Could not find pricing info"):
            compute_cost(
                "non-existent-model-xyz-12345",
                prompt_tokens=100,
                completion_tokens=50,
            )


class TestComputeCostFromResponse:
    """Tests for the compute_cost_from_response function."""

    def test_from_response_dict(self):
        """Test cost computation from a dict response."""
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        }
        cost = compute_cost_from_response("gpt-4o", response)

        # Expected: (1000 * 2.5e-6) + (500 * 1e-5) = 0.0075
        assert cost is not None
        assert abs(cost - 0.0075) < 1e-9

    def test_from_response_object(self):
        """Test cost computation from an object with usage attribute."""

        class MockUsage:
            prompt_tokens = 2000
            completion_tokens = 1000

        class MockResponse:
            usage = MockUsage()

        cost = compute_cost_from_response("gpt-4o", MockResponse())

        # Expected: (2000 * 2.5e-6) + (1000 * 1e-5) = 0.005 + 0.01 = 0.015
        assert cost is not None
        assert abs(cost - 0.015) < 1e-9

    def test_from_response_missing_usage(self):
        """Test that missing usage returns None."""
        response = {"id": "chatcmpl-123", "model": "gpt-4o"}
        cost = compute_cost_from_response("gpt-4o", response)
        assert cost is None

    def test_from_response_empty_usage(self):
        """Test that empty usage returns None."""
        response = {"id": "chatcmpl-123", "model": "gpt-4o", "usage": {}}
        cost = compute_cost_from_response("gpt-4o", response)
        assert cost is None

    def test_from_response_zero_tokens(self):
        """Test that zero tokens returns None."""
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }
        cost = compute_cost_from_response("gpt-4o", response)
        assert cost is None
