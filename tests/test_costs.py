"""Tests for the cost computation module."""

from unittest.mock import patch

import pytest

from unillm.costs import (
    compute_cost,
    compute_cost_from_response,
    compute_full_cost_from_usage,
    deduct_credits_for_usage,
)


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


class TestComputeFullCostFromUsage:
    """Tests for compute_full_cost_from_usage with audio tokens."""

    def test_standard_chat_completion_usage(self):
        """Test with standard prompt_tokens/completion_tokens format."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }
        cost = compute_full_cost_from_usage("gpt-4o", usage)

        # gpt-4o: input=$2.50/M, output=$10.00/M
        # Expected: (1000 * 2.5e-6) + (500 * 1e-5) = 0.0075
        assert abs(cost - 0.0075) < 1e-9

    def test_realtime_api_usage_with_audio_tokens(self):
        """Test with Realtime API format including audio tokens."""
        usage = {
            "total_tokens": 275,
            "input_tokens": 127,
            "output_tokens": 148,
            "input_token_details": {
                "text_tokens": 100,
                "audio_tokens": 27,
            },
            "output_token_details": {
                "text_tokens": 48,
                "audio_tokens": 100,
            },
        }
        cost = compute_full_cost_from_usage("gpt-4o-realtime-preview", usage)

        # gpt-4o-realtime-preview:
        # text: input=$5/M, output=$20/M
        # audio: input=$40/M, output=$80/M
        # Expected:
        #   text: (100 * 5e-6) + (48 * 2e-5) = 0.0005 + 0.00096 = 0.00146
        #   audio: (27 * 4e-5) + (100 * 8e-5) = 0.00108 + 0.008 = 0.00908
        #   total: 0.00146 + 0.00908 = 0.01054
        assert abs(cost - 0.01054) < 1e-6

    def test_realtime_api_usage_audio_only(self):
        """Test Realtime API with only audio tokens."""
        usage = {
            "input_token_details": {
                "text_tokens": 0,
                "audio_tokens": 1000,
            },
            "output_token_details": {
                "text_tokens": 0,
                "audio_tokens": 500,
            },
        }
        cost = compute_full_cost_from_usage("gpt-4o-realtime-preview", usage)

        # audio: (1000 * 4e-5) + (500 * 8e-5) = 0.04 + 0.04 = 0.08
        assert abs(cost - 0.08) < 1e-9

    def test_input_output_tokens_format(self):
        """Test with input_tokens/output_tokens format (alternative to prompt/completion)."""
        usage = {
            "input_tokens": 2000,
            "output_tokens": 1000,
        }
        cost = compute_full_cost_from_usage("gpt-4o", usage)

        # gpt-4o: input=$2.50/M, output=$10.00/M
        # Expected: (2000 * 2.5e-6) + (1000 * 1e-5) = 0.005 + 0.01 = 0.015
        assert abs(cost - 0.015) < 1e-9

    def test_usage_as_object(self):
        """Test with usage as an object with attributes."""

        class MockTokenDetails:
            text_tokens = 50
            audio_tokens = 200

        class MockUsage:
            input_token_details = MockTokenDetails()
            output_token_details = MockTokenDetails()

        cost = compute_full_cost_from_usage("gpt-4o-realtime-preview", MockUsage())

        # text: (50 * 5e-6) + (50 * 2e-5) = 0.00025 + 0.001 = 0.00125
        # audio: (200 * 4e-5) + (200 * 8e-5) = 0.008 + 0.016 = 0.024
        # total: 0.00125 + 0.024 = 0.02525
        assert abs(cost - 0.02525) < 1e-9

    def test_unknown_model_raises(self):
        """Test that unknown models raise ValueError."""
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        with pytest.raises(ValueError, match="Could not find pricing info"):
            compute_full_cost_from_usage("non-existent-model-xyz", usage)


class TestDeductCreditsForUsage:
    """Tests for the deduct_credits_for_usage function."""

    @patch("unillm.costs.unify.deduct_credits")
    def test_deducts_credits_for_chat_completion(self, mock_deduct):
        """Test that credits are deducted for a standard chat completion."""
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        }
        cost = deduct_credits_for_usage("gpt-4o", response)

        # Verify cost was computed correctly
        assert abs(cost - 0.0075) < 1e-9

        # Verify deduct_credits was called with the cost
        mock_deduct.assert_called_once()
        call_args = mock_deduct.call_args[0][0]
        assert abs(call_args - 0.0075) < 1e-9

    @patch("unillm.costs.unify.deduct_credits")
    def test_deducts_credits_for_realtime_audio(self, mock_deduct):
        """Test that credits are deducted for Realtime API with audio."""
        response = {
            "usage": {
                "input_token_details": {
                    "text_tokens": 100,
                    "audio_tokens": 1000,
                },
                "output_token_details": {
                    "text_tokens": 50,
                    "audio_tokens": 500,
                },
            },
        }
        cost = deduct_credits_for_usage("gpt-4o-realtime-preview", response)

        # text: (100 * 5e-6) + (50 * 2e-5) = 0.0005 + 0.001 = 0.0015
        # audio: (1000 * 4e-5) + (500 * 8e-5) = 0.04 + 0.04 = 0.08
        # total: 0.0015 + 0.08 = 0.0815
        assert abs(cost - 0.0815) < 1e-6
        mock_deduct.assert_called_once()

    @patch("unillm.costs.unify.deduct_credits")
    def test_deducts_credits_from_response_object(self, mock_deduct):
        """Test with a response object (not dict)."""

        class MockUsage:
            prompt_tokens = 2000
            completion_tokens = 1000

            def model_dump(self):
                return {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                }

        class MockResponse:
            usage = MockUsage()

        cost = deduct_credits_for_usage("gpt-4o", MockResponse())

        # Expected: (2000 * 2.5e-6) + (1000 * 1e-5) = 0.015
        assert abs(cost - 0.015) < 1e-9
        mock_deduct.assert_called_once()

    @patch("unillm.costs.unify.deduct_credits")
    def test_zero_cost_does_not_deduct(self, mock_deduct):
        """Test that zero cost doesn't call deduct_credits."""
        response = {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }
        cost = deduct_credits_for_usage("gpt-4o", response)

        assert cost == 0
        mock_deduct.assert_not_called()

    def test_missing_usage_raises(self):
        """Test that missing usage raises ValueError."""
        response = {"id": "chatcmpl-123", "model": "gpt-4o"}
        with pytest.raises(ValueError, match="No usage information found"):
            deduct_credits_for_usage("gpt-4o", response)

    def test_none_usage_raises(self):
        """Test that None usage raises ValueError."""

        class MockResponse:
            usage = None

        with pytest.raises(ValueError, match="No usage information found"):
            deduct_credits_for_usage("gpt-4o", MockResponse())

    def test_unknown_model_raises(self):
        """Test that unknown models raise ValueError."""
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        }
        with pytest.raises(ValueError, match="Could not find pricing info"):
            deduct_credits_for_usage("non-existent-model-xyz", response)

    @patch("unillm.costs.unify.deduct_credits")
    def test_returns_cost_value(self, mock_deduct):
        """Test that the function returns the computed cost."""
        response = {
            "usage": {
                "prompt_tokens": 1000000,  # 1M tokens
                "completion_tokens": 0,
            },
        }
        cost = deduct_credits_for_usage("gpt-4o", response)

        # gpt-4o: input=$2.50/M
        assert abs(cost - 2.5) < 1e-9

    @patch("unillm.costs.unify.deduct_credits")
    def test_handles_nested_object_usage(self, mock_deduct):
        """Test with deeply nested object structure."""

        class InputDetails:
            text_tokens = 100
            audio_tokens = 500

        class OutputDetails:
            text_tokens = 50
            audio_tokens = 250

        class Usage:
            input_token_details = InputDetails()
            output_token_details = OutputDetails()

            def model_dump(self):
                return {
                    "input_token_details": {
                        "text_tokens": 100,
                        "audio_tokens": 500,
                    },
                    "output_token_details": {
                        "text_tokens": 50,
                        "audio_tokens": 250,
                    },
                }

        class Response:
            usage = Usage()

        cost = deduct_credits_for_usage("gpt-4o-realtime-preview", Response())

        # text: (100 * 5e-6) + (50 * 2e-5) = 0.0005 + 0.001 = 0.0015
        # audio: (500 * 4e-5) + (250 * 8e-5) = 0.02 + 0.02 = 0.04
        # total: 0.0015 + 0.04 = 0.0415
        assert abs(cost - 0.0415) < 1e-6
        mock_deduct.assert_called_once()
