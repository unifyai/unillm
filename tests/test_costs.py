"""Tests for the cost computation module."""

import pytest
from unittest.mock import patch

import litellm

from unillm.costs import (
    _normalize_model_name,
    compute_cost,
    compute_cost_from_response,
    compute_full_cost_from_usage,
)
from unillm.endpoints import (
    anthropic,
    deepseek,
    minimax,
    moonshotai,
    openai,
    xiaomi_mimo,
    zai,
)


class TestNormalizeModelName:
    """Tests for the _normalize_model_name function."""

    def test_strips_provider_suffix(self):
        """Test that @provider suffix is stripped."""
        assert _normalize_model_name("gpt-5.2@openai") == "gpt-5.2"
        assert _normalize_model_name("gpt-4o@openai") == "gpt-4o"
        assert (
            _normalize_model_name("claude-3-5-sonnet@anthropic") == "claude-3-5-sonnet"
        )

    def test_preserves_model_without_suffix(self):
        """Test that models without @provider are unchanged."""
        assert _normalize_model_name("gpt-5.2") == "gpt-5.2"
        assert _normalize_model_name("gpt-4o") == "gpt-4o"
        assert (
            _normalize_model_name("claude-3-5-sonnet-20241022")
            == "claude-3-5-sonnet-20241022"
        )

    def test_handles_empty_string(self):
        """Test that empty string is handled."""
        assert _normalize_model_name("") == ""

    def test_handles_multiple_at_symbols(self):
        """Test that only first @ is used for splitting."""
        # Edge case: if model name somehow has multiple @, only split on first
        assert _normalize_model_name("model@provider@extra") == "model"


class TestComputeCostWithProviderSuffix:
    """Tests for cost computation with @provider suffix (unify/unillm format)."""

    def test_compute_cost_with_openai_suffix(self):
        """Test that gpt-5.2@openai works like gpt-5.2."""
        cost_with_suffix = compute_cost(
            "gpt-5.2@openai",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        cost_without_suffix = compute_cost(
            "gpt-5.2",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        assert cost_with_suffix == cost_without_suffix
        assert cost_with_suffix > 0

    def test_compute_cost_with_anthropic_suffix(self):
        """Test that claude model with @anthropic suffix works."""
        cost_with_suffix = compute_cost(
            "claude-sonnet-4-20250514@anthropic",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        cost_without_suffix = compute_cost(
            "claude-sonnet-4-20250514",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        assert cost_with_suffix == cost_without_suffix
        assert cost_with_suffix > 0

    def test_compute_cost_from_response_with_suffix(self):
        """Test compute_cost_from_response with @provider suffix."""
        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        }
        cost = compute_cost_from_response("gpt-5.2@openai", response)

        assert cost is not None
        assert cost > 0

    def test_compute_full_cost_with_suffix(self):
        """Test compute_full_cost_from_usage with @provider suffix."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        }
        cost = compute_full_cost_from_usage("gpt-5.2@openai", usage)

        assert cost > 0


class TestComputeCost:
    """Tests for the compute_cost function."""

    def test_compute_cost_gpt4o(self):
        """Test cost computation for gpt-4o model."""
        # gpt-4o: input=$2.50/M, output=$10.00/M
        cost = compute_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)

        # Expected: (1000 * 2.5e-6) + (500 * 1e-5) = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 1e-9

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

    def test_from_response_unknown_model_returns_none(self):
        """Test that unknown models return None instead of raising."""
        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        }
        # Should return None, not raise ValueError
        cost = compute_cost_from_response("non-existent-model-xyz-12345", response)
        assert cost is None

    def test_from_response_unknown_model_with_suffix_returns_none(self):
        """Test that unknown models with @provider suffix return None."""
        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        }
        cost = compute_cost_from_response("non-existent-model@openai", response)
        assert cost is None

    def test_from_response_uses_cached_prompt_token_pricing(self):
        """Cached prompt tokens should use LiteLLM's cache-read token rate."""
        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 900},
            },
        }

        cost = compute_cost_from_response("deepseek/deepseek-v4-pro", response)

        assert cost is not None
        expected = (
            100 * (0.435 / 1_000_000)
            + 900 * (0.003625 / 1_000_000)
            + 100 * (0.87 / 1_000_000)
        )
        assert abs(cost - expected) < 1e-12


class TestSupportedModelPricingCoverage:
    """Supported endpoint aliases should resolve to priced LiteLLM model entries."""

    PROVIDERS = {
        "openai": openai.models,
        "anthropic": anthropic.models,
        "deepseek": deepseek.models,
        "minimax": minimax.models,
        "moonshotai": moonshotai.models,
        "xiaomi-mimo": xiaomi_mimo.models,
        "zai": zai.models,
    }

    @pytest.mark.parametrize(
        ("provider", "model", "provider_model"),
        [
            (provider, model, provider_model)
            for provider, models in PROVIDERS.items()
            for model, provider_model in models.items()
        ],
    )
    def test_litellm_has_pricing_for_supported_model(
        self,
        provider,
        model,
        provider_model,
    ):
        info = litellm.get_model_info(provider_model)

        assert info.get("input_cost_per_token"), f"{model}@{provider} lacks input price"
        assert info.get(
            "output_cost_per_token",
        ), f"{model}@{provider} lacks output price"


class TestTieredLongContextPricing:
    """Tests for tiered pricing when prompts exceed a token threshold.

    Some providers charge higher rates for input/output tokens when the prompt
    exceeds a threshold (e.g. 200k tokens):
      - Input:  2x standard rate for tokens above the threshold
      - Output: 1.5x standard rate when prompt exceeds the threshold

    LiteLLM provides these rates as `input_cost_per_token_above_{N}k_tokens`
    and `output_cost_per_token_above_{N}k_tokens`. Our cost functions must
    use them for accurate billing.

    These tests use a fake model info dict to avoid depending on any real
    model's pricing data staying stable in LiteLLM.
    """

    FAKE_MODEL = "fake-tiered-model"
    FAKE_MODEL_INFO = {
        "input_cost_per_token": 1e-6,  # $1/M
        "output_cost_per_token": 2e-6,  # $2/M
        "input_cost_per_token_above_200k_tokens": 4e-6,  # $4/M
        "output_cost_per_token_above_200k_tokens": 8e-6,  # $8/M
    }

    @pytest.fixture(autouse=True)
    def _patch_model_info(self):
        with patch(
            "unillm.costs._get_model_info",
            return_value=self.FAKE_MODEL_INFO,
        ):
            yield

    @property
    def input_rate(self):
        return self.FAKE_MODEL_INFO["input_cost_per_token"]

    @property
    def output_rate(self):
        return self.FAKE_MODEL_INFO["output_cost_per_token"]

    @property
    def input_rate_tiered(self):
        return self.FAKE_MODEL_INFO["input_cost_per_token_above_200k_tokens"]

    @property
    def output_rate_tiered(self):
        return self.FAKE_MODEL_INFO["output_cost_per_token_above_200k_tokens"]

    def test_compute_cost_applies_tiered_input_pricing(self):
        """Input tokens above 200k should be billed at the higher rate."""
        cost = compute_cost(
            self.FAKE_MODEL,
            prompt_tokens=300_000,
            completion_tokens=0,
        )
        expected = (200_000 * self.input_rate) + (100_000 * self.input_rate_tiered)
        assert abs(cost - expected) < 1e-9, (
            f"Expected ${expected:.4f} (tiered) but got ${cost:.4f} "
            f"(flat rate would give ${300_000 * self.input_rate:.4f})"
        )

    def test_compute_cost_applies_tiered_output_pricing(self):
        """Output tokens should be billed at the higher rate when prompt > 200k."""
        cost = compute_cost(
            self.FAKE_MODEL,
            prompt_tokens=250_000,
            completion_tokens=10_000,
        )
        expected = (
            (200_000 * self.input_rate)
            + (50_000 * self.input_rate_tiered)
            + (10_000 * self.output_rate_tiered)
        )
        assert (
            abs(cost - expected) < 1e-9
        ), f"Expected ${expected:.4f} (tiered) but got ${cost:.4f}"

    def test_compute_cost_no_tiered_pricing_under_200k(self):
        """Requests under 200k should use standard rates only."""
        cost = compute_cost(
            self.FAKE_MODEL,
            prompt_tokens=150_000,
            completion_tokens=5_000,
        )
        expected = (150_000 * self.input_rate) + (5_000 * self.output_rate)
        assert abs(cost - expected) < 1e-9

    def test_compute_full_cost_tiered_pricing(self):
        """compute_full_cost_from_usage should also apply tiered pricing."""
        usage = {
            "prompt_tokens": 300_000,
            "completion_tokens": 10_000,
        }
        cost = compute_full_cost_from_usage(self.FAKE_MODEL, usage)
        expected = (
            (200_000 * self.input_rate)
            + (100_000 * self.input_rate_tiered)
            + (10_000 * self.output_rate_tiered)
        )
        assert (
            abs(cost - expected) < 1e-9
        ), f"Expected ${expected:.4f} (tiered) but got ${cost:.4f}"


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
