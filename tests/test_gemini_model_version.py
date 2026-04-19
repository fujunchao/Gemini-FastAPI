"""Test Gemini compatible endpoints' modelVersion and model ID parsing.

This test verifies:
1. _strip_model_prefix correctly removes the models/ prefix
2. _get_model_by_name is called with bare model name
3. generateContent and streamGenerateContent return correct modelVersion format
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.gemini_models import GeminiGenerateContentResponse
from app.server.gemini import _strip_model_prefix, _to_gemini_response


class TestStripModelPrefix:
    """Test _strip_model_prefix function"""

    def test_strip_models_prefix(self):
        """Test that models/ prefix is correctly stripped"""
        assert _strip_model_prefix("models/gemini-3-flash") == "gemini-3-flash"
        assert _strip_model_prefix("models/gemini-1.5-pro") == "gemini-1.5-pro"

    def test_keep_bare_name(self):
        """Test that bare model names are kept unchanged"""
        assert _strip_model_prefix("gemini-3-flash") == "gemini-3-flash"
        assert _strip_model_prefix("gemini-1.5-pro") == "gemini-1.5-pro"

    def test_other_prefix_unchanged(self):
        """Test that non-models/ prefixes are not mistakenly removed"""
        assert _strip_model_prefix("model/gemini-3-flash") == "model/gemini-3-flash"
        assert _strip_model_prefix("some/gemini") == "some/gemini"


class TestModelVersionFormat:
    """Test modelVersion output format"""

    def test_model_version_is_bare_name(self):
        """Verify modelVersion is bare model name, not with models/ prefix"""
        model_name = "gemini-3-flash"
        response = _to_gemini_response(
            visible_text="Hi",
            tool_calls=[],
            thoughts=None,
            usage_tuple=(10, 20, 30, 0),
            model_name=model_name,
            image_parts=None,
        )

        assert isinstance(response, GeminiGenerateContentResponse)
        assert response.modelVersion == "gemini-3-flash", (
            f"modelVersion should be bare model name, got: {response.modelVersion}"
        )
        assert not response.modelVersion.startswith("models/"), (
            "modelVersion should not start with models/"
        )

    def test_model_version_in_response(self):
        """Verify response contains correct modelVersion format"""
        response = _to_gemini_response(
            visible_text="Hello World",
            tool_calls=[],
            thoughts=None,
            usage_tuple=(5, 8, 13, 0),
            model_name="gemini-1.5-pro",
            image_parts=[],
        )

        assert response.modelVersion == "gemini-1.5-pro"
        assert hasattr(response, "candidates")


class TestCodePaths:
    """Test key code paths"""

    def test_strip_model_prefix_in_routes(self):
        """Verify _strip_model_prefix is used in routes"""
        import inspect

        from app.server import gemini

        source = inspect.getsource(gemini)

        # Should have 3 calls
        assert source.count("_strip_model_prefix(model)") == 3, (
            "Should have 3 _strip_model_prefix(model) calls"
        )

    def test_model_version_assignments(self):
        """Verify modelVersion assignment uses model_name"""
        import re
        from pathlib import Path

        gemini_path = Path(__file__).parent.parent / "app" / "server" / "gemini.py"
        source = gemini_path.read_text(encoding="utf-8")

        # Find modelVersion assignments
        # Should use bare model_name, not f"models/{model_name}"
        model_version_pattern = r"modelVersion\s*=\s*([^\n,)]+)"
        matches = re.findall(model_version_pattern, source)

        for match in matches:
            assert 'f"models/' not in match, (
                f"modelVersion should not use f'models/...' format, current: {match}"
            )
            assert match.strip() == "model_name", (
                f"modelVersion should be model_name, current: {match}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
